"""In-process job queue for asynchronous extraction.

A single invoice takes tens of seconds. Holding an HTTP connection open that
long fails against the 60-second idle timeouts most load balancers ship with,
and gives the caller nothing to poll. Clients submit a job, get an id
immediately, and poll for the result.

Scope: this queue is per-process and in-memory. It is the right size for a
pharmacy or a clinic running one container. Multiple replicas or durability
across restarts need an external broker — see the deployment notes in README.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pharmacy_ocr.domain.errors import (
    BatchNotFoundError,
    EmptyBatchError,
    JobNotFoundError,
    ServiceOverloadedError,
)
from pharmacy_ocr.domain.models import InvoiceResult
from pharmacy_ocr.observability import metrics
from pharmacy_ocr.observability.logging import bind_request_id
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import ExtractionMode, Settings

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL_S = 60.0


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


@dataclass
class Job:
    """One queued extraction. `path` is internal and never serialized."""

    id: str
    filename: str
    status: JobStatus = JobStatus.queued
    submitted_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    result: InvoiceResult | None = None
    error: str | None = None
    path: Path | None = None
    photo_mode: bool | None = None
    extraction_mode: ExtractionMode | None = None
    batch_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.succeeded, JobStatus.failed)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 2)


@dataclass(frozen=True)
class PendingUpload:
    """One accepted file on its way into a batch."""

    filename: str
    path: Path


@dataclass
class Batch:
    """A folder upload: several jobs submitted and polled as a unit."""

    id: str
    job_ids: tuple[str, ...]
    submitted_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


class JobManager:
    """Bounded worker pool over an asyncio queue."""

    def __init__(self, pipeline: InvoicePipeline, settings: Settings) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._batches: dict[str, Batch] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.job_queue_max_size)
        self._workers: list[asyncio.Task[None]] = []
        self._cleaner: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._running = False

    # ------------------------------------------------------------ lifecycle --
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._settings.max_concurrent_extractions):
            self._workers.append(asyncio.create_task(self._worker(index), name=f"ocr-worker-{index}"))
        self._cleaner = asyncio.create_task(self._cleanup_loop(), name="job-cleaner")
        logger.info("job manager started", extra={"workers": len(self._workers)})

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        tasks = [*self._workers, *(task for task in [self._cleaner] if task)]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._cleaner = None

        # Anything still queued will never run; do not leave its upload behind.
        for job in self._jobs.values():
            self._discard_upload(job)
        logger.info("job manager stopped")

    # --------------------------------------------------------------- public --
    async def submit(
        self,
        *,
        filename: str,
        path: Path,
        photo_mode: bool | None = None,
        extraction_mode: ExtractionMode | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            filename=filename,
            path=path,
            photo_mode=photo_mode,
            extraction_mode=extraction_mode,
        )
        async with self._lock:
            self._jobs[job.id] = job
        try:
            self._queue.put_nowait(job.id)
        except asyncio.QueueFull as exc:
            async with self._lock:
                self._jobs.pop(job.id, None)
            self._discard_upload(job)
            raise ServiceOverloadedError(
                "Extraction queue is full; retry shortly.",
                detail={"queue_size": self._settings.job_queue_max_size},
            ) from exc

        metrics.JOBS_QUEUED.set(self._queue.qsize())
        logger.info("job queued", extra={"job_id": job.id, "file": filename})
        return job

    async def submit_batch(
        self,
        uploads: list[PendingUpload],
        *,
        photo_mode: bool | None = None,
        extraction_mode: ExtractionMode | None = None,
    ) -> tuple[Batch, list[Job]]:
        """Queue a whole folder as one unit.

        All-or-nothing: capacity is checked before anything is enqueued, so a
        caller never has to reconcile a batch that was half accepted. On
        rejection the caller still owns every upload and must delete them.
        """
        if not uploads:
            raise EmptyBatchError("A batch needs at least one file.")

        jobs = [
            Job(
                id=uuid.uuid4().hex,
                filename=upload.filename,
                path=upload.path,
                photo_mode=photo_mode,
                extraction_mode=extraction_mode,
            )
            for upload in uploads
        ]
        batch = Batch(id=uuid.uuid4().hex, job_ids=tuple(job.id for job in jobs))
        for job in jobs:
            job.batch_id = batch.id

        async with self._lock:
            # The queue only drains while we hold the lock and never await, so
            # a free-slot count taken here cannot go stale before the puts.
            free = self._settings.job_queue_max_size - self._queue.qsize()
            if len(jobs) > free:
                raise ServiceOverloadedError(
                    "Not enough queue capacity for this batch; retry shortly or split it.",
                    detail={"requested": len(jobs), "available": max(free, 0)},
                )
            for job in jobs:
                self._jobs[job.id] = job
                self._queue.put_nowait(job.id)
            self._batches[batch.id] = batch

        metrics.JOBS_QUEUED.set(self._queue.qsize())
        logger.info("batch queued", extra={"batch_id": batch.id, "files": len(jobs)})
        return batch, jobs

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown or expired job '{job_id}'.")
        return job

    def get_batch(self, batch_id: str) -> tuple[Batch, list[Job]]:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise BatchNotFoundError(f"Unknown or expired batch '{batch_id}'.")
        return batch, [self._jobs[job_id] for job_id in batch.job_ids if job_id in self._jobs]

    def stats(self) -> dict[str, int]:
        counts = dict.fromkeys((status.value for status in JobStatus), 0)
        for job in self._jobs.values():
            counts[job.status.value] += 1
        counts["total"] = len(self._jobs)
        return counts

    # -------------------------------------------------------------- workers --
    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            metrics.JOBS_QUEUED.set(self._queue.qsize())
            job = self._jobs.get(job_id)
            try:
                if job is not None and job.path is not None:
                    await self._run_job(job)
            except asyncio.CancelledError:
                if job is not None:
                    job.status = JobStatus.failed
                    job.error = "service shut down before the job completed"
                    job.finished_at = dt.datetime.now(dt.UTC)
                raise
            except Exception:
                logger.exception("job worker error", extra={"job_id": job_id, "worker": index})
            finally:
                self._queue.task_done()

    async def _run_job(self, job: Job) -> None:
        job.status = JobStatus.running
        job.started_at = dt.datetime.now(dt.UTC)
        try:
            with bind_request_id(job.id):
                result = await self._pipeline.process(
                    job.path,  # type: ignore[arg-type]
                    photo_mode=job.photo_mode,
                    extraction_mode=job.extraction_mode,
                )
            result.source_file = job.filename
            job.result = result
            job.status = JobStatus.failed if result.error else JobStatus.succeeded
            job.error = result.error
        except Exception as exc:
            logger.exception("job failed", extra={"job_id": job.id})
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = dt.datetime.now(dt.UTC)
            self._discard_upload(job)
            logger.info(
                "job finished",
                extra={"job_id": job.id, "status": job.status.value, "seconds": job.duration_s},
            )

    # -------------------------------------------------------------- cleanup --
    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL_S)
            try:
                self.purge_expired()
            except Exception:
                logger.exception("job cleanup failed")

    def purge_expired(self) -> int:
        """Forget terminal jobs older than the retention window.

        A batch expires as a unit. Dropping its members one at a time would let
        a client poll the batch and see rows silently disappear, which reads as
        data loss rather than expiry.
        """
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=self._settings.job_retention_minutes)

        def expired(job: Job) -> bool:
            return job.is_terminal and (job.finished_at or job.submitted_at) < cutoff

        doomed = {job_id for job_id, job in self._jobs.items() if job.batch_id is None and expired(job)}

        for batch_id, batch in list(self._batches.items()):
            members = [self._jobs[job_id] for job_id in batch.job_ids if job_id in self._jobs]
            if members and all(expired(job) for job in members):
                doomed.update(job.id for job in members)
                del self._batches[batch_id]

        for job_id in doomed:
            job = self._jobs.pop(job_id, None)
            if job is not None:
                self._discard_upload(job)
        if doomed:
            logger.info("purged expired jobs", extra={"count": len(doomed)})
        return len(doomed)

    @staticmethod
    def _discard_upload(job: Job) -> None:
        if job.path is None:
            return
        try:
            job.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete job upload", extra={"job_id": job.id})
        job.path = None
