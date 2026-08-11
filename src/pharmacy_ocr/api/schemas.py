"""Request and response models for the HTTP layer.

Kept separate from `domain.models` so the wire format can evolve without
changing the pipeline's internal types.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from pharmacy_ocr.domain.models import InvoiceResult
from pharmacy_ocr.jobs.manager import Batch, Job, JobStatus


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    detail: Any | None = Field(default=None, description="Structured context, when available.")


class ErrorResponse(BaseModel):
    """Every non-2xx response from this service uses this shape."""

    error: ErrorDetail
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class JobSummary(BaseModel):
    """Job state without the extraction payload."""

    id: str
    filename: str
    status: JobStatus
    submitted_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_s: float | None = None
    error: str | None = None

    @classmethod
    def from_job(cls, job: Job) -> JobSummary:
        return cls(
            id=job.id,
            filename=job.filename,
            status=job.status,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration_s=job.duration_s,
            error=job.error,
        )


class JobResponse(JobSummary):
    """Job state plus the result once it is available."""

    result: InvoiceResult | None = None

    @classmethod
    def from_job(cls, job: Job) -> JobResponse:
        summary = JobSummary.from_job(job)
        return cls(**summary.model_dump(), result=job.result)


class JobStatsResponse(BaseModel):
    counts: dict[str, int]


class RejectedFile(BaseModel):
    """A file in a folder upload that never became a job."""

    filename: str
    code: str
    message: str


def _batch_status(jobs: list[Job]) -> str:
    if not jobs or all(job.is_terminal for job in jobs):
        return "completed"
    if any(job.status is not JobStatus.queued for job in jobs):
        return "running"
    return "queued"


def _batch_counts(jobs: list[Job]) -> dict[str, int]:
    counts = dict.fromkeys((status.value for status in JobStatus), 0)
    for job in jobs:
        counts[job.status.value] += 1
    counts["total"] = len(jobs)
    return counts


class BatchAcceptedResponse(BaseModel):
    """Answer to a folder upload: what was queued and what was turned away."""

    id: str
    submitted_at: dt.datetime
    accepted: int = Field(description="Files queued for extraction.")
    rejected: list[RejectedFile] = Field(
        default_factory=list,
        description="Files skipped, with the reason. The batch still runs.",
    )
    jobs: list[JobSummary] = Field(default_factory=list)

    @classmethod
    def build(cls, batch: Batch, jobs: list[Job], rejected: list[RejectedFile]) -> BatchAcceptedResponse:
        return cls(
            id=batch.id,
            submitted_at=batch.submitted_at,
            accepted=len(jobs),
            rejected=rejected,
            jobs=[JobSummary.from_job(job) for job in jobs],
        )


class BatchStatusResponse(BaseModel):
    """Cheap poll target: progress without the extraction payloads."""

    id: str
    status: str = Field(description="queued, running, or completed.")
    submitted_at: dt.datetime
    counts: dict[str, int]
    jobs: list[JobSummary]

    @classmethod
    def build(cls, batch: Batch, jobs: list[Job]) -> BatchStatusResponse:
        return cls(
            id=batch.id,
            status=_batch_status(jobs),
            submitted_at=batch.submitted_at,
            counts=_batch_counts(jobs),
            jobs=[JobSummary.from_job(job) for job in jobs],
        )


class BatchResultsResponse(BaseModel):
    """Every extraction in the batch, once you are done polling."""

    id: str
    status: str
    counts: dict[str, int]
    results: list[JobResponse]

    @classmethod
    def build(cls, batch: Batch, jobs: list[Job]) -> BatchResultsResponse:
        return cls(
            id=batch.id,
            status=_batch_status(jobs),
            counts=_batch_counts(jobs),
            results=[JobResponse.from_job(job) for job in jobs],
        )
