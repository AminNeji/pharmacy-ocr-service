"""Folder upload: submit many invoices in one request, poll them as one unit.

The single-file endpoints work fine in a loop, but a client scanning a shared
drive would then have to track N job ids itself and re-derive when the set is
done. A batch gives it one id to poll and one place to collect results.

Unreadable files are reported rather than fatal. A real folder contains
`Thumbs.db` and stray notes, and failing the whole upload over one of them
would force every client to reimplement this service's allow-list.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status

from pharmacy_ocr.api.dependencies import AuthDep, JobManagerDep, SettingsDep
from pharmacy_ocr.api.schemas import (
    BatchAcceptedResponse,
    BatchResultsResponse,
    BatchStatusResponse,
    ErrorResponse,
    RejectedFile,
)
from pharmacy_ocr.api.uploads import check_content_length, discard_upload, receive_upload
from pharmacy_ocr.domain.errors import BatchTooLargeError, EmptyBatchError, PharmacyOCRError
from pharmacy_ocr.jobs.manager import PendingUpload
from pharmacy_ocr.settings import ExtractionMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/batches", tags=["batches"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "No readable invoice in the upload"},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    413: {"model": ErrorResponse, "description": "Too many files or too many bytes"},
    503: {"model": ErrorResponse, "description": "Queue has no room for the batch"},
}


@router.post(
    "",
    response_model=BatchAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERROR_RESPONSES,
    summary="Queue a folder of invoices",
)
async def submit_batch(
    request: Request,
    response: Response,
    settings: SettingsDep,
    manager: JobManagerDep,
    _: AuthDep,
    files: list[UploadFile] = File(..., description="One part per invoice, all named `files`."),
    photo_mode: bool | None = Query(default=None),
    mode: ExtractionMode | None = Query(default=None),
) -> BatchAcceptedResponse:
    check_content_length(request.headers.get("content-length"), settings.max_batch_upload_bytes)
    if len(files) > settings.max_batch_files:
        raise BatchTooLargeError(
            f"A batch may contain at most {settings.max_batch_files} files.",
            detail={"max_batch_files": settings.max_batch_files, "received": len(files)},
        )

    accepted: list[PendingUpload] = []
    rejected: list[RejectedFile] = []
    try:
        for file in files:
            name = file.filename or "unnamed"
            try:
                path, filename = await receive_upload(file, settings)
            except PharmacyOCRError as exc:
                # One bad file must not cost the caller the whole folder.
                rejected.append(RejectedFile(filename=name, code=exc.code, message=exc.message))
                continue
            accepted.append(PendingUpload(filename=filename, path=path))

        if not accepted:
            raise EmptyBatchError(
                "No file in this upload could be read as an invoice.",
                detail={"rejected": [item.model_dump() for item in rejected]},
            )

        batch, jobs = await manager.submit_batch(accepted, photo_mode=photo_mode, extraction_mode=mode)
    except BaseException:
        for upload in accepted:
            discard_upload(upload.path)
        raise

    response.headers["Location"] = str(request.url_for("get_batch", batch_id=batch.id))
    logger.info(
        "batch accepted",
        extra={"batch_id": batch.id, "accepted": len(jobs), "rejected": len(rejected)},
    )
    return BatchAcceptedResponse.build(batch, jobs, rejected)


@router.get(
    "/{batch_id}",
    response_model=BatchStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Poll batch progress (no results)",
    name="get_batch",
)
async def get_batch(batch_id: str, manager: JobManagerDep, _: AuthDep) -> BatchStatusResponse:
    batch, jobs = manager.get_batch(batch_id)
    return BatchStatusResponse.build(batch, jobs)


@router.get(
    "/{batch_id}/results",
    response_model=BatchResultsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Collect every extraction in the batch",
)
async def get_batch_results(batch_id: str, manager: JobManagerDep, _: AuthDep) -> BatchResultsResponse:
    batch, jobs = manager.get_batch(batch_id)
    return BatchResultsResponse.build(batch, jobs)
