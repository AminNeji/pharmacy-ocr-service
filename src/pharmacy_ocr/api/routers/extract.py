"""Synchronous extraction endpoint.

Use this for interactive, single-invoice calls. Anything batch-shaped, or any
client behind a proxy with a short idle timeout, should use the job API in
`routers/jobs.py` instead — OCR routinely takes longer than 60 seconds.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Query, Request, UploadFile

from pharmacy_ocr.api.dependencies import AuthDep, PipelineDep, SettingsDep
from pharmacy_ocr.api.schemas import ErrorResponse
from pharmacy_ocr.api.uploads import check_content_length, discard_upload, receive_upload
from pharmacy_ocr.domain.errors import ServiceOverloadedError
from pharmacy_ocr.domain.models import InvoiceResult
from pharmacy_ocr.settings import ExtractionMode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extraction"])

# Literal codes rather than `status.*`: Starlette renamed several constants
# (413, 422) and the old spellings emit deprecation warnings.
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "Corrupt or empty file"},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    413: {"model": ErrorResponse, "description": "Upload too large"},
    415: {"model": ErrorResponse, "description": "Unsupported file type"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Server at capacity"},
}

PhotoModeQuery = Query(
    default=None,
    description="Override the configured photo mode: true for phone photos, false for clean scans.",
)
ModeQuery = Query(default=None, description="Override the extraction mode for this request.")


@router.post(
    "/v1/extract",
    response_model=InvoiceResult,
    responses=_ERROR_RESPONSES,
    summary="Extract invoice items (synchronous)",
)
async def extract(
    request: Request,
    settings: SettingsDep,
    pipeline: PipelineDep,
    _: AuthDep,
    file: UploadFile = File(..., description="Invoice image or PDF."),
    photo_mode: bool | None = PhotoModeQuery,
    mode: ExtractionMode | None = ModeQuery,
) -> InvoiceResult:
    check_content_length(request.headers.get("content-length"), settings.max_upload_bytes)
    path, filename = await receive_upload(file, settings)

    semaphore: asyncio.Semaphore = request.app.state.extraction_semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=settings.request_timeout_s)
    except TimeoutError as exc:
        discard_upload(path)
        raise ServiceOverloadedError(
            "Server is at capacity; retry shortly or use the asynchronous job API.",
            detail={"max_concurrent_extractions": settings.max_concurrent_extractions},
        ) from exc

    try:
        result = await pipeline.process(path, photo_mode=photo_mode, extraction_mode=mode)
    finally:
        semaphore.release()
        discard_upload(path)

    result.source_file = filename
    return result


@router.post(
    "/extract",
    response_model=InvoiceResult,
    responses=_ERROR_RESPONSES,
    summary="Extract invoice items (deprecated path)",
    deprecated=True,
    include_in_schema=False,
)
async def extract_legacy(
    request: Request,
    settings: SettingsDep,
    pipeline: PipelineDep,
    auth: AuthDep,
    file: UploadFile = File(...),
    photo_mode: bool | None = PhotoModeQuery,
    mode: ExtractionMode | None = ModeQuery,
) -> InvoiceResult:
    """Unversioned alias kept so v2 clients keep working. Prefer /v1/extract."""
    return await extract(request, settings, pipeline, auth, file, photo_mode, mode)
