"""Asynchronous extraction jobs.

Submit returns 202 with a job id and a Location header; poll the job until its
status is terminal. This is the endpoint to use for anything batch-shaped.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status

from pharmacy_ocr.api.dependencies import AuthDep, JobManagerDep, SettingsDep
from pharmacy_ocr.api.schemas import ErrorResponse, JobResponse, JobStatsResponse
from pharmacy_ocr.api.uploads import check_content_length, discard_upload, receive_upload
from pharmacy_ocr.settings import ExtractionMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        503: {"model": ErrorResponse, "description": "Queue full"},
    },
    summary="Queue an invoice for extraction",
)
async def submit_job(
    request: Request,
    response: Response,
    settings: SettingsDep,
    manager: JobManagerDep,
    _: AuthDep,
    file: UploadFile = File(..., description="Invoice image or PDF."),
    photo_mode: bool | None = Query(default=None),
    mode: ExtractionMode | None = Query(default=None),
) -> JobResponse:
    check_content_length(request.headers.get("content-length"), settings.max_upload_bytes)
    path, filename = await receive_upload(file, settings)
    try:
        job = await manager.submit(filename=filename, path=path, photo_mode=photo_mode, extraction_mode=mode)
    except BaseException:
        discard_upload(path)
        raise

    response.headers["Location"] = str(request.url_for("get_job", job_id=job.id))
    return JobResponse.from_job(job)


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Fetch a job and its result",
    name="get_job",
)
async def get_job(job_id: str, manager: JobManagerDep, _: AuthDep) -> JobResponse:
    return JobResponse.from_job(manager.get(job_id))


@router.get("", response_model=JobStatsResponse, summary="Job counts by status")
async def job_stats(manager: JobManagerDep, _: AuthDep) -> JobStatsResponse:
    return JobStatsResponse(counts=manager.stats())
