"""Turning an untrusted multipart upload into a validated file on disk."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import UploadFile

from pharmacy_ocr.domain.errors import FileTooLargeError
from pharmacy_ocr.imaging.files import awrite_stream_with_limit, normalized_suffix, verify_magic_bytes
from pharmacy_ocr.settings import Settings

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024


async def _chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def check_content_length(content_length: str | None, max_bytes: int) -> None:
    """Reject an oversized body before reading it.

    Advisory only — the header can lie — but it turns a 2 GB upload into an
    immediate 413 instead of two gigabytes of transfer.
    """
    if not content_length:
        return
    try:
        declared = int(content_length)
    except ValueError:
        return
    if declared > max_bytes:
        raise FileTooLargeError(
            f"Upload exceeds the {max_bytes} byte limit.",
            detail={"max_bytes": max_bytes, "declared_bytes": declared},
        )


async def receive_upload(file: UploadFile, settings: Settings) -> tuple[Path, str]:
    """Persist an upload to a temp file. Returns (path, original filename).

    Validates in order: extension allow-list, streamed size ceiling, then magic
    bytes. The caller owns the returned path and must delete it.
    """
    suffix = normalized_suffix(file.filename)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    # The stored name is generated, never derived from user input, so a crafted
    # filename cannot escape the temp directory.
    destination = settings.temp_dir / f"upload_{uuid.uuid4().hex}{suffix}"

    await awrite_stream_with_limit(_chunks(file), destination, settings.max_upload_bytes)
    try:
        verify_magic_bytes(destination, suffix)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    return destination, (file.filename or destination.name)


def discard_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete upload", extra={"upload": str(path)})
