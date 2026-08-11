"""Untrusted-file intake.

Everything here runs before a decoder sees the bytes. The previous version
trusted the uploaded filename's extension and streamed the whole body to disk
with no size ceiling, which made both a disk-fill and a decompression bomb
trivial to trigger.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from pharmacy_ocr.domain.errors import CorruptDocumentError, FileTooLargeError, UnsupportedFileTypeError
from pharmacy_ocr.settings import MAGIC_PREFIXES, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_MAGIC_READ_BYTES = 16


def normalized_suffix(filename: str | None, *, supported: frozenset[str] = SUPPORTED_EXTENSIONS) -> str:
    """Return the validated lowercase extension of `filename`.

    Only the final suffix is considered and it must be in the allow-list, so a
    crafted name like `../../etc/passwd` or `x.php.png` cannot influence the
    path we later write to.
    """
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in supported:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or '(none)'}'.",
            detail={"supported": sorted(supported)},
        )
    return suffix


def write_stream_with_limit(chunks: Iterator[bytes], destination: Path, max_bytes: int) -> int:
    """Stream `chunks` to `destination`, aborting past `max_bytes`.

    The partial file is removed on overflow so a rejected upload leaves nothing
    behind. Returns the number of bytes written.
    """
    written = 0
    try:
        with destination.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"Upload exceeds the {max_bytes} byte limit.",
                        detail={"max_bytes": max_bytes},
                    )
                handle.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise CorruptDocumentError("Uploaded file is empty.")
    return written


async def awrite_stream_with_limit(chunks: AsyncIterator[bytes], destination: Path, max_bytes: int) -> int:
    """Async twin of `write_stream_with_limit` for ASGI uploads.

    Enforcing the ceiling while streaming — rather than after the body has
    landed on disk — is the point: an unbounded upload previously filled the
    disk before anything checked its size.
    """
    written = 0
    try:
        with destination.open("wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"Upload exceeds the {max_bytes} byte limit.",
                        detail={"max_bytes": max_bytes},
                    )
                handle.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise CorruptDocumentError("Uploaded file is empty.")
    return written


def verify_magic_bytes(path: Path, suffix: str | None = None) -> None:
    """Reject files whose leading bytes contradict their extension."""
    suffix = (suffix or path.suffix).lower()
    expected = MAGIC_PREFIXES.get(suffix)
    if not expected:
        return
    try:
        with path.open("rb") as handle:
            header = handle.read(_MAGIC_READ_BYTES)
    except OSError as exc:
        raise CorruptDocumentError(f"Could not read uploaded file: {exc}") from exc

    if not any(header.startswith(prefix) for prefix in expected):
        raise CorruptDocumentError(
            f"File contents do not match the '{suffix}' extension.",
            detail={"expected_type": suffix},
        )
    # WEBP additionally carries its format tag at offset 8.
    if suffix == ".webp" and header[8:12] != b"WEBP":
        raise CorruptDocumentError("File is a RIFF container but not a WEBP image.")


def probe_image_dimensions(path: Path) -> tuple[int, int] | None:
    """Read image dimensions from the header without decoding pixel data.

    Returns None when the format cannot be probed (PDFs, unusual encodings);
    callers then fall back to post-decode checks.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001 - probing is best-effort by design
        return None


def guard_megapixels(width: int, height: int, max_megapixels: float, *, source: str) -> None:
    """Raise before allocating a decode buffer that would exhaust memory."""
    megapixels = (width * height) / 1_000_000
    if megapixels > max_megapixels:
        raise CorruptDocumentError(
            f"Image is too large to process ({megapixels:.1f} MP, limit {max_megapixels:.0f} MP).",
            detail={"source": source, "width": width, "height": height},
        )
