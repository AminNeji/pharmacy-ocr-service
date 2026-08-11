"""Untrusted-file intake: extensions, size ceilings, magic bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmacy_ocr.domain.errors import (
    CorruptDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from pharmacy_ocr.imaging.files import (
    guard_megapixels,
    normalized_suffix,
    verify_magic_bytes,
    write_stream_with_limit,
)
from tests.conftest import PDF_MAGIC, PNG_MAGIC


@pytest.mark.parametrize("filename", ["invoice.png", "INVOICE.PNG", "scan.pdf", "photo.jpeg"])
def test_supported_extensions_are_accepted(filename: str) -> None:
    assert normalized_suffix(filename).startswith(".")


@pytest.mark.parametrize("filename", ["payload.exe", "script.php", "archive.zip", "noext", None])
def test_unsupported_extensions_are_rejected(filename: str | None) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        normalized_suffix(filename)


def test_only_the_final_suffix_counts() -> None:
    """`shell.php.png` is a PNG upload; the earlier segment is inert."""
    assert normalized_suffix("shell.php.png") == ".png"


def test_traversal_in_filename_cannot_change_the_suffix() -> None:
    assert normalized_suffix("../../etc/passwd.png") == ".png"
    with pytest.raises(UnsupportedFileTypeError):
        normalized_suffix("../../etc/passwd")


def test_stream_writer_enforces_the_size_ceiling(tmp_path: Path) -> None:
    destination = tmp_path / "big.png"
    chunks = iter([b"x" * 100, b"y" * 100])
    with pytest.raises(FileTooLargeError):
        write_stream_with_limit(chunks, destination, max_bytes=150)
    assert not destination.exists(), "a rejected upload must not be left on disk"


def test_stream_writer_rejects_empty_uploads(tmp_path: Path) -> None:
    destination = tmp_path / "empty.png"
    with pytest.raises(CorruptDocumentError):
        write_stream_with_limit(iter([b""]), destination, max_bytes=100)
    assert not destination.exists()


def test_stream_writer_accepts_a_normal_upload(tmp_path: Path) -> None:
    destination = tmp_path / "ok.png"
    written = write_stream_with_limit(iter([PNG_MAGIC, b"body"]), destination, max_bytes=1000)
    assert written == len(PNG_MAGIC) + 4
    assert destination.read_bytes().startswith(PNG_MAGIC)


def test_magic_bytes_must_match_the_extension(tmp_path: Path) -> None:
    """A PDF renamed to .png is rejected before any decoder sees it."""
    path = tmp_path / "disguised.png"
    path.write_bytes(PDF_MAGIC + b"content")
    with pytest.raises(CorruptDocumentError, match="do not match"):
        verify_magic_bytes(path)


def test_magic_bytes_accept_a_genuine_file(tmp_path: Path) -> None:
    path = tmp_path / "real.png"
    path.write_bytes(PNG_MAGIC + b"content")
    verify_magic_bytes(path)


def test_webp_requires_the_format_tag(tmp_path: Path) -> None:
    path = tmp_path / "fake.webp"
    path.write_bytes(b"RIFF" + b"\x00" * 4 + b"AVI ")
    with pytest.raises(CorruptDocumentError, match="not a WEBP"):
        verify_magic_bytes(path)


def test_megapixel_guard_blocks_decompression_bombs() -> None:
    with pytest.raises(CorruptDocumentError, match="too large"):
        guard_megapixels(30000, 30000, max_megapixels=80, source="bomb.png")


def test_megapixel_guard_allows_a_normal_scan() -> None:
    guard_megapixels(2480, 3508, max_megapixels=80, source="a4.png")
