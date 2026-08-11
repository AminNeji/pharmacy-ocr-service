"""Shared fixtures.

The OCR engine and the image preprocessor are the only components that need
model weights or real pixels, so both are faked here. Everything else — cache,
parser, pipeline, job queue, HTTP layer — runs exactly as it does in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmacy_ocr.ocr.types import OCRPage, OCRToken
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import Settings

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PDF_MAGIC = b"%PDF-1.7\n"

SAMPLE_ROWS = [
    "303616 OMNICEF 125 MG SUSP ORAL 80 ML 2 17.340 34.680",
    "PHYSIOMER JET NORMAL 1 23.300 23.300",
    "301034 MUCOLARE 2% ENF FL/125 ML 1 2.875 2.875",
    "Mnt Total Facture 60.855 Dinars",
]


class FakeOCREngine:
    """Returns scripted OCR pages instead of loading PaddleOCR."""

    def __init__(self, lines: list[str] | None = None, tokens: list[OCRToken] | None = None) -> None:
        self.lines = list(lines) if lines is not None else list(SAMPLE_ROWS)
        self.tokens = tokens
        self.warmed = False
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def warmup(self) -> None:
        self.warmed = True

    def run_page(self, image_path: str | Path, page_index: int = 0) -> OCRPage:
        self.calls += 1
        return OCRPage(
            text="\n".join(self.lines),
            tokens=list(self.tokens or []),
            source_image=str(image_path),
            page_index=page_index,
        )


class FakePreprocessor:
    """Yields page paths without decoding anything."""

    def __init__(self, pages: int = 1) -> None:
        self.pages = pages
        self.calls = 0
        self.rotations: list[int] = []

    def preprocess(
        self,
        input_path: str | Path,
        photo_mode: bool | None = None,
        pdf_render_dpi: int | None = None,
        target_long_side: int | None = None,
        max_long_side: int | None = None,
        rotate_degrees: int = 0,
    ) -> list[Path]:
        self.calls += 1
        self.rotations.append(rotate_degrees)
        # Rotation is encoded in the page path so a fake engine can return
        # different OCR for the upright and sideways renders.
        return [
            Path(f"{input_path}.rot{rotate_degrees % 360}.page{index}.png") for index in range(self.pages)
        ]

    @staticmethod
    def cleanup(paths: list[Path]) -> None:
        for path in paths:
            path.unlink(missing_ok=True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="development",
        data_dir=tmp_path / "data",
        log_to_file=False,
        log_format="text",
        warmup_on_startup=False,
        extraction_mode="deterministic",
        rate_limit_per_minute=0,
        max_concurrent_extractions=2,
        docs_enabled=True,
    )


@pytest.fixture
def engine() -> FakeOCREngine:
    return FakeOCREngine()


@pytest.fixture
def preprocessor() -> FakePreprocessor:
    return FakePreprocessor()


@pytest.fixture
def pipeline(settings: Settings, engine: FakeOCREngine, preprocessor: FakePreprocessor):
    settings.ensure_directories()
    return InvoicePipeline(settings, engine=engine, preprocessor=preprocessor)


@pytest.fixture
def invoice_file(tmp_path: Path) -> Path:
    """A file that passes intake validation; its bytes are never decoded."""
    path = tmp_path / "invoice1.png"
    path.write_bytes(PNG_MAGIC + b"fake-image-payload" * 8)
    return path


def make_token(
    text: str,
    x0: float,
    y0: float,
    *,
    width: float = 60,
    height: float = 20,
    score: float = 0.95,
    page: int = 0,
) -> OCRToken:
    return OCRToken(
        text=text,
        score=score,
        box=(x0, y0, x0 + width, y0 + height),
        page_index=page,
    )
