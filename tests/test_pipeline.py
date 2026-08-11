"""End-to-end pipeline behaviour against a fake OCR engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pharmacy_ocr.domain.errors import OCREngineError
from pharmacy_ocr.domain.models import ReviewStatus
from pharmacy_ocr.ocr.types import OCRPage
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import ExtractionMode, Settings
from tests.conftest import FakeOCREngine, FakePreprocessor, make_token

MISMATCHED_ROWS = [
    "300000 ASPIRINE 500MG 1 5.000 5.000",
    "Mnt Total Facture 99.999 Dinars",
]


class FakeMapper:
    """Stands in for Ollama."""

    def __init__(self, items: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.items = items or []
        self.error = error
        self.calls = 0
        self.closed = False

    async def map_pages(self, pages: list[OCRPage], **_: Any) -> list[dict[str, Any]]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.items

    async def aclose(self) -> None:
        self.closed = True


def build(settings: Settings, **overrides: Any) -> InvoicePipeline:
    return InvoicePipeline(
        settings,
        engine=overrides.pop("engine", FakeOCREngine()),
        preprocessor=overrides.pop("preprocessor", FakePreprocessor()),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_extracts_items_and_passes_checksum(pipeline: InvoicePipeline, invoice_file: Path):
    result = await pipeline.process(invoice_file)

    assert result.status is ReviewStatus.ok
    assert result.error is None
    assert len(result.items) == 3
    assert result.items_total == 60.855
    assert result.invoice_total == 60.855
    assert result.checksum_ok is True
    assert result.warnings == []


async def test_result_carries_provenance(pipeline: InvoicePipeline, invoice_file: Path):
    result = await pipeline.process(invoice_file)
    assert result.source_file == "invoice1.png"
    assert result.page_count == 1
    assert result.ocr_engine == "fake"
    assert result.extraction_mode == ExtractionMode.deterministic.value
    assert result.service_version


async def test_timings_are_reported(pipeline: InvoicePipeline, invoice_file: Path):
    result = await pipeline.process(invoice_file)
    assert {"parsing_s", "ocr_s", "preprocessing_s", "total_s"} <= result.timings.keys()
    assert all(value >= 0 for value in result.timings.values())


async def test_parsing_time_tracks_document_size(settings: Settings, invoice_file: Path):
    """Regression: `parsing_s` used to time an empty block, so it read 0 for
    every invoice regardless of how much text was parsed."""
    big = [f"30{index:04d} MEDICINE NUMBER {index} 1 5.000 5.000" for index in range(4000)]
    pipeline = build(settings, engine=FakeOCREngine(lines=big))

    result = await pipeline.process(invoice_file)

    assert result.timings["parsing_s"] > 0, "parsing time must reflect real work"


# --------------------------------------------------------------------------- #
# Failure handling — the pipeline must never raise
# --------------------------------------------------------------------------- #
async def test_ocr_failure_becomes_a_failed_result(settings: Settings, invoice_file: Path):
    class BrokenEngine(FakeOCREngine):
        def run_page(self, image_path, page_index=0):
            raise OCREngineError("model weights missing")

    result = await build(settings, engine=BrokenEngine()).process(invoice_file)

    assert result.status is ReviewStatus.failed
    assert result.error is not None
    assert "model weights missing" in result.error
    assert result.items == []


async def test_preprocessing_failure_becomes_a_failed_result(settings: Settings, invoice_file: Path):
    class BrokenPreprocessor(FakePreprocessor):
        def preprocess(self, *args, **kwargs):
            raise ValueError("could not decode image")

    result = await build(settings, preprocessor=BrokenPreprocessor()).process(invoice_file)
    assert result.status is ReviewStatus.failed
    assert "could not decode image" in (result.error or "")


async def test_missing_input_file_is_reported_not_raised(settings: Settings, tmp_path: Path):
    pipeline = InvoicePipeline(settings, engine=FakeOCREngine())
    result = await pipeline.process(tmp_path / "nope.png")
    assert result.status is ReviewStatus.failed


async def test_blank_page_is_flagged_for_review(settings: Settings, invoice_file: Path):
    result = await build(settings, engine=FakeOCREngine(lines=[])).process(invoice_file)

    assert result.status is ReviewStatus.needs_review
    assert result.error is None
    assert any("no text" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
async def test_second_run_reuses_cached_ocr(settings: Settings, invoice_file: Path):
    engine = FakeOCREngine()
    preprocessor = FakePreprocessor()
    pipeline = build(settings, engine=engine, preprocessor=preprocessor)

    first = await pipeline.process(invoice_file)
    assert engine.calls == 1

    second = await pipeline.process(invoice_file)
    assert engine.calls == 1, "OCR must not run again for identical input"
    assert preprocessor.calls == 1
    assert second.items_total == first.items_total
    assert second.timings.get("primary_ocr_cache_hit") == 1.0


# --------------------------------------------------------------------------- #
# Page orientation
# --------------------------------------------------------------------------- #
class RotationAwareEngine(FakeOCREngine):
    """Returns unreadable vertical strips until the page is turned upright.

    Mirrors the real failure: three corpus invoices came back with 80%+ of
    their boxes taller than wide, and two of those produced no items at all.
    """

    def __init__(self, upright_angle: int, lines: list[str]) -> None:
        super().__init__(lines=lines)
        self.upright_angle = upright_angle
        self.seen_angles: list[int] = []

    def run_page(self, image_path, page_index: int = 0) -> OCRPage:
        self.calls += 1
        angle = int(str(image_path).split(".rot")[1].split(".")[0])
        self.seen_angles.append(angle)

        if angle == self.upright_angle:
            # Wide boxes with clear column gaps, as an upright scan produces.
            tokens = [
                make_token(text, 10 + 220 * i, 100, width=90, height=18)
                for i, text in enumerate(self.lines[0].split())
            ]
            return OCRPage(text="\n".join(self.lines), tokens=tokens, page_index=page_index)

        # Sideways: tall, narrow boxes carrying fragments.
        tokens = [make_token(f"x{i}", 10, 40 * i, width=14, height=90) for i in range(12)]
        return OCRPage(text="\n".join(t.text for t in tokens), tokens=tokens, page_index=page_index)


async def test_sideways_page_is_re_ocred_upright(settings: Settings, invoice_file: Path):
    engine = RotationAwareEngine(270, ["300000 ASPIRINE 500MG 1 5.000 5.000", "Total 5.000"])
    result = await build(settings, engine=engine).process(invoice_file)

    assert 270 in engine.seen_angles, "the sideways page must be retried upright"
    assert result.timings["orientation_degrees"] == 270
    assert [item.MED for item in result.items] == ["ASPIRINE 500MG"]


async def test_upright_page_is_never_re_ocred(settings: Settings, invoice_file: Path):
    """The extra pass must not be paid on the 96% of pages that are fine."""
    preprocessor = FakePreprocessor()
    engine = FakeOCREngine()
    await build(settings, engine=engine, preprocessor=preprocessor).process(invoice_file)

    assert preprocessor.rotations == [0]
    assert engine.calls == 1


class UpsideDownEngine(FakeOCREngine):
    """Confident nonsense until the page is turned the right way up.

    A 180-degree page keeps normal box proportions, so the aspect-ratio test
    reports nothing wrong. Only the fact that no rows parse gives it away.
    """

    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines=lines)
        self.seen_angles: list[int] = []

    def run_page(self, image_path, page_index: int = 0) -> OCRPage:
        self.calls += 1
        angle = int(str(image_path).split(".rot")[1].split(".")[0])
        self.seen_angles.append(angle)
        source = self.lines if angle == 180 else ["ale6 sdreqwoo XAT0AVON", "9EZ 9l TH IejoL"]
        tokens = [
            make_token(text, 10 + 220 * i, 100, width=90, height=18)
            for i, text in enumerate(source[0].split())
        ]
        return OCRPage(text="\n".join(source), tokens=tokens, page_index=page_index)


async def test_upside_down_page_is_recovered(settings: Settings, invoice_file: Path):
    engine = UpsideDownEngine(["301108 NOVACLAV 500 COMP B/24 1 16.236 16.236", "Total TTC 16.236"])
    result = await build(settings, engine=engine).process(invoice_file)

    assert 180 in engine.seen_angles, "an unparseable page must try 180 degrees"
    assert [item.REFCODE for item in result.items] == ["301108"]
    assert result.items[0].prix == 16.236


async def test_recovery_sweep_is_budgeted(settings: Settings, invoice_file: Path):
    """Each probe is a full re-OCR; an unbounded sweep cost ~5 min per invoice."""
    engine = UpsideDownEngine(["nothing parses here at any angle"])
    tuned = settings.model_copy(update={"max_orientation_probes": 1})
    await build(tuned, engine=engine).process(invoice_file)

    # One first-pass OCR plus at most one probe.
    assert engine.calls <= 2
    assert engine.seen_angles == [0, 180], "180 must be the angle worth spending on"


async def test_recovery_sweep_can_be_switched_off(settings: Settings, invoice_file: Path):
    engine = UpsideDownEngine(["nothing parses here at any angle"])
    tuned = settings.model_copy(update={"max_orientation_probes": 0})
    await build(tuned, engine=engine).process(invoice_file)
    assert engine.seen_angles == [0]


async def test_recovery_sweep_only_runs_when_nothing_parsed(settings: Settings, invoice_file: Path):
    """The sweep is expensive; a page that parsed rows must not pay for it."""
    preprocessor = FakePreprocessor()
    await build(settings, preprocessor=preprocessor).process(invoice_file)
    assert preprocessor.rotations == [0]


async def test_orientation_correction_can_be_disabled(settings: Settings, invoice_file: Path):
    tuned = settings.model_copy(update={"correct_page_orientation": False})
    engine = RotationAwareEngine(270, ["300000 ASPIRINE 500MG 1 5.000 5.000"])
    await build(tuned, engine=engine).process(invoice_file)
    assert engine.seen_angles == [0]


# --------------------------------------------------------------------------- #
# PDF re-OCR policy
# --------------------------------------------------------------------------- #
def _pdf(tmp_path: Path) -> Path:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.7\nfake")
    return path


async def test_no_retry_when_the_checksum_already_passes(settings: Settings, tmp_path: Path):
    """A passing checksum is proof the extraction is right; re-OCR is waste.

    Previously any soft warning forced two extra full-document OCR passes,
    roughly tripling the cost of a typical PDF for no accuracy gain.
    """
    rows = [
        "300000 ASPIRINE 500MG 1 5.000 5.000",
        "SOME PRODUCT WITHOUT A QUANTITY 3.000",  # emits a quantity warning
        "Mnt Total Facture 8.000 Dinars",
    ]
    engine = FakeOCREngine(lines=rows)
    result = await build(settings, engine=engine).process(_pdf(tmp_path))

    assert result.checksum_ok is True
    assert result.warnings, "the soft warning is still reported"
    assert engine.calls == 1, "a verified invoice must not be re-OCR'd"


async def test_retry_when_the_checksum_fails(settings: Settings, tmp_path: Path):
    engine = FakeOCREngine(lines=MISMATCHED_ROWS)
    await build(settings, engine=engine).process(_pdf(tmp_path))
    assert engine.calls > 1, "a failed checksum must trigger a higher-DPI retry"


async def test_retry_when_no_rows_were_found(settings: Settings, tmp_path: Path):
    engine = FakeOCREngine(lines=["Pharmacie Centrale", "Merci de votre visite"])
    await build(settings, engine=engine).process(_pdf(tmp_path))
    assert engine.calls > 1


async def test_retry_when_warnings_and_no_total_to_verify(settings: Settings, tmp_path: Path):
    engine = FakeOCREngine(lines=["ASPIRINE 500 MG COMPRIME 12.345"])
    await build(settings, engine=engine).process(_pdf(tmp_path))
    assert engine.calls > 1, "with no printed total, warnings are the only signal"


async def test_images_are_never_re_ocred(settings: Settings, invoice_file: Path):
    """The retry re-renders a PDF at higher DPI; an image has no such knob."""
    engine = FakeOCREngine(lines=MISMATCHED_ROWS)
    await build(settings, engine=engine).process(invoice_file)
    assert engine.calls == 1


# --------------------------------------------------------------------------- #
# LLM fallback policy
# --------------------------------------------------------------------------- #
async def test_deterministic_mode_never_calls_the_llm(settings: Settings, invoice_file: Path):
    mapper = FakeMapper(items=[{"MED": "SHOULD NOT APPEAR", "prix": 1.0}])
    pipeline = build(settings, engine=FakeOCREngine(MISMATCHED_ROWS), mapper=mapper)

    result = await pipeline.process(invoice_file)
    assert mapper.calls == 0
    assert result.checksum_ok is False


async def test_hybrid_accepts_the_llm_when_it_fixes_the_checksum(settings: Settings, invoice_file: Path):
    hybrid = settings.model_copy(update={"extraction_mode": ExtractionMode.hybrid})
    mapper = FakeMapper(items=[{"REFCODE": "300000", "MED": "ASPIRINE 500MG", "QT": 1, "prix": 99.999}])
    pipeline = build(hybrid, engine=FakeOCREngine(MISMATCHED_ROWS), mapper=mapper)

    result = await pipeline.process(invoice_file)

    assert mapper.calls == 1
    assert result.checksum_ok is True
    assert result.items[0].prix == 99.999
    assert any("LLM mapping used" in warning for warning in result.warnings)


async def test_hybrid_rejects_the_llm_when_the_parser_is_no_worse(settings: Settings, invoice_file: Path):
    """The parser is reproducible; the model is not. Ties go to the parser."""
    hybrid = settings.model_copy(update={"extraction_mode": ExtractionMode.hybrid})
    mapper = FakeMapper(items=[{"MED": "HALLUCINATED PRODUCT", "QT": 1, "prix": 1.0}])
    pipeline = build(hybrid, engine=FakeOCREngine(MISMATCHED_ROWS), mapper=mapper)

    result = await pipeline.process(invoice_file)

    assert mapper.calls == 1
    assert [item.MED for item in result.items] == ["ASPIRINE 500MG"]
    assert any("rejected" in warning for warning in result.warnings)


async def test_llm_failure_keeps_the_parser_rows(settings: Settings, invoice_file: Path):
    hybrid = settings.model_copy(update={"extraction_mode": ExtractionMode.hybrid})
    mapper = FakeMapper(error=RuntimeError("ollama is down"))
    pipeline = build(hybrid, engine=FakeOCREngine(MISMATCHED_ROWS), mapper=mapper)

    result = await pipeline.process(invoice_file)

    assert result.status is ReviewStatus.needs_review
    assert len(result.items) == 1
    assert any("LLM fallback failed" in warning for warning in result.warnings)


async def test_per_request_mode_override_beats_settings(settings: Settings, invoice_file: Path):
    mapper = FakeMapper(items=[{"MED": "FROM LLM", "QT": 1, "prix": 60.855}])
    pipeline = build(settings, mapper=mapper)  # settings say deterministic

    result = await pipeline.process(invoice_file, extraction_mode=ExtractionMode.llm)

    assert mapper.calls == 1
    assert [item.MED for item in result.items] == ["FROM LLM"]
    assert result.extraction_mode == "llm"


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #
async def test_low_quality_rows_are_dropped_with_a_reason(settings: Settings, invoice_file: Path):
    mapper = FakeMapper(
        items=[
            {"MED": "VALID MEDICINE", "QT": 1, "prix": 10.0},
            {"MED": "sirop", "QT": 1, "prix": 5.0},
            {"MED": "OK BUT FREE", "QT": 1, "prix": 0},
        ]
    )
    pipeline = build(settings, mapper=mapper)

    result = await pipeline.process(invoice_file, extraction_mode=ExtractionMode.llm)

    assert [item.MED for item in result.items] == ["VALID MEDICINE"]
    assert sum("dropped" in warning for warning in result.warnings) == 2


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
async def test_aclose_releases_the_llm_client(settings: Settings):
    mapper = FakeMapper()
    await build(settings, mapper=mapper).aclose()
    assert mapper.closed is True


def test_warmup_delegates_to_the_engine(settings: Settings):
    engine = FakeOCREngine()
    build(settings, engine=engine).warmup()
    assert engine.warmed is True


@pytest.mark.parametrize("pages", [1, 3])
async def test_page_count_reflects_the_document(settings: Settings, invoice_file: Path, pages: int):
    pipeline = build(settings, preprocessor=FakePreprocessor(pages=pages))
    result = await pipeline.process(invoice_file)
    assert result.page_count == pages
