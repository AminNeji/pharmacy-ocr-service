"""Orchestration: preprocess -> OCR (cached) -> parse -> validate -> checksum.

Contract: `InvoicePipeline.process` never raises. Every failure becomes an
`InvoiceResult` carrying `error`/`warnings`, so one bad invoice cannot kill a
batch of two thousand.

Collaborators are injected rather than imported as module singletons, which is
what lets the whole pipeline run in tests against a fake OCR engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pharmacy_ocr import __version__
from pharmacy_ocr.domain.errors import PharmacyOCRError
from pharmacy_ocr.domain.models import InvoiceItem, InvoiceResult, ParsedInvoice, ReviewStatus
from pharmacy_ocr.imaging import orientation
from pharmacy_ocr.imaging.preprocessing import Preprocessor
from pharmacy_ocr.observability import metrics
from pharmacy_ocr.ocr.cache import OCRCache
from pharmacy_ocr.ocr.engine import OCREngine, create_engine
from pharmacy_ocr.ocr.types import OCRPage, pages_text
from pharmacy_ocr.parsing import rules
from pharmacy_ocr.parsing.llm import LLMMapper
from pharmacy_ocr.parsing.rules import RowParser
from pharmacy_ocr.settings import ExtractionMode, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RenderProfile:
    """One OCR attempt's rendering parameters."""

    name: str
    pdf_render_dpi: int | None = None
    target_long_side: int | None = None
    max_long_side: int | None = None
    rotate_degrees: int = 0


class InvoicePipeline:
    """Runs one invoice end to end."""

    def __init__(
        self,
        settings: Settings,
        *,
        engine: OCREngine | None = None,
        cache: OCRCache | None = None,
        preprocessor: Preprocessor | None = None,
        parser: RowParser | None = None,
        mapper: LLMMapper | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine or create_engine(settings)
        self._cache = cache or OCRCache(settings)
        self._preprocessor = preprocessor or Preprocessor(settings)
        self._parser = parser or RowParser(settings)
        self._mapper = mapper or LLMMapper(settings)

    # ------------------------------------------------------------ lifecycle --
    def warmup(self) -> None:
        self._engine.warmup()

    async def aclose(self) -> None:
        await self._mapper.aclose()

    async def warm_cache(
        self,
        input_path: str | Path,
        *,
        photo_mode: bool | None = None,
        profile: RenderProfile | None = None,
    ) -> list[OCRPage]:
        """Run OCR for one render profile and store it in the cache.

        Used by `pharmacy-ocr cache warm` so parser and prompt iteration can
        replay byte-identical OCR without paying for it again.
        """
        return await self._ocr_pages(Path(input_path), photo_mode, {}, profile or RenderProfile("primary"))

    # --------------------------------------------------------------- public --
    async def process(
        self,
        input_path: str | Path,
        *,
        photo_mode: bool | None = None,
        extraction_mode: ExtractionMode | None = None,
    ) -> InvoiceResult:
        """Extract one invoice. Never raises."""
        path = Path(input_path)
        mode = extraction_mode or self._settings.extraction_mode
        warnings: list[str] = []
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        metrics.EXTRACTIONS_IN_FLIGHT.inc()
        try:
            return await self._process(path, mode, photo_mode, warnings, timings, total_start)
        except Exception as exc:
            logger.exception("pipeline crashed", extra={"file": path.name})
            return self._failed(path, f"unexpected pipeline error: {exc}", warnings, timings, total_start)
        finally:
            metrics.EXTRACTIONS_IN_FLIGHT.dec()

    async def _process(
        self,
        path: Path,
        mode: ExtractionMode,
        photo_mode: bool | None,
        warnings: list[str],
        timings: dict[str, float],
        total_start: float,
    ) -> InvoiceResult:
        try:
            ocr_pages = await self._ocr_pages(path, photo_mode, timings, RenderProfile("primary"))
        except PharmacyOCRError as exc:
            logger.warning("OCR failed", extra={"file": path.name, "error": exc.message})
            return self._failed(path, f"OCR failed: {exc.message}", warnings, timings, total_start)
        except Exception as exc:
            logger.exception("OCR failed", extra={"file": path.name})
            return self._failed(path, f"OCR failed: {exc}", warnings, timings, total_start)

        ocr_pages, tried_angles = await self._correct_orientation(
            path, photo_mode, ocr_pages, timings, warnings
        )

        if not pages_text(ocr_pages).strip():
            warnings.append("OCR returned no text; the document may be blank or unreadable")
            return self._finalize(path, [], warnings, [], ocr_pages, mode, timings, total_start)

        start = time.perf_counter()
        parsed = self._parser.parse(ocr_pages)
        timings["parsing_s"] = round(time.perf_counter() - start, 3)

        # An upside-down page reads as plausible text but parses to nothing, so
        # "no rows" is the only signal that the page is the wrong way up.
        if not parsed.items:
            ocr_pages, parsed = await self._recover_by_rotation(
                path, photo_mode, ocr_pages, parsed, tried_angles, timings
            )

        if self._needs_retry(path, parsed):
            ocr_pages, parsed = await self._retry_ocr(path, photo_mode, ocr_pages, parsed, timings, warnings)

        raw_items = list(parsed.items)
        if mode is ExtractionMode.llm:
            raw_items = []
        else:
            warnings.extend(parsed.warnings)
            logger.info(
                "deterministic parsing complete",
                extra={"file": path.name, "raw_items": len(raw_items)},
            )

        raw_items = await self._maybe_map_with_llm(
            path, mode, ocr_pages, raw_items, parsed.invoice_total, warnings, timings
        )
        return self._finalize(path, raw_items, warnings, parsed.lines, ocr_pages, mode, timings, total_start)

    # ------------------------------------------------------------------ ocr --
    async def _ocr_pages(
        self,
        path: Path,
        photo_mode: bool | None,
        timings: dict[str, float],
        profile: RenderProfile,
    ) -> list[OCRPage]:
        key = self._cache.key_for(
            path,
            photo_mode=photo_mode,
            pdf_render_dpi=profile.pdf_render_dpi,
            target_long_side=profile.target_long_side,
            max_long_side=profile.max_long_side,
            rotate_degrees=profile.rotate_degrees,
        )
        cached = self._cache.load(key)
        if cached is not None:
            timings[f"{profile.name}_ocr_cache_hit"] = 1.0
            return cached

        profile_timings: dict[str, float] = {}
        start = time.perf_counter()
        page_paths = await asyncio.to_thread(
            self._preprocessor.preprocess,
            path,
            photo_mode,
            profile.pdf_render_dpi,
            profile.target_long_side,
            profile.max_long_side,
            profile.rotate_degrees,
        )
        try:
            preprocessing_s = round(time.perf_counter() - start, 2)
            self._record_stage(timings, profile, "preprocessing_s", preprocessing_s, profile_timings)

            start = time.perf_counter()
            pages: list[OCRPage] = []
            for page_index, page_path in enumerate(page_paths):
                pages.append(await asyncio.to_thread(self._engine.run_page, page_path, page_index))
            ocr_s = round(time.perf_counter() - start, 2)
            self._record_stage(timings, profile, "ocr_s", ocr_s, profile_timings)

            logger.info(
                "OCR complete",
                extra={
                    "file": path.name,
                    "profile": profile.name,
                    "pages": len(pages),
                    "tokens": sum(len(page.tokens) for page in pages),
                    "chars": sum(page.char_count for page in pages),
                    "seconds": ocr_s,
                },
            )
        finally:
            self._preprocessor.cleanup(page_paths)

        self._cache.save(key, pages, profile_timings)
        return pages

    @staticmethod
    def _record_stage(
        timings: dict[str, float],
        profile: RenderProfile,
        stage: str,
        value: float,
        profile_timings: dict[str, float],
    ) -> None:
        timings[f"{profile.name}_{stage}"] = value
        profile_timings[stage] = value
        if profile.name == "primary":
            timings[stage] = value
        metrics.STAGE_LATENCY.labels(stage=stage.removesuffix("_s")).observe(value)

    # ---------------------------------------------------------- orientation --
    async def _correct_orientation(
        self,
        path: Path,
        photo_mode: bool | None,
        ocr_pages: list[OCRPage],
        timings: dict[str, float],
        warnings: list[str],
    ) -> tuple[list[OCRPage], set[int]]:
        """Re-OCR upright when the first pass came back sideways.

        PaddleOCR's document-orientation classifier misses some scans. When it
        does, every "row" is a vertical strip of one or two characters and the
        invoice yields nothing at all, so the extra pass pays for itself many
        times over. Only pages that actually look rotated pay it.
        """
        if not self._settings.correct_page_orientation:
            return ocr_pages, {0}

        evidence = orientation.assess(ocr_pages)
        rotations = orientation.candidate_rotations(evidence)
        if not rotations:
            return ocr_pages, {0}

        logger.info(
            "page looks sideways; re-running OCR upright",
            extra={"file": path.name, "tall_fraction": round(evidence.tall_fraction, 2)},
        )
        best_pages, best_quality, best_angle = ocr_pages, evidence.quality, 0
        tried = {0}

        for angle in rotations:
            try:
                candidate = await self._ocr_pages(
                    path, photo_mode, timings, RenderProfile(f"rot{angle}", rotate_degrees=angle)
                )
            except Exception as exc:  # noqa: BLE001 - a failed probe must not fail the invoice
                logger.warning(
                    "orientation probe failed",
                    extra={"file": path.name, "angle": angle, "error": str(exc)},
                )
                continue

            tried.add(angle % 360)
            quality = orientation.assess(candidate).quality
            if quality > best_quality:
                best_pages, best_quality, best_angle = candidate, quality, angle
            # An upright page reads so much better than a sideways one that a
            # clear win means the other direction is not worth trying.
            if best_angle and best_quality > evidence.quality * 1.5:
                break

        timings["orientation_degrees"] = float(best_angle)
        if best_angle:
            logger.info("orientation corrected", extra={"file": path.name, "degrees": best_angle})
        else:
            warnings.append("page looked rotated but no orientation read better than the original")
        return best_pages, tried

    async def _recover_by_rotation(
        self,
        path: Path,
        photo_mode: bool | None,
        ocr_pages: list[OCRPage],
        parsed: ParsedInvoice,
        tried_angles: set[int],
        timings: dict[str, float],
    ) -> tuple[list[OCRPage], ParsedInvoice]:
        """Last resort for a page that parsed to nothing: try other rotations.

        An upside-down invoice is the case this exists for. Its boxes keep
        normal proportions, so the aspect-ratio test sees nothing wrong, and the
        recognizer happily returns confident-looking nonsense. Parsing zero rows
        from non-empty text is the symptom that gives it away.

        Selection is on rows parsed, not OCR confidence: confidence rates how
        sure the recognizer is about characters, which is exactly what misleads
        here.
        """
        if not self._settings.correct_page_orientation:
            return ocr_pages, parsed

        budget = self._settings.max_orientation_probes
        angles = orientation.recovery_rotations(tried_angles)[:budget]
        if not angles:
            return ocr_pages, parsed

        logger.info(
            "no rows parsed; sweeping other orientations",
            extra={"file": path.name, "angles": list(angles)},
        )
        for angle in angles:
            try:
                candidate = await self._ocr_pages(
                    path, photo_mode, timings, RenderProfile(f"rot{angle}", rotate_degrees=angle)
                )
                candidate_parsed = self._parser.parse(candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "orientation recovery failed",
                    extra={"file": path.name, "angle": angle, "error": str(exc)},
                )
                continue

            if candidate_parsed.items:
                timings["orientation_degrees"] = float(angle)
                logger.info(
                    "orientation recovered rows",
                    extra={"file": path.name, "degrees": angle, "rows": len(candidate_parsed.items)},
                )
                return candidate, candidate_parsed

        return ocr_pages, parsed

    # ---------------------------------------------------------------- retry --
    def _retry_profiles(self) -> list[RenderProfile]:
        settings = self._settings
        profiles = [
            RenderProfile(
                "retry",
                settings.pdf_retry_render_dpi,
                settings.pdf_retry_target_long_side,
                settings.pdf_retry_max_long_side,
            )
        ]
        deep = RenderProfile(
            "retry_deep",
            settings.pdf_deep_retry_render_dpi,
            settings.pdf_deep_retry_target_long_side,
            settings.pdf_deep_retry_max_long_side,
        )
        if settings.enable_deep_ocr_retry and (
            deep.pdf_render_dpi,
            deep.target_long_side,
            deep.max_long_side,
        ) != (
            profiles[0].pdf_render_dpi,
            profiles[0].target_long_side,
            profiles[0].max_long_side,
        ):
            profiles.append(deep)
        return profiles

    def _needs_retry(self, path: Path, parsed: ParsedInvoice) -> bool:
        """Re-render at higher DPI only when the cheap pass looks suspicious.

        The ordering here matters for cost. A retry re-renders and re-OCRs the
        whole document, which is by far the most expensive thing this service
        does — so it is only worth paying when something is actually wrong.

        In particular, a passing checksum is independent arithmetic proof that
        the extraction is right, and no amount of extra DPI improves on "right".
        Retrying such documents merely because some line produced a soft warning
        tripled the cost of a typical PDF for no benefit.
        """
        if not self._settings.retry_ocr_on_review_flags or path.suffix.lower() != ".pdf":
            return False
        if not parsed.items:
            return True

        checksum = self._checksum(parsed)
        if checksum is True:
            return False
        if checksum is False:
            return True
        # No printed total to verify against: warnings are the only signal left.
        return bool(parsed.warnings)

    def _checksum(self, parsed: ParsedInvoice) -> bool | None:
        return rules.checksum_ok(
            rules.items_total(parsed.items), parsed.invoice_total, self._settings.checksum_tolerance
        )

    def _score(self, parsed: ParsedInvoice) -> int:
        """Rank two OCR attempts. Checksum agreement dominates row count."""
        score = len(parsed.items) * 10
        checksum = self._checksum(parsed)
        if checksum is True:
            score += 100
        elif checksum is False:
            score -= 100
        return score - len(parsed.warnings) * 3

    def _is_clean(self, parsed: ParsedInvoice) -> bool:
        return bool(parsed.items) and not parsed.warnings and self._checksum(parsed) is True

    async def _retry_ocr(
        self,
        path: Path,
        photo_mode: bool | None,
        ocr_pages: list[OCRPage],
        parsed: ParsedInvoice,
        timings: dict[str, float],
        warnings: list[str],
    ) -> tuple[list[OCRPage], ParsedInvoice]:
        best_score = self._score(parsed)
        selected = False

        for profile in self._retry_profiles():
            try:
                retry_pages = await self._ocr_pages(path, photo_mode, timings, profile)
                retry_parsed = self._parser.parse(retry_pages)
            except Exception as exc:  # noqa: BLE001 - a failed retry must not fail the invoice
                logger.warning(
                    "OCR retry failed",
                    extra={"file": path.name, "profile": profile.name, "error": str(exc)},
                )
                warnings.append(f"{profile.name} OCR retry failed: {exc}")
                continue

            retry_score = self._score(retry_parsed)
            if retry_score > best_score:
                best_score = retry_score
                ocr_pages, parsed = retry_pages, retry_parsed
                selected = True
                logger.info(
                    "OCR retry produced a better parse",
                    extra={"file": path.name, "profile": profile.name, "score": retry_score},
                )
            if self._is_clean(parsed):
                break

        timings["ocr_retry_used"] = 1.0 if selected else 0.0
        return ocr_pages, parsed

    # ------------------------------------------------------------------ llm --
    async def _maybe_map_with_llm(
        self,
        path: Path,
        mode: ExtractionMode,
        ocr_pages: list[OCRPage],
        raw_items: list[dict[str, Any]],
        invoice_total: float | None,
        warnings: list[str],
        timings: dict[str, float],
    ) -> list[dict[str, Any]]:
        tolerance = self._settings.checksum_tolerance
        parser_checksum = rules.checksum_ok(rules.items_total(raw_items), invoice_total, tolerance)

        should_use_llm = mode is ExtractionMode.llm or (
            mode is ExtractionMode.hybrid and (not raw_items or parser_checksum is False)
        )
        if not should_use_llm:
            return raw_items

        start = time.perf_counter()
        try:
            llm_items = await self._mapper.map_pages(ocr_pages)
        except Exception as exc:
            logger.warning("LLM mapping failed", extra={"file": path.name, "error": str(exc)})
            metrics.LLM_CALLS.labels(outcome="failed").inc()
            if not raw_items and mode is ExtractionMode.llm:
                raise
            warnings.append(f"LLM fallback failed: {exc}")
            return raw_items

        timings["mapping_s"] = round(time.perf_counter() - start, 2)
        llm_checksum = rules.checksum_ok(rules.items_total(llm_items), invoice_total, tolerance)
        logger.info(
            "LLM mapping candidate",
            extra={"file": path.name, "items": len(llm_items), "checksum_ok": llm_checksum},
        )

        # Accept the LLM only when it beats the parser: the parser is
        # reproducible, the model is not.
        accept = (
            mode is ExtractionMode.llm
            or not raw_items
            or (llm_checksum is True and parser_checksum is not True)
        )
        if accept:
            metrics.LLM_CALLS.labels(outcome="accepted").inc()
            warnings.append("LLM mapping used; checksum validation still required")
            return llm_items

        metrics.LLM_CALLS.labels(outcome="rejected").inc()
        warnings.append("LLM mapping was tested but rejected because the parser result scored better")
        return raw_items

    # ------------------------------------------------------------- assembly --
    def _validate_items(self, raw_items: list[dict[str, Any]], warnings: list[str]) -> list[InvoiceItem]:
        items: list[InvoiceItem] = []
        for index, raw in enumerate(raw_items, start=1):
            try:
                item = InvoiceItem.model_validate(raw)
            except ValidationError as exc:
                reasons = "; ".join(error["msg"] for error in exc.errors())
                warnings.append(f"item {index} dropped ({reasons}): {raw!r}")
                continue

            issue = rules.item_quality_issue(item.REFCODE, item.MED, item.prix)
            if issue:
                warnings.append(f"item {index} dropped ({issue}): {raw!r}")
                continue
            items.append(item)
        return items

    def _finalize(
        self,
        path: Path,
        raw_items: list[dict[str, Any]],
        warnings: list[str],
        lines: list[str],
        ocr_pages: list[OCRPage],
        mode: ExtractionMode,
        timings: dict[str, float],
        total_start: float,
    ) -> InvoiceResult:
        items = self._validate_items(raw_items, warnings)
        if raw_items and not items:
            warnings.append("all mapped items failed validation")
        if not raw_items and lines:
            warnings.append("no invoice items found despite non-empty OCR text")

        invoice_total = rules.extract_invoice_total(lines)
        items_total = round(sum(float(item.prix or 0) for item in items), 3)
        checksum = rules.checksum_ok(items_total, invoice_total, self._settings.checksum_tolerance)

        if checksum is False:
            warnings.append(
                f"checksum failed: item total {items_total:.3f} differs from "
                f"invoice total {invoice_total:.3f}"
            )
        elif invoice_total is None:
            warnings.append("invoice total could not be detected, so the checksum was skipped")

        elapsed = round(time.perf_counter() - total_start, 2)
        timings["total_s"] = elapsed
        status = InvoiceResult.derive_status(error=None, items=items, checksum_ok=checksum, warnings=warnings)

        metrics.PIPELINE_LATENCY.observe(elapsed)
        metrics.ITEMS_EXTRACTED.observe(len(items))
        metrics.INVOICES_PROCESSED.labels(status=status.value).inc()

        logger.info(
            "invoice complete",
            extra={
                "file": path.name,
                "status": status.value,
                "items": len(items),
                "checksum_ok": checksum,
                "warnings": len(warnings),
                "seconds": elapsed,
            },
        )
        return InvoiceResult(
            source_file=path.name,
            status=status,
            items=items,
            warnings=warnings,
            invoice_total=invoice_total,
            items_total=items_total,
            checksum_ok=checksum,
            page_count=len(ocr_pages),
            extraction_mode=mode.value,
            ocr_engine=self._engine.name,
            service_version=__version__,
            timings=timings,
        )

    def _failed(
        self,
        path: Path,
        error: str,
        warnings: list[str],
        timings: dict[str, float],
        total_start: float,
    ) -> InvoiceResult:
        elapsed = round(time.perf_counter() - total_start, 2)
        timings["total_s"] = elapsed
        metrics.PIPELINE_LATENCY.observe(elapsed)
        metrics.INVOICES_PROCESSED.labels(status=ReviewStatus.failed.value).inc()
        return InvoiceResult(
            source_file=path.name,
            status=ReviewStatus.failed,
            error=error,
            warnings=warnings,
            extraction_mode=self._settings.extraction_mode.value,
            ocr_engine=self._engine.name,
            service_version=__version__,
            timings=timings,
        )
