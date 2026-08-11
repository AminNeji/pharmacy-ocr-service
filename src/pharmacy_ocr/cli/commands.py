"""Implementations behind the `pharmacy-ocr` subcommands."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pharmacy_ocr import __version__
from pharmacy_ocr.domain.models import InvoiceResult, ReviewStatus
from pharmacy_ocr.evaluation import Report, load_ground_truth, score_invoice
from pharmacy_ocr.ocr.cache import OCRCache
from pharmacy_ocr.parsing.rules import (
    RowParser,
    checksum_ok,
    extract_invoice_total,
    extract_lines,
    item_quality_issue,
    items_total,
)
from pharmacy_ocr.pipeline import InvoicePipeline, RenderProfile
from pharmacy_ocr.settings import SUPPORTED_EXTENSIONS, ExtractionMode, Settings

logger = logging.getLogger("pharmacy_ocr.cli")

EXIT_OK = 0
EXIT_NO_INPUT = 1
EXIT_SOME_FAILED = 2


@dataclass
class BatchStats:
    ok: int = 0
    needs_review: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return self.ok + self.needs_review + self.failed


def collect_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    if not input_dir.is_dir():
        return []
    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files[: max(0, limit)] if limit is not None else files


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
async def run_batch(
    settings: Settings,
    *,
    input_dir: Path,
    output_dir: Path,
    concurrency: int,
    photo_mode: bool | None,
    extraction_mode: ExtractionMode | None,
    limit: int | None,
    force: bool,
) -> int:
    """Process every supported file in `input_dir`, one JSON per invoice."""
    files = collect_files(input_dir, limit)
    if not files:
        logger.error("no supported files found", extra={"input_dir": str(input_dir)})
        return EXIT_NO_INPUT

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InvoicePipeline(settings)
    stats = BatchStats()
    queue: asyncio.Queue[Path] = asyncio.Queue()
    for path in files:
        queue.put_nowait(path)

    logger.info(
        "batch start",
        extra={
            "invoices": len(files),
            "concurrency": concurrency,
            "mode": (extraction_mode or settings.extraction_mode).value,
            "output_dir": str(output_dir),
            "force": force,
        },
    )
    start = time.perf_counter()

    async def worker() -> None:
        while True:
            try:
                path = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process_one(pipeline, path, output_dir, photo_mode, extraction_mode, force, stats)
            finally:
                queue.task_done()

    try:
        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.warning("batch interrupted; partial results were written")
    finally:
        await pipeline.aclose()

    elapsed = time.perf_counter() - start
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "service_version": __version__,
        "input_dir": str(input_dir),
        "invoices": len(files),
        "ok": stats.ok,
        "needs_review": stats.needs_review,
        "failed": stats.failed,
        "skipped": stats.skipped,
        "elapsed_s": round(elapsed, 1),
        "seconds_per_invoice": round(elapsed / max(1, stats.processed), 1),
    }
    (output_dir / "_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("batch done", extra=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return EXIT_SOME_FAILED if stats.failed else EXIT_OK


async def _process_one(
    pipeline: InvoicePipeline,
    path: Path,
    output_dir: Path,
    photo_mode: bool | None,
    extraction_mode: ExtractionMode | None,
    force: bool,
    stats: BatchStats,
) -> None:
    out_path = output_dir / f"{path.stem}.json"
    if out_path.exists() and not force:
        stats.skipped += 1
        logger.info("skipped (already extracted)", extra={"file": path.name})
        return

    result = await pipeline.process(path, photo_mode=photo_mode, extraction_mode=extraction_mode)
    _write_result(out_path, result)

    if result.status is ReviewStatus.failed:
        stats.failed += 1
        stats.errors.append(f"{path.name}: {result.error}")
        logger.warning("invoice failed", extra={"file": path.name, "error": result.error})
    elif result.status is ReviewStatus.needs_review:
        stats.needs_review += 1
    else:
        stats.ok += 1


def _write_result(out_path: Path, result: InvoiceResult) -> None:
    """Write atomically so an interrupted batch never leaves half a JSON file."""
    temp_path = out_path.with_suffix(".json.tmp")
    temp_path.write_text(
        result.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    temp_path.replace(out_path)


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
async def run_cache_warm(
    settings: Settings,
    *,
    input_dir: Path,
    files: list[str] | None,
    limit: int | None,
    photo_mode: bool | None,
    include_retry_profiles: bool,
) -> int:
    """Pre-compute OCR for invoices so later parser work is instant."""
    if files:
        targets = [Path(name) if Path(name).is_absolute() else input_dir / name for name in files]
    else:
        targets = collect_files(input_dir, limit)

    targets = [path for path in targets if path.is_file()]
    if not targets:
        logger.error("no supported files found", extra={"input_dir": str(input_dir)})
        return EXIT_NO_INPUT

    pipeline = InvoicePipeline(settings)
    profiles = [RenderProfile("primary")]
    if include_retry_profiles:
        profiles.extend(
            [
                RenderProfile(
                    "retry",
                    settings.pdf_retry_render_dpi,
                    settings.pdf_retry_target_long_side,
                    settings.pdf_retry_max_long_side,
                ),
                RenderProfile(
                    "retry_deep",
                    settings.pdf_deep_retry_render_dpi,
                    settings.pdf_deep_retry_target_long_side,
                    settings.pdf_deep_retry_max_long_side,
                ),
            ]
        )

    start = time.perf_counter()
    failures = 0
    try:
        for index, path in enumerate(targets, start=1):
            print(f"[{index}/{len(targets)}] {path.name}")
            for profile in profiles:
                if profile.name != "primary" and path.suffix.lower() != ".pdf":
                    continue
                try:
                    await pipeline.warm_cache(path, photo_mode=photo_mode, profile=profile)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.warning(
                        "cache warm failed",
                        extra={"file": path.name, "profile": profile.name, "error": str(exc)},
                    )
    finally:
        await pipeline.aclose()

    print(
        f"Cached {len(targets)} invoice(s) in {time.perf_counter() - start:.1f}s "
        f"at {settings.ocr_cache_dir} ({failures} failure(s))"
    )
    return EXIT_SOME_FAILED if failures else EXIT_OK


def run_cache_prune(settings: Settings) -> int:
    removed = OCRCache(settings).prune()
    print(f"Removed {removed} cache entr{'y' if removed == 1 else 'ies'} from {settings.ocr_cache_dir}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "ground_truth.json"


async def run_evaluate(
    settings: Settings,
    *,
    input_dir: Path,
    ground_truth: Path,
    photo_mode: bool | None = None,
    extraction_mode: ExtractionMode | None = None,
    json_out: Path | None = None,
) -> int:
    """Score the pipeline against hand-labelled invoices."""
    if not ground_truth.is_file():
        logger.error("ground truth not found", extra={"path": str(ground_truth)})
        return EXIT_NO_INPUT

    truth = load_ground_truth(ground_truth)
    if not truth:
        logger.error("ground truth file has no invoices")
        return EXIT_NO_INPUT

    pipeline = InvoicePipeline(settings)
    report = Report()
    try:
        for stem, entry in truth.items():
            matches = list(input_dir.glob(f"{stem}.*"))
            if not matches:
                logger.warning("labelled invoice is missing", extra={"stem": stem})
                continue
            started = time.perf_counter()
            result = await pipeline.process(
                matches[0], photo_mode=photo_mode, extraction_mode=extraction_mode
            )
            elapsed = time.perf_counter() - started
            report.add(
                score_invoice(
                    stem,
                    entry.get("items", []),
                    [item.model_dump() for item in result.items],
                    seconds=elapsed,
                )
            )
    finally:
        await pipeline.aclose()

    print(report.render())
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {json_out}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# diagnose
# --------------------------------------------------------------------------- #
async def run_diagnose(
    settings: Settings,
    *,
    invoice: Path,
    photo_mode: bool | None = None,
    show_all: bool = False,
) -> int:
    """Show what the parser decided about every line of one invoice.

    Runs off the OCR cache when it is warm, so iterating on parser rules costs
    seconds instead of a fresh OCR pass.
    """
    if not invoice.is_file():
        logger.error("invoice not found", extra={"file": str(invoice)})
        return EXIT_NO_INPUT

    pipeline = InvoicePipeline(settings)
    try:
        pages = await pipeline.warm_cache(invoice, photo_mode=photo_mode)
    finally:
        await pipeline.aclose()

    parser = RowParser(settings)
    lines = extract_lines(pages)
    verdicts = [parser.explain_line(line) for line in lines]

    # Apply the same post-parse quality gate the pipeline uses, otherwise this
    # report shows rows that never reach the output.
    kept: list[dict[str, Any]] = []
    dropped: dict[int, str] = {}
    for index, verdict in enumerate(verdicts):
        if verdict.item is None:
            continue
        issue = item_quality_issue(verdict.item["REFCODE"], verdict.item["MED"], verdict.item["prix"])
        if issue:
            dropped[index] = issue
        else:
            kept.append(verdict.item)
    items = kept

    tokens = sum(len(page.tokens) for page in pages)
    confidences = [page.mean_confidence for page in pages if page.mean_confidence is not None]
    print(f"\n{'=' * 78}")
    print(f"{invoice.name}")
    print(
        f"pages={len(pages)} tokens={tokens} lines={len(lines)} "
        f"mean_confidence={(sum(confidences) / len(confidences)):.3f}"
        if confidences
        else f"pages={len(pages)} tokens={tokens} lines={len(lines)}"
    )
    print(f"{'=' * 78}\n")

    for index, verdict in enumerate(verdicts):
        label = index + 1
        if verdict.item is not None:
            item = verdict.item
            tag = "DROP  " if index in dropped else "ITEM  "
            print(f"{label:3}. {tag} {verdict.text}")
            print(
                f"          -> REFCODE={item['REFCODE']} QT={item['QT']} "
                f"prix={item['prix']} unit={item['unit_price']} MED={item['MED']!r}"
            )
            if index in dropped:
                print(f"          xx dropped by validation: {dropped[index]}")
            elif verdict.warning:
                print(f"          !! {verdict.warning}")
        elif show_all:
            print(f"{label:3}. skip   {verdict.text}")
            print(f"          -> {verdict.skip_reason}")

    invoice_total = extract_invoice_total(lines)
    total = items_total(items)
    ok = checksum_ok(total, invoice_total, settings.checksum_tolerance)

    print(f"\n{'-' * 78}")
    print(f"items         : {len(items)}")
    print(f"items_total   : {total}")
    print(f"invoice_total : {invoice_total}")
    print(f"checksum_ok   : {ok}")
    if ok is False and invoice_total is not None:
        print(f"difference    : {round(total - invoice_total, 3)}")
    if dropped:
        print(f"dropped       : {len(dropped)} row(s) failed validation (shown as DROP above)")
    if not show_all:
        skipped = sum(1 for verdict in verdicts if verdict.item is None)
        print(f"({skipped} line(s) skipped; re-run with --show-all to see why)")
    print()
    return EXIT_OK


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
_WARNING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("quantity could not be determined", "quantity could not be determined"),
    ("checksum failed", "checksum failed"),
    ("invoice total could not be detected", "no invoice total printed/detected"),
    ("no invoice items found", "OCR text present but no rows parsed"),
    ("deterministic parser found no invoice rows", "parser found no rows"),
    ("OCR returned no text", "OCR returned no text"),
    ("all mapped items failed validation", "all items failed validation"),
    ("LLM mapping used", "LLM mapping used"),
    ("LLM mapping was tested but rejected", "LLM mapping rejected"),
    ("LLM fallback failed", "LLM fallback failed"),
    ("OCR retry failed", "OCR retry failed"),
)


def _warning_pattern(warning: str) -> str:
    """Collapse a warning to a stable category so they can be counted.

    Warnings embed the offending line and its numbers, so raw strings are
    almost all unique; the category is what shows a systematic problem.
    """
    for needle, label in _WARNING_PATTERNS:
        if needle in warning:
            return label
    if "dropped (" in warning:
        reason = warning.split("dropped (", 1)[1].split(")", 1)[0]
        return f"item dropped: {reason}"
    return warning[:60]


def run_audit(output_dir: Path, *, show_all: bool = False) -> int:
    """Summarize extraction outputs and list everything needing review."""
    paths = sorted(path for path in output_dir.glob("*.json") if not path.name.startswith("_"))
    if not paths:
        logger.error("no output JSON found", extra={"output_dir": str(output_dir)})
        return EXIT_NO_INPUT

    counter: Counter[str] = Counter()
    warning_patterns: Counter[str] = Counter()
    flagged: list[dict[str, object]] = []
    item_count = 0

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counter["unreadable"] += 1
            flagged.append({"file": path.name, "error": "output file is not valid JSON"})
            continue

        items = data.get("items", [])
        item_count += len(items)
        items_total = data.get("items_total")
        if items_total is None:
            items_total = round(sum(float(item.get("prix") or 0) for item in items), 3)

        status = data.get("status") or ("failed" if data.get("error") else "unknown")
        counter[status] += 1
        checksum = data.get("checksum_ok")
        counter[f"checksum_{checksum}"] += 1

        warnings = data.get("warnings") or []
        for warning in warnings:
            warning_patterns[_warning_pattern(str(warning))] += 1

        row: dict[str, object] = {
            "file": path.name,
            "status": status,
            "items": len(items),
            "items_total": items_total,
            "invoice_total": data.get("invoice_total"),
            "checksum_ok": checksum,
            "warnings": len(warnings),
            "error": data.get("error"),
        }
        if checksum is False and data.get("invoice_total") is not None:
            row["difference"] = round(float(items_total) - float(data["invoice_total"]), 3)
        if show_all or status != ReviewStatus.ok.value:
            flagged.append(row)

    total = len(paths)
    ok = counter.get(ReviewStatus.ok.value, 0)
    print(
        f"files={total} ok={ok} "
        f"needs_review={counter.get(ReviewStatus.needs_review.value, 0)} "
        f"failed={counter.get(ReviewStatus.failed.value, 0)} ({ok / total:.0%} clean)"
    )
    print(
        f"checksum: passed={counter.get('checksum_True', 0)} "
        f"failed={counter.get('checksum_False', 0)} "
        f"unknown={counter.get('checksum_None', 0)} | items extracted: {item_count}"
    )

    if warning_patterns:
        print("\ntop warning categories:")
        for label, count in warning_patterns.most_common(12):
            print(f"  {count:5}  {label}")

    if flagged:
        print(f"\n{len(flagged)} invoice(s) needing attention:")
        for row in flagged:
            print(json.dumps(row, ensure_ascii=False))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def run_show_config(settings: Settings) -> int:
    """Print the effective configuration, with secrets redacted."""
    payload = settings.model_dump(mode="json")
    payload["api_keys_raw"] = f"<{len(settings.api_keys)} key(s) configured>"
    payload["resolved_paths"] = {
        "invoices_dir": str(settings.invoices_dir),
        "outputs_dir": str(settings.outputs_dir),
        "logs_dir": str(settings.logs_dir),
        "temp_dir": str(settings.temp_dir),
        "ocr_cache_dir": str(settings.ocr_cache_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return EXIT_OK
