"""`pharmacy-ocr` command-line entry point.

    pharmacy-ocr serve
    pharmacy-ocr batch --limit 5 --mode deterministic
    pharmacy-ocr cache warm --retry-profiles
    pharmacy-ocr audit
    pharmacy-ocr config

Every subcommand accepts overrides that used to require editing config.py or
exporting environment variables; the overrides build a `Settings` copy rather
than mutating global state.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from pharmacy_ocr import __version__
from pharmacy_ocr.cli import commands
from pharmacy_ocr.observability.logging import configure_logging
from pharmacy_ocr.settings import ExtractionMode, Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharmacy-ocr",
        description="Extract medicine line items from Tunisian pharmacy invoices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="override LOG_LEVEL for this run",
    )
    parser.add_argument("--log-format", choices=("json", "text"), help="override LOG_FORMAT for this run")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- serve -------------------------------------------------------------
    serve = subparsers.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", help="bind address (default: from settings)")
    serve.add_argument("--port", type=int, help="bind port (default: from settings)")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev only)")

    # --- batch -------------------------------------------------------------
    batch = subparsers.add_parser("batch", help="process a folder of invoices to JSON")
    batch.add_argument("--input", type=Path, help="input folder (default: <data>/invoices)")
    batch.add_argument("--output", type=Path, help="output folder (default: <data>/outputs)")
    batch.add_argument("--concurrency", type=int, help="parallel invoices (OCR is serialized inside)")
    batch.add_argument("--limit", type=int, help="process only the first N files")
    batch.add_argument("--mode", type=ExtractionMode, choices=list(ExtractionMode), help="extraction mode")
    batch.add_argument(
        "--force", action="store_true", help="reprocess invoices that already have output JSON"
    )
    _add_photo_mode(batch)

    # --- cache -------------------------------------------------------------
    cache = subparsers.add_parser("cache", help="manage the OCR cache")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)

    warm = cache_sub.add_parser("warm", help="pre-compute OCR for invoices")
    warm.add_argument("--input", type=Path, help="input folder (default: <data>/invoices)")
    warm.add_argument("--files", nargs="*", help="specific filenames under --input, or absolute paths")
    warm.add_argument("--limit", type=int, help="warm only the first N files")
    warm.add_argument(
        "--retry-profiles", action="store_true", help="also cache the higher-DPI PDF retry renders"
    )
    _add_photo_mode(warm)

    cache_sub.add_parser("prune", help="drop expired and surplus cache entries")

    # --- evaluate ----------------------------------------------------------
    evaluate = subparsers.add_parser("evaluate", help="score extraction against hand-labelled ground truth")
    evaluate.add_argument("--input", type=Path, help="invoice folder (default: <data>/invoices)")
    evaluate.add_argument("--ground-truth", type=Path, help="labels JSON")
    evaluate.add_argument("--json-out", type=Path, help="write the summary as JSON")
    evaluate.add_argument("--mode", type=ExtractionMode, choices=list(ExtractionMode), help="extraction mode")
    _add_photo_mode(evaluate)

    # --- diagnose ----------------------------------------------------------
    diagnose = subparsers.add_parser(
        "diagnose", help="show what the parser decided about every line of one invoice"
    )
    diagnose.add_argument("invoice", type=Path, help="path to a single invoice file")
    diagnose.add_argument(
        "--show-all", action="store_true", help="also list skipped lines and why they were skipped"
    )
    _add_photo_mode(diagnose)

    # --- audit -------------------------------------------------------------
    audit = subparsers.add_parser("audit", help="summarize extraction outputs")
    audit.add_argument("--output", type=Path, help="output folder to audit")
    audit.add_argument("--all", action="store_true", help="list every invoice, not just flagged ones")

    # --- config ------------------------------------------------------------
    subparsers.add_parser("config", help="print the effective configuration")

    return parser


def _add_photo_mode(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--photo-mode",
        dest="photo_mode",
        action="store_true",
        default=None,
        help="force heavy preprocessing (phone photos)",
    )
    group.add_argument(
        "--scan-mode",
        dest="photo_mode",
        action="store_false",
        help="force light preprocessing (clean scans)",
    )


def _resolve_settings(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides by copying settings, never by mutating globals."""
    overrides: dict[str, Any] = {}
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level
    if getattr(args, "log_format", None):
        overrides["log_format"] = args.log_format
    if getattr(args, "host", None):
        overrides["host"] = args.host
    if getattr(args, "port", None):
        overrides["port"] = args.port

    settings = get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _resolve_settings(args)
    configure_logging(settings, force=True)

    if args.command == "serve":
        return _serve(settings, reload=args.reload)

    if args.command == "batch":
        return asyncio.run(
            commands.run_batch(
                settings,
                input_dir=args.input or settings.invoices_dir,
                output_dir=args.output or settings.outputs_dir,
                concurrency=args.concurrency or settings.batch_concurrency,
                photo_mode=args.photo_mode,
                extraction_mode=args.mode,
                limit=args.limit,
                force=args.force,
            )
        )

    if args.command == "cache":
        if args.cache_command == "warm":
            return asyncio.run(
                commands.run_cache_warm(
                    settings,
                    input_dir=args.input or settings.invoices_dir,
                    files=args.files,
                    limit=args.limit,
                    photo_mode=args.photo_mode,
                    include_retry_profiles=args.retry_profiles,
                )
            )
        return commands.run_cache_prune(settings)

    if args.command == "evaluate":
        return asyncio.run(
            commands.run_evaluate(
                settings,
                input_dir=args.input or settings.invoices_dir,
                ground_truth=args.ground_truth or commands.DEFAULT_GROUND_TRUTH,
                photo_mode=args.photo_mode,
                extraction_mode=args.mode,
                json_out=args.json_out,
            )
        )

    if args.command == "diagnose":
        return asyncio.run(
            commands.run_diagnose(
                settings,
                invoice=args.invoice,
                photo_mode=args.photo_mode,
                show_all=args.show_all,
            )
        )

    if args.command == "audit":
        return commands.run_audit(args.output or settings.outputs_dir, show_all=args.all)

    return commands.run_show_config(settings)


def _serve(settings: Settings, *, reload: bool) -> int:
    import uvicorn

    tls: dict[str, object] = {}
    if settings.tls_enabled:
        tls = {
            "ssl_certfile": str(settings.ssl_certfile),
            "ssl_keyfile": str(settings.ssl_keyfile),
        }
        print(f"serving HTTPS on https://{settings.host}:{settings.port}")
    else:
        print(f"serving HTTP on http://{settings.host}:{settings.port}")

    uvicorn.run(
        "pharmacy_ocr.asgi:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        # Access logging, request ids, and metrics come from our middleware;
        # uvicorn's own access log would duplicate every line unformatted.
        access_log=False,
        log_config=None,
        **tls,  # type: ignore[arg-type]
    )
    return commands.EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
