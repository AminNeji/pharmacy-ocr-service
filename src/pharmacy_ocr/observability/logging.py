"""Structured logging.

Emits one JSON object per line in production so log aggregators can index
fields directly, and a human-readable format in development. Every record
carries the current request/job id, which is what makes a single invoice
traceable across the API thread, the preprocessing worker, and the OCR thread.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import logging.handlers
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pharmacy_ocr.settings import Settings

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

#: Attributes present on every LogRecord; anything else was supplied by the
#: caller via `extra=` and is worth emitting as a structured field.
_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

_NOISY_LOGGERS = ("PIL", "ppocr", "paddleocr", "paddlex", "httpx", "httpcore", "urllib3")


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def bind_request_id(request_id: str) -> Iterator[str]:
    """Attach `request_id` to every log record emitted inside the block."""
    token = _request_id.set(request_id)
    try:
        yield request_id
    finally:
        _request_id.reset(token)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with caller-supplied `extra` fields inlined."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key in payload or key.startswith("_"):
                continue
            payload[key] = value if _is_json_safe(value) else repr(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Compact human-readable format for local development."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        request_id = getattr(record, "request_id", None)
        return f"{base} [req={request_id}]" if request_id else base


def _is_json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))


def configure_logging(settings: Settings, *, force: bool = False) -> None:
    """Configure root logging. Idempotent: safe to call from API and CLI."""
    root = logging.getLogger()
    if root.handlers and not force:
        return
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(settings.log_level)
    formatter: logging.Formatter = JsonFormatter() if settings.log_format == "json" else TextFormatter()
    request_filter = _RequestIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(request_filter)
    root.addHandler(console)

    if settings.log_to_file:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logs_dir / "pipeline.log",
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_filter)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
