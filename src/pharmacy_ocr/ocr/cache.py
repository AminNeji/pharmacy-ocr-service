"""Input-level OCR cache.

OCR is by far the most expensive stage, so caching its output lets parser and
prompt work iterate in seconds while replaying byte-identical OCR text.

Two things changed from the previous implementation and both matter:

1. The cache key no longer includes the file's mtime. Content hash plus size is
   already an exact identity check, and mtime made every copied, re-downloaded,
   or OneDrive-resynced invoice miss the cache for no reason.
2. Writes are atomic (temp file + atomic rename). A crash mid-write previously
   left truncated JSON that every later run had to fail to parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharmacy_ocr.observability import metrics
from pharmacy_ocr.ocr.engine import resolve_device
from pharmacy_ocr.ocr.types import Box, OCRPage, OCRToken, Polygon
from pharmacy_ocr.settings import Settings

logger = logging.getLogger(__name__)

_HASH_CHUNK = 1024 * 1024
#: Run eviction once every N writes rather than on each one; listing a large
#: cache directory is far more expensive than the write itself.
_PRUNE_EVERY = 100


@dataclass(frozen=True, slots=True)
class CacheKey:
    digest: str
    metadata: dict[str, Any]


def file_digest(path: Path) -> tuple[str, int]:
    """Return (sha256, size) for `path`, streamed so large PDFs stay off-heap."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


class OCRCache:
    """Content-addressed store of OCR pages keyed by file bytes + OCR settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._writes = 0
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._settings.ocr_cache_dir

    @property
    def enabled(self) -> bool:
        return self._settings.use_ocr_cache

    # ------------------------------------------------------------------ keys --
    def _settings_fingerprint(
        self,
        *,
        photo_mode: bool | None,
        pdf_render_dpi: int | None,
        target_long_side: int | None,
        max_long_side: int | None,
        rotate_degrees: int = 0,
    ) -> dict[str, Any]:
        settings = self._settings
        return {
            "rotate_degrees": rotate_degrees % 360,
            "cache_version": settings.ocr_cache_version,
            "ocr_engine": settings.ocr_engine.value,
            # The resolved device, not the configured string: "auto" and "cpu"
            # produce the same OCR on a CPU-only machine and must share cache
            # entries, while a real GPU switch still invalidates them.
            "ocr_device": resolve_device(settings.ocr_device),
            "ocr_lang": settings.ocr_lang,
            "ocr_version": settings.ocr_version,
            "det_model": settings.ocr_text_detection_model_name,
            "rec_model": settings.ocr_text_recognition_model_name,
            "min_confidence": settings.ocr_min_confidence,
            "doc_orientation": settings.use_doc_orientation_classify,
            "doc_unwarping": settings.use_doc_unwarping,
            "textline_orientation": settings.use_textline_orientation,
            "adaptive_threshold": settings.apply_adaptive_threshold,
            "photo_mode": settings.photo_mode if photo_mode is None else photo_mode,
            "pdf_render_dpi": pdf_render_dpi or settings.pdf_render_dpi,
            "target_long_side": target_long_side or settings.target_long_side,
            "max_long_side": max_long_side or settings.max_long_side,
            "pdf_max_pages": settings.pdf_max_pages,
        }

    def key_for(
        self,
        input_path: str | Path,
        *,
        photo_mode: bool | None = None,
        pdf_render_dpi: int | None = None,
        target_long_side: int | None = None,
        max_long_side: int | None = None,
        rotate_degrees: int = 0,
    ) -> CacheKey:
        path = Path(input_path)
        digest, size = file_digest(path)
        payload = {
            # `name` is recorded for debugging but deliberately excluded from
            # the hash: identical bytes must hit regardless of filename.
            "file": {"sha256": digest, "size": size},
            "settings": self._settings_fingerprint(
                photo_mode=photo_mode,
                pdf_render_dpi=pdf_render_dpi,
                target_long_side=target_long_side,
                max_long_side=max_long_side,
                rotate_degrees=rotate_degrees,
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        metadata = {**payload, "source_name": path.name}
        return CacheKey(digest=hashlib.sha256(encoded).hexdigest(), metadata=metadata)

    def path_for(self, digest: str) -> Path:
        return self.directory / f"{digest}.json"

    # ------------------------------------------------------------------- io ---
    def load(self, key: CacheKey) -> list[OCRPage] | None:
        if self._settings.force_ocr_refresh or not self.enabled:
            return None
        path = self.path_for(key.digest)
        if not path.exists():
            metrics.CACHE_EVENTS.labels(event="miss").inc()
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pages = [_page_from_dict(page) for page in data.get("pages", [])]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # A corrupt entry must never break a run: drop it and re-OCR.
            logger.warning("discarding unreadable OCR cache entry", extra={"cache_file": path.name})
            metrics.CACHE_EVENTS.labels(event="error").inc()
            path.unlink(missing_ok=True)
            return None
        metrics.CACHE_EVENTS.labels(event="hit").inc()
        logger.info("OCR cache hit", extra={"cache_file": path.name, "pages": len(pages)})
        return pages

    def save(self, key: CacheKey, pages: list[OCRPage], timings: dict[str, float]) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key.digest)
        payload = {
            "metadata": key.metadata,
            "cached_at": time.time(),
            "timings": timings,
            "pages": [_page_to_dict(page) for page in pages],
        }
        temp_path = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(path)  # atomic within the same filesystem
        except OSError:
            temp_path.unlink(missing_ok=True)
            logger.warning("could not write OCR cache entry", extra={"cache_file": path.name})
            metrics.CACHE_EVENTS.labels(event="error").inc()
            return None

        metrics.CACHE_EVENTS.labels(event="write").inc()
        logger.info("OCR cache write", extra={"cache_file": path.name})
        self._maybe_prune()
        return path

    @staticmethod
    def load_file(path: Path) -> tuple[dict[str, Any], list[OCRPage]]:
        """Read a cache entry directly, for offline analysis scripts."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("metadata", {}), [_page_from_dict(page) for page in data.get("pages", [])]

    # -------------------------------------------------------------- eviction --
    def _maybe_prune(self) -> None:
        with self._lock:
            self._writes += 1
            due = self._writes % _PRUNE_EVERY == 0
        if due:
            self.prune()

    def prune(self) -> int:
        """Drop expired and surplus entries. Returns the number removed."""
        settings = self._settings
        if not self.directory.exists():
            return 0

        try:
            entries = [(entry, entry.stat().st_mtime) for entry in self.directory.glob("*.json")]
        except OSError:
            return 0

        removed = 0
        keep: list[tuple[Path, float]] = []
        if settings.ocr_cache_ttl_days > 0:
            cutoff = time.time() - settings.ocr_cache_ttl_days * 86400
            for entry, mtime in entries:
                if mtime < cutoff:
                    entry.unlink(missing_ok=True)
                    removed += 1
                else:
                    keep.append((entry, mtime))
        else:
            keep = entries

        if settings.ocr_cache_max_entries > 0 and len(keep) > settings.ocr_cache_max_entries:
            keep.sort(key=lambda item: item[1])  # oldest first
            for entry, _ in keep[: len(keep) - settings.ocr_cache_max_entries]:
                entry.unlink(missing_ok=True)
                removed += 1

        if removed:
            metrics.CACHE_EVENTS.labels(event="evict").inc(removed)
            logger.info("pruned OCR cache", extra={"removed": removed})
        return removed


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _token_to_dict(token: OCRToken) -> dict[str, Any]:
    return {
        "text": token.text,
        "score": token.score,
        "box": list(token.box) if token.box else None,
        "polygon": [list(point) for point in token.polygon] if token.polygon else None,
        "page_index": token.page_index,
    }


def _page_to_dict(page: OCRPage) -> dict[str, Any]:
    return {
        "text": page.text,
        "tokens": [_token_to_dict(token) for token in page.tokens],
        "source_image": page.source_image,
        "page_index": page.page_index,
    }


def _box_from_list(raw: Any) -> Box | None:
    if not raw:
        return None
    values = [float(value) for value in raw]
    if len(values) < 4:
        return None
    return (values[0], values[1], values[2], values[3])


def _polygon_from_list(raw: Any) -> Polygon | None:
    if not raw:
        return None
    points = tuple((float(point[0]), float(point[1])) for point in raw if len(point) >= 2)
    return points or None


def _token_from_dict(data: dict[str, Any]) -> OCRToken:
    return OCRToken(
        text=str(data.get("text") or ""),
        score=data.get("score"),
        box=_box_from_list(data.get("box")),
        polygon=_polygon_from_list(data.get("polygon")),
        page_index=int(data.get("page_index") or 0),
    )


def _page_from_dict(data: dict[str, Any]) -> OCRPage:
    return OCRPage(
        text=str(data.get("text") or ""),
        tokens=[_token_from_dict(token) for token in data.get("tokens", [])],
        source_image=str(data.get("source_image") or "cache"),
        page_index=int(data.get("page_index") or 0),
    )
