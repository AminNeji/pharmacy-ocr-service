"""OCR engines, result types, and the input-level OCR cache."""

from __future__ import annotations

from pharmacy_ocr.ocr.cache import CacheKey, OCRCache
from pharmacy_ocr.ocr.engine import OCREngine, PaddleOCREngine, create_engine
from pharmacy_ocr.ocr.types import OCRPage, OCRToken

__all__ = [
    "CacheKey",
    "OCRCache",
    "OCREngine",
    "OCRPage",
    "OCRToken",
    "PaddleOCREngine",
    "create_engine",
]
