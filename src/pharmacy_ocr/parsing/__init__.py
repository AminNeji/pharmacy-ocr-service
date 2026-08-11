"""Deterministic invoice row parsing and the optional LLM fallback."""

from __future__ import annotations

from pharmacy_ocr.parsing.numbers import PriceMatch, find_prices, nearly_equal
from pharmacy_ocr.parsing.rules import (
    RowParser,
    TextLine,
    checksum_ok,
    extract_invoice_total,
    items_total,
)

__all__ = [
    "PriceMatch",
    "RowParser",
    "TextLine",
    "checksum_ok",
    "extract_invoice_total",
    "find_prices",
    "items_total",
    "nearly_equal",
]
