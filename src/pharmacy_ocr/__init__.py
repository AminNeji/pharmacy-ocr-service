"""Pharmacy invoice OCR service.

Extracts medicine line items from Tunisian pharmacy invoices using OpenCV
preprocessing, PaddleOCR text detection, and a deterministic row parser, with
an optional local LLM fallback for layouts the parser cannot handle.
"""

from __future__ import annotations

__version__ = "3.3.0"

__all__ = ["__version__"]
