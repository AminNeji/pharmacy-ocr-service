"""Core domain types shared by every layer of the service."""

from __future__ import annotations

from pharmacy_ocr.domain.errors import (
    ConfigurationError,
    CorruptDocumentError,
    ExtractionError,
    FileTooLargeError,
    JobNotFoundError,
    LLMMappingError,
    OCREngineError,
    PharmacyOCRError,
    RateLimitedError,
    ServiceOverloadedError,
    UnsupportedFileTypeError,
)
from pharmacy_ocr.domain.models import (
    ExtractionEnvelope,
    InvoiceItem,
    InvoiceResult,
    ParsedInvoice,
    ReviewStatus,
    clean_refcode,
    is_integral,
    to_number,
)

__all__ = [
    "ConfigurationError",
    "CorruptDocumentError",
    "ExtractionEnvelope",
    "ExtractionError",
    "FileTooLargeError",
    "InvoiceItem",
    "InvoiceResult",
    "JobNotFoundError",
    "LLMMappingError",
    "OCREngineError",
    "ParsedInvoice",
    "PharmacyOCRError",
    "RateLimitedError",
    "ReviewStatus",
    "ServiceOverloadedError",
    "UnsupportedFileTypeError",
    "clean_refcode",
    "is_integral",
    "to_number",
]
