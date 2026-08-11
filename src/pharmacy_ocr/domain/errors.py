"""Exception hierarchy.

Every failure the service can produce on purpose has a class here with a stable
machine-readable `code`. The API layer maps codes to HTTP statuses in exactly
one place, so error responses cannot drift between endpoints.
"""

from __future__ import annotations


class PharmacyOCRError(Exception):
    """Base class for all errors raised by this service."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigurationError(PharmacyOCRError):
    """The service is misconfigured and cannot serve traffic."""

    code = "configuration_error"
    http_status = 500


class UnsupportedFileTypeError(PharmacyOCRError):
    """Upload extension is not in the allow-list."""

    code = "unsupported_file_type"
    http_status = 415


class FileTooLargeError(PharmacyOCRError):
    """Upload exceeded `max_upload_bytes`."""

    code = "file_too_large"
    http_status = 413


class CorruptDocumentError(PharmacyOCRError):
    """File is empty, truncated, not the type it claims, or undecodable."""

    code = "corrupt_document"
    http_status = 400


class OCREngineError(PharmacyOCRError):
    """The OCR engine failed to load or to process a page."""

    code = "ocr_failed"
    http_status = 502


class LLMMappingError(PharmacyOCRError):
    """The LLM fallback did not return usable JSON."""

    code = "llm_mapping_failed"
    http_status = 502


class ExtractionError(PharmacyOCRError):
    """The pipeline could not produce a result at all."""

    code = "extraction_failed"
    http_status = 422


class ServiceOverloadedError(PharmacyOCRError):
    """Work was rejected because the queue or concurrency limit is full."""

    code = "service_overloaded"
    http_status = 503


class JobNotFoundError(PharmacyOCRError):
    """Unknown or expired job id."""

    code = "job_not_found"
    http_status = 404


class BatchNotFoundError(PharmacyOCRError):
    """Unknown or expired batch id."""

    code = "batch_not_found"
    http_status = 404


class BatchTooLargeError(PharmacyOCRError):
    """Folder upload exceeded the per-request file count or byte ceiling."""

    code = "batch_too_large"
    http_status = 413


class EmptyBatchError(PharmacyOCRError):
    """A folder upload contained no file this service can read."""

    code = "empty_batch"
    http_status = 400


class RateLimitedError(PharmacyOCRError):
    """Client exceeded its request budget."""

    code = "rate_limited"
    http_status = 429


class UnauthorizedError(PharmacyOCRError):
    """Missing or invalid API key."""

    code = "unauthorized"
    http_status = 401
