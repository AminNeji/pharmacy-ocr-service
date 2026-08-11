"""Exception handlers.

Domain errors carry their own code and HTTP status, so mapping happens once
here instead of being restated at each call site. Unexpected exceptions are
logged with a stack trace but reported to the client as a generic 500 — an
internal traceback is not something an invoice-upload client should receive.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from pharmacy_ocr.api.schemas import ErrorDetail, ErrorResponse
from pharmacy_ocr.domain.errors import PharmacyOCRError
from pharmacy_ocr.observability.logging import current_request_id

logger = logging.getLogger(__name__)


def _response(status_code: int, code: str, message: str, detail: object | None = None) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, detail=detail),
        request_id=current_request_id(),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PharmacyOCRError)
    async def _domain_error(_: Request, exc: PharmacyOCRError) -> JSONResponse:
        # `message` is a reserved LogRecord attribute: passing it via `extra=`
        # makes logging raise, which would turn every handled error into an
        # unhandled 500. It belongs in the log message itself.
        if exc.http_status >= 500:
            logger.error("domain error: %s", exc.message, extra={"code": exc.code})
        else:
            logger.info("client error: %s", exc.message, extra={"code": exc.code})
        return _response(exc.http_status, exc.code, exc.message, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # 422 literal: Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY and the
        # old constant now warns on access.
        return _response(422, "validation_error", "Request validation failed.", exc.errors())

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _response(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception", extra={"error_type": type(exc).__name__})
        return _response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An internal error occurred. Quote the request id when reporting this.",
        )
