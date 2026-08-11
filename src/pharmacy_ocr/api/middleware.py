"""ASGI middleware: request identity, access logging, metrics, security headers.

The request id is generated (or taken from an inbound `X-Request-ID`) and bound
to a contextvar, so every log line produced while handling the request — from
the upload writer through the OCR worker thread — carries the same id.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from pharmacy_ocr.observability import metrics
from pharmacy_ocr.observability.logging import bind_request_id

logger = logging.getLogger("pharmacy_ocr.access")

REQUEST_ID_HEADER = "X-Request-ID"

#: Sent on every response. The API returns JSON only, so a restrictive CSP and
#: nosniff cost nothing and stop a JSON body from ever being treated as markup.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs the access line, and records metrics."""

    def __init__(self, app: ASGIApp, *, include_docs_headers: bool = False) -> None:
        super().__init__(app)
        self._include_docs_headers = include_docs_headers

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        route = _route_label(request)
        start = time.perf_counter()

        with bind_request_id(request_id):
            request.state.request_id = request_id
            try:
                response = await call_next(request)
            except Exception:
                # The exception handlers turn this into a 500 response; record
                # the attempt here so failures are visible in the metrics.
                elapsed = time.perf_counter() - start
                metrics.HTTP_REQUESTS.labels(request.method, route, "500").inc()
                metrics.HTTP_LATENCY.labels(request.method, route).observe(elapsed)
                logger.exception(
                    "request failed",
                    extra={"method": request.method, "path": route, "seconds": round(elapsed, 3)},
                )
                raise

            elapsed = time.perf_counter() - start
            metrics.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
            metrics.HTTP_LATENCY.labels(request.method, route).observe(elapsed)

            response.headers[REQUEST_ID_HEADER] = request_id
            for header, value in _SECURITY_HEADERS.items():
                if header == "Cache-Control" and self._include_docs_headers:
                    continue
                response.headers.setdefault(header, value)

            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": route,
                    "status": response.status_code,
                    "seconds": round(elapsed, 3),
                    "client": _client_host(request),
                },
            )
            return response


def _route_label(request: Request) -> str:
    """Use the route template, not the raw path.

    Labelling metrics with `/v1/jobs/{job_id}` keeps cardinality bounded; the
    raw path would create one time series per job id.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None
