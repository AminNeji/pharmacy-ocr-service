"""Application factory.

`create_app` takes explicit settings so tests can build an app with a temp data
directory and a fake OCR engine without touching the environment.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from pharmacy_ocr import __version__
from pharmacy_ocr.api.dependencies import API_KEY_HEADER, RateLimiter
from pharmacy_ocr.api.errors import register_exception_handlers
from pharmacy_ocr.api.middleware import RequestContextMiddleware
from pharmacy_ocr.api.routers import batches, extract, health, jobs
from pharmacy_ocr.jobs.manager import JobManager
from pharmacy_ocr.observability.logging import configure_logging
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DESCRIPTION = f"""\
Extracts medicine line items from Tunisian pharmacy invoices.

**Pipeline:** OpenCV preprocessing -> PaddleOCR text boxes -> deterministic row
parser -> invoice-total checksum. A local LLM is consulted only in `hybrid`
mode, and only when the parser finds nothing or its arithmetic disagrees.

**Choosing an endpoint**
* `POST /v1/extract` - synchronous, for a single interactive upload.
* `POST /v1/jobs` - one invoice, asynchronous. Returns a job id immediately.
  Use this behind proxies with short idle timeouts.
* `POST /v1/batches` - a whole folder in one request. Returns one batch id to
  poll; unreadable files are reported rather than failing the upload.

**Authentication:** send the `{API_KEY_HEADER}` header when the deployment
configures API keys.

**Reading the result:** `status` is the field to route on. `needs_review` means
the extraction is plausible but unverified — the checksum failed, no total was
printed, or rows were dropped. Never book a `needs_review` invoice unseen.
"""


def create_app(settings: Settings | None = None, *, pipeline: InvoicePipeline | None = None) -> FastAPI:
    """Build the ASGI application."""
    resolved = settings or get_settings()
    configure_logging(resolved)
    resolved.ensure_directories()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.extraction_semaphore = asyncio.Semaphore(resolved.max_concurrent_extractions)
        app.state.pipeline = pipeline or InvoicePipeline(resolved)
        app.state.job_manager = JobManager(app.state.pipeline, resolved)
        await app.state.job_manager.start()

        # Model loading takes tens of seconds. Do it off the event loop and let
        # readiness reflect progress, so an orchestrator holds traffic back
        # instead of failing the first real request.
        app.state.ocr_ready = not resolved.warmup_on_startup
        app.state.warmup_task = None
        if resolved.warmup_on_startup:
            app.state.warmup_task = asyncio.create_task(_warmup(app), name="ocr-warmup")

        logger.info(
            "service started",
            extra={"version": __version__, "environment": resolved.environment.value},
        )
        try:
            yield
        finally:
            task = getattr(app.state, "warmup_task", None)
            if task is not None and not task.done():
                task.cancel()
            await app.state.job_manager.stop()
            await app.state.pipeline.aclose()
            logger.info("service stopped")

    app = FastAPI(
        title="Pharmacy Invoice OCR",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )

    app.state.settings = resolved
    app.state.rate_limiter = RateLimiter(resolved.rate_limit_per_minute)
    app.state.ocr_ready = False

    register_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware, include_docs_headers=resolved.docs_enabled)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=[API_KEY_HEADER, "Content-Type"],
        )

    app.include_router(health.router)
    app.include_router(extract.router)
    app.include_router(jobs.router)
    app.include_router(batches.router)
    return app


async def _warmup(app: FastAPI) -> None:
    try:
        await asyncio.to_thread(app.state.pipeline.warmup)
        app.state.ocr_ready = True
        logger.info("OCR warmup complete")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("OCR warmup failed; the engine will load on first use")
        app.state.ocr_ready = True
