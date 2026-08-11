"""Liveness, readiness, and metrics.

Liveness and readiness are separate on purpose. During model warmup the process
is alive but cannot serve an extraction; an orchestrator must not restart it
(liveness ok) and must not route traffic to it yet (readiness not ok).
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from pharmacy_ocr import __version__
from pharmacy_ocr.api.dependencies import AuthDep, SettingsDep
from pharmacy_ocr.api.schemas import HealthResponse
from pharmacy_ocr.observability import metrics
from pharmacy_ocr.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def live() -> HealthResponse:
    """Returns 200 as long as the process can serve requests."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready(request: Request, settings: SettingsDep, response: Response) -> HealthResponse:
    checks = {
        "ocr_engine_loaded": bool(getattr(request.app.state, "ocr_ready", False)),
        "job_manager": getattr(request.app.state, "job_manager", None) is not None,
        "temp_dir_writable": _is_writable(settings),
    }
    healthy = all(checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "not_ready", version=__version__, checks=checks)


@router.get("/health", response_model=HealthResponse, summary="Health and configuration snapshot")
async def health(request: Request, settings: SettingsDep) -> HealthResponse:
    manager = getattr(request.app.state, "job_manager", None)
    checks = {
        "ocr_engine_loaded": bool(getattr(request.app.state, "ocr_ready", False)),
        "jobs": manager.stats() if manager else None,
    }
    return HealthResponse(status="ok", version=__version__, checks=checks, config=settings.public_snapshot())


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def prometheus_metrics(_: AuthDep) -> Response:
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


def _is_writable(settings: Settings) -> bool:
    try:
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.temp_dir / ".probe"
        probe.write_bytes(b"")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True
