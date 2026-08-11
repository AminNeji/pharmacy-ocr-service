"""FastAPI dependencies: configuration access, authentication, rate limiting."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from pharmacy_ocr.domain.errors import ConfigurationError, RateLimitedError, UnauthorizedError
from pharmacy_ocr.jobs.manager import JobManager
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import Settings

API_KEY_HEADER = "X-API-Key"

#: auto_error=False so a missing key reaches our handler and produces the
#: standard error envelope instead of FastAPI's default shape.
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_pipeline(request: Request) -> InvoicePipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - only reachable if lifespan is skipped
        raise ConfigurationError("Pipeline is not initialized.")
    return pipeline


def get_job_manager(request: Request) -> JobManager:
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:  # pragma: no cover
        raise ConfigurationError("Job manager is not initialized.")
    return manager


class RateLimiter:
    """Sliding-window request counter, keyed by API key or client address.

    In-memory and therefore per-process: with several replicas each enforces
    its own budget. That is a deliberate trade — a shared Redis counter is the
    upgrade path, documented in the README, and not worth the dependency for a
    single-container deployment.
    """

    #: Sweep idle identities out of the map every N checks. Without this the
    #: map grows by one entry per distinct client address, forever.
    _PRUNE_EVERY = 1000

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self._limit = limit_per_minute
        self._window = window_s
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._checks = 0
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    @property
    def tracked_identities(self) -> int:
        with self._lock:
            return len(self._hits)

    def check(self, identity: str) -> None:
        """Record a request for `identity`, raising once the budget is spent."""
        if not self.enabled:
            return
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            self._checks += 1
            if self._checks % self._PRUNE_EVERY == 0:
                self._prune_locked(cutoff)

            hits = self._hits[identity]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = max(1, int(hits[0] + self._window - now) + 1)
                raise RateLimitedError(
                    "Too many requests.",
                    detail={"limit_per_minute": self._limit, "retry_after_s": retry_after},
                )
            hits.append(now)

    def prune(self) -> int:
        """Drop identities with no recent activity so the map stays bounded."""
        with self._lock:
            return self._prune_locked(time.monotonic() - self._window)

    def _prune_locked(self, cutoff: float) -> int:
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for key in stale:
            self._hits.pop(key, None)
        return len(stale)


def _client_identity(request: Request, api_key: str | None, settings: Settings) -> str:
    if api_key:
        # Never key the limiter on the raw secret.
        return f"key:{hashlib.blake2b(api_key.encode(), digest_size=8).hexdigest()}"
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Depends(api_key_scheme)] = None,
) -> str | None:
    """Authenticate the caller and charge them a rate-limit token.

    Keys come from `API_KEYS` in the environment. Comparison is constant-time
    against every configured key so response latency does not leak how much of
    a guessed key was correct.
    """
    settings: Settings = request.app.state.settings

    if settings.auth_enabled:
        supplied = api_key or ""
        # Compare against all keys unconditionally: short-circuiting on the
        # first match would make timing depend on key order.
        matched = False
        for candidate in settings.api_keys:
            if secrets.compare_digest(supplied, candidate):
                matched = True
        if not matched:
            raise UnauthorizedError("A valid X-API-Key header is required.")

    limiter: RateLimiter = request.app.state.rate_limiter
    limiter.check(_client_identity(request, api_key, settings))
    return api_key


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[InvoicePipeline, Depends(get_pipeline)]
JobManagerDep = Annotated[JobManager, Depends(get_job_manager)]
AuthDep = Annotated[str | None, Depends(require_api_key)]
