"""HTTP layer: contracts, auth, limits, error envelope, job flow."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pharmacy_ocr.api.app import create_app
from pharmacy_ocr.api.dependencies import RateLimiter
from pharmacy_ocr.domain.errors import RateLimitedError
from pharmacy_ocr.pipeline import InvoicePipeline
from pharmacy_ocr.settings import Settings
from tests.conftest import PDF_MAGIC, PNG_MAGIC, FakeOCREngine, FakePreprocessor

VALID_PNG = PNG_MAGIC + b"payload" * 16


@contextmanager
def client_for(settings: Settings, engine: FakeOCREngine | None = None) -> Iterator[TestClient]:
    pipeline = InvoicePipeline(settings, engine=engine or FakeOCREngine(), preprocessor=FakePreprocessor())
    with TestClient(create_app(settings, pipeline=pipeline)) as client:
        yield client


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with client_for(settings) as test_client:
        yield test_client


def upload(content: bytes = VALID_PNG, name: str = "invoice.png", mime: str = "image/png") -> dict:
    return {"file": (name, content, mime)}


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_liveness(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_individual_checks(client: TestClient) -> None:
    body = client.get("/health/ready").json()
    assert body["status"] == "ok"
    assert body["checks"]["temp_dir_writable"] is True
    assert body["checks"]["job_manager"] is True


def test_health_exposes_config_without_secrets(settings: Settings) -> None:
    secured = settings.model_copy(update={"api_keys_raw": "topsecret"})
    with client_for(secured) as client:
        body = client.get("/health", headers={"X-API-Key": "topsecret"}).json()
    assert body["config"]["auth_enabled"] is True
    assert "topsecret" not in str(body)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_returns_items(client: TestClient) -> None:
    response = client.post("/v1/extract", files=upload())
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["source_file"] == "invoice.png"
    assert len(body["items"]) == 3
    assert body["checksum_ok"] is True
    assert body["items"][0]["REFCODE"] == "303616"


def test_legacy_unversioned_path_still_works(client: TestClient) -> None:
    assert client.post("/extract", files=upload()).status_code == 200


def test_photo_mode_query_is_accepted(client: TestClient) -> None:
    assert client.post("/v1/extract?photo_mode=true", files=upload()).status_code == 200


def test_mode_override_is_validated(client: TestClient) -> None:
    response = client.post("/v1/extract?mode=nonsense", files=upload())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/extract", files=upload(b"MZ\x90", "payload.exe", "application/exe"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_content_not_matching_extension_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/extract", files=upload(PDF_MAGIC + b"x", "disguised.png"))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_document"


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/extract", files=upload(b"", "invoice.png"))
    assert response.status_code == 400


def test_oversized_upload_is_rejected(settings: Settings) -> None:
    tiny = settings.model_copy(update={"max_upload_bytes": 1024})
    with client_for(tiny) as client:
        response = client.post("/v1/extract", files=upload(PNG_MAGIC + b"x" * 4096))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_missing_file_field_returns_the_error_envelope(client: TestClient) -> None:
    response = client.post("/v1/extract")
    assert response.status_code == 422
    assert "error" in response.json()


# --------------------------------------------------------------------------- #
# Authentication and rate limiting
# --------------------------------------------------------------------------- #
def test_missing_api_key_is_unauthorized(settings: Settings) -> None:
    secured = settings.model_copy(update={"api_keys_raw": "k1"})
    with client_for(secured) as client:
        response = client.post("/v1/extract", files=upload())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_wrong_api_key_is_unauthorized(settings: Settings) -> None:
    secured = settings.model_copy(update={"api_keys_raw": "k1"})
    with client_for(secured) as client:
        response = client.post("/v1/extract", files=upload(), headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_valid_api_key_is_accepted(settings: Settings) -> None:
    secured = settings.model_copy(update={"api_keys_raw": "k1,k2"})
    with client_for(secured) as client:
        response = client.post("/v1/extract", files=upload(), headers={"X-API-Key": "k2"})
    assert response.status_code == 200


def test_rate_limit_returns_429_with_retry_hint(settings: Settings) -> None:
    limited = settings.model_copy(update={"rate_limit_per_minute": 2})
    with client_for(limited) as client:
        assert client.get("/health/live").status_code == 200  # unauthenticated, not limited
        assert client.post("/v1/extract", files=upload()).status_code == 200
        assert client.post("/v1/extract", files=upload()).status_code == 200
        response = client.post("/v1/extract", files=upload())

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["detail"]["limit_per_minute"] == 2


def test_rate_limiter_isolates_identities() -> None:
    limiter = RateLimiter(limit_per_minute=1)
    limiter.check("key:a")
    limiter.check("key:b")  # a different caller has its own budget
    with pytest.raises(RateLimitedError):
        limiter.check("key:a")


def test_rate_limiter_disabled_at_zero() -> None:
    limiter = RateLimiter(limit_per_minute=0)
    for _ in range(1000):
        limiter.check("key:a")
    assert limiter.enabled is False


def test_rate_limiter_forgets_idle_identities() -> None:
    """Without pruning the identity map grows by one entry per client forever."""
    limiter = RateLimiter(limit_per_minute=100, window_s=0.01)
    for index in range(50):
        limiter.check(f"ip:10.0.0.{index}")
    assert limiter.tracked_identities == 50

    time.sleep(0.02)
    assert limiter.prune() == 50
    assert limiter.tracked_identities == 0


def test_rate_limiter_prunes_itself_during_normal_use() -> None:
    limiter = RateLimiter(limit_per_minute=10_000, window_s=0.01)
    for index in range(RateLimiter._PRUNE_EVERY - 1):
        limiter.check(f"ip:{index}")
    tracked_before = limiter.tracked_identities

    time.sleep(0.02)
    limiter.check("ip:trigger")  # the Nth check runs a sweep

    assert tracked_before == RateLimiter._PRUNE_EVERY - 1
    assert limiter.tracked_identities == 1


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _await_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_job_submission_returns_202_and_a_location(client: TestClient) -> None:
    response = client.post("/v1/jobs", files=upload())
    assert response.status_code == 202

    body = response.json()
    assert body["status"] in ("queued", "running", "succeeded")
    assert body["filename"] == "invoice.png"
    assert f"/v1/jobs/{body['id']}" in response.headers["location"]


def test_job_completes_and_carries_the_result(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", files=upload()).json()["id"]
    body = _await_job(client, job_id)

    assert body["status"] == "succeeded"
    assert body["result"]["checksum_ok"] is True
    assert len(body["result"]["items"]) == 3
    assert body["result"]["source_file"] == "invoice.png"
    assert body["duration_s"] is not None


def test_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_job_stats(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", files=upload()).json()["id"]
    _await_job(client, job_id)
    counts = client.get("/v1/jobs").json()["counts"]
    assert counts["total"] >= 1


def test_job_rejects_bad_uploads_before_queueing(client: TestClient) -> None:
    assert client.post("/v1/jobs", files=upload(b"MZ", "x.exe")).status_code == 415


# --------------------------------------------------------------------------- #
# Batches (folder upload)
# --------------------------------------------------------------------------- #
def folder(*names: str) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, VALID_PNG, "image/png")) for name in names]


def _await_batch(client: TestClient, batch_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/batches/{batch_id}").json()
        if body["status"] == "completed":
            return body
        time.sleep(0.05)
    raise AssertionError(f"batch {batch_id} did not finish within {timeout}s")


def test_batch_queues_every_file(client: TestClient) -> None:
    response = client.post("/v1/batches", files=folder("a.png", "b.png", "c.png"))
    assert response.status_code == 202

    body = response.json()
    assert body["accepted"] == 3
    assert body["rejected"] == []
    assert [job["filename"] for job in body["jobs"]] == ["a.png", "b.png", "c.png"]
    assert f"/v1/batches/{body['id']}" in response.headers["location"]


def test_batch_results_carry_every_extraction(client: TestClient) -> None:
    batch_id = client.post("/v1/batches", files=folder("a.png", "b.png")).json()["id"]
    _await_batch(client, batch_id)

    body = client.get(f"/v1/batches/{batch_id}/results").json()
    assert body["status"] == "completed"
    assert body["counts"]["succeeded"] == 2
    assert all(len(job["result"]["items"]) == 3 for job in body["results"])


def test_batch_skips_unreadable_files_instead_of_failing(client: TestClient) -> None:
    files = [*folder("real.png"), ("files", ("notes.txt", b"hello", "text/plain"))]
    body = client.post("/v1/batches", files=files).json()

    assert body["accepted"] == 1
    assert body["rejected"] == [
        {
            "filename": "notes.txt",
            "code": "unsupported_file_type",
            "message": "Unsupported file type '.txt'.",
        }
    ]


def test_batch_of_only_unreadable_files_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/batches", files=[("files", ("x.txt", b"hi", "text/plain"))])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_batch"


def test_batch_over_the_file_limit_is_rejected(settings: Settings) -> None:
    with client_for(settings.model_copy(update={"max_batch_files": 2})) as client:
        response = client.post("/v1/batches", files=folder("a.png", "b.png", "c.png"))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "batch_too_large"


def test_batch_too_large_for_the_queue_is_rejected_whole(settings: Settings) -> None:
    """Partial acceptance would leave the caller reconciling a half-queued folder."""
    tiny = settings.model_copy(update={"job_queue_max_size": 2, "max_concurrent_extractions": 1})
    with client_for(tiny, engine=_SlowEngine()) as client:
        response = client.post("/v1/batches", files=folder("a.png", "b.png", "c.png", "d.png"))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_overloaded"
        assert client.get("/v1/jobs").json()["counts"]["total"] == 0


def test_unknown_batch_is_404(client: TestClient) -> None:
    response = client.get("/v1/batches/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "batch_not_found"


def test_batch_requires_authentication(settings: Settings) -> None:
    with client_for(settings.model_copy(update={"api_keys_raw": "k1"})) as client:
        assert client.post("/v1/batches", files=folder("a.png")).status_code == 401


class _SlowEngine(FakeOCREngine):
    """Keeps jobs occupied so queue-capacity behaviour is observable."""

    def run_page(self, image_path, page_index: int = 0):  # type: ignore[no-untyped-def]
        time.sleep(0.5)
        return super().run_page(image_path, page_index)


# --------------------------------------------------------------------------- #
# Cross-cutting concerns
# --------------------------------------------------------------------------- #
def test_server_side_domain_errors_still_return_the_envelope(client: TestClient) -> None:
    """A 5xx domain error is logged at ERROR; a bad `extra=` key there used to
    raise inside the handler and replace the envelope with a bare 500."""
    from pharmacy_ocr.domain.errors import OCREngineError

    @client.app.get("/_boom")  # type: ignore[attr-defined]
    async def _boom() -> None:
        raise OCREngineError("engine exploded")

    response = client.get("/_boom")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ocr_failed"


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "trace-abc"})
    assert response.headers["X-Request-ID"] == "trace-abc"


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    assert client.get("/health/live").headers.get("X-Request-ID")


def test_security_headers_are_present(client: TestClient) -> None:
    headers = client.get("/health/live").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_error_responses_include_the_request_id(client: TestClient) -> None:
    body = client.post("/v1/extract", files=upload(b"MZ", "x.exe")).json()
    assert body["request_id"]


def test_metrics_endpoint_exposes_prometheus_text(client: TestClient) -> None:
    client.post("/v1/extract", files=upload())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "pharmacy_ocr_http_requests_total" in response.text
    assert "pharmacy_ocr_invoices_processed_total" in response.text


def test_docs_are_disabled_by_default(settings: Settings) -> None:
    hidden = settings.model_copy(update={"docs_enabled": False})
    with client_for(hidden) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_docs_can_be_enabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/v1/extract" in schema["paths"]
    assert "/v1/jobs" in schema["paths"]
