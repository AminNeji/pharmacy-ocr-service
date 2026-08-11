"""Prometheus metrics.

Deliberately small: request/latency counters for the API, plus the pipeline
signals that actually predict a bad night — OCR duration, cache hit rate, and
how many invoices come out needing human review.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REGISTRY = CollectorRegistry(auto_describe=True)

# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
HTTP_REQUESTS = Counter(
    "pharmacy_ocr_http_requests_total",
    "HTTP requests handled.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "pharmacy_ocr_http_request_duration_seconds",
    "HTTP request latency.",
    labelnames=("method", "path"),
    buckets=(0.05, 0.25, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
INVOICES_PROCESSED = Counter(
    "pharmacy_ocr_invoices_processed_total",
    "Invoices that completed the pipeline, by review status.",
    labelnames=("status",),
    registry=REGISTRY,
)
PIPELINE_LATENCY = Histogram(
    "pharmacy_ocr_pipeline_duration_seconds",
    "End-to-end pipeline duration per invoice.",
    buckets=(1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0),
    registry=REGISTRY,
)
STAGE_LATENCY = Histogram(
    "pharmacy_ocr_stage_duration_seconds",
    "Duration of individual pipeline stages.",
    labelnames=("stage",),
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 180.0),
    registry=REGISTRY,
)
ITEMS_EXTRACTED = Histogram(
    "pharmacy_ocr_items_per_invoice",
    "Validated line items extracted per invoice.",
    buckets=(0, 1, 2, 5, 10, 20, 40, 80),
    registry=REGISTRY,
)
CACHE_EVENTS = Counter(
    "pharmacy_ocr_ocr_cache_events_total",
    "OCR cache outcomes.",
    labelnames=("event",),  # hit | miss | write | evict | error
    registry=REGISTRY,
)
LLM_CALLS = Counter(
    "pharmacy_ocr_llm_calls_total",
    "LLM fallback invocations.",
    labelnames=("outcome",),  # accepted | rejected | failed
    registry=REGISTRY,
)
EXTRACTIONS_IN_FLIGHT = Gauge(
    "pharmacy_ocr_extractions_in_flight",
    "Invoices currently being processed.",
    registry=REGISTRY,
)
JOBS_QUEUED = Gauge(
    "pharmacy_ocr_jobs_queued",
    "Jobs waiting for a worker.",
    registry=REGISTRY,
)


def render() -> bytes:
    """Serialize the registry in Prometheus text exposition format."""
    return generate_latest(REGISTRY)
