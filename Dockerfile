# syntax=docker/dockerfile:1
#
# Multi-stage build. The builder produces a virtualenv; the runtime image gets
# only that venv, so build tooling never ships to production.

# --------------------------------------------------------------------------- #
# Base: shared OS layer
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1/libglib2.0-0: OpenCV. libgomp1: PaddlePaddle's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
FROM base AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependencies first: this layer is cached until requirements.txt changes,
# which matters because PaddlePaddle alone is a few hundred megabytes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    DATA_DIR=/data \
    HOME=/home/appuser \
    # Keep downloaded OCR weights on the mounted volume so a container restart
    # does not re-download several hundred megabytes.
    PADDLE_PDX_CACHE_HOME=/data/models \
    ENVIRONMENT=production \
    LOG_FORMAT=json \
    HOST=0.0.0.0 \
    PORT=8000

RUN useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /data /app \
    && chown -R appuser:appuser /data /app

WORKDIR /app
USER appuser

VOLUME ["/data"]
EXPOSE 8000

# start-period is generous: the first boot downloads and loads OCR weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health/live" || exit 1

# One worker on purpose: each worker loads its own copy of the OCR models.
# Scale with MAX_CONCURRENT_EXTRACTIONS inside the container, and with replicas
# outside it.
CMD ["sh", "-c", "exec uvicorn pharmacy_ocr.asgi:app --host \"$HOST\" --port \"$PORT\" --workers 1 --no-access-log"]
