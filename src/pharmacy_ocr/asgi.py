"""ASGI entry point for process managers.

    uvicorn pharmacy_ocr.asgi:app --host 0.0.0.0 --port 8000
    gunicorn pharmacy_ocr.asgi:app -k uvicorn.workers.UvicornWorker

Note on workers: each worker process loads its own copy of the OCR models
(hundreds of MB) and holds its own job queue. Prefer one worker with
`MAX_CONCURRENT_EXTRACTIONS` tuned to the box, and scale out with containers.
"""

from __future__ import annotations

from pharmacy_ocr.api.app import create_app

app = create_app()
