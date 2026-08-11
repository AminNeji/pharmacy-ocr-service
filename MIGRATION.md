# Migrating from `pharmacy_ocr` (v2) to `pharmacy-ocr-service` (v3)

The old project still works and has not been modified. This document covers what
changes for you when you switch.

## 1. The output JSON is backward compatible

`REFCODE`, `MED`, `QT`, and `prix` are unchanged in both name and meaning, and
so are `source_file`, `items`, `warnings`, `error`, `invoice_total`,
`items_total`, `checksum_ok`, and `timings`. Existing consumers keep working.

New fields were added, all optional:

| Field | Where | Meaning |
|---|---|---|
| `status` | result | `ok`, `needs_review`, or `failed` — the one field to route on |
| `page_count` | result | Pages actually OCR'd |
| `extraction_mode` | result | Which mode produced this result |
| `ocr_engine` | result | Engine name |
| `service_version` | result | Version that produced the result |
| `processed_at` | result | UTC timestamp |
| `unit_price` | item | Unit price when the invoice prints one |
| `confidence` | item | Mean OCR confidence for that row |

**Recommended change:** route on `status` instead of re-deriving it. Previously
you had to check `error`, then `checksum_ok`, then count `warnings` yourself.

## 2. Configuration moved from `app/config.py` to environment variables

Every setting keeps its old environment-variable name, so an existing `.env` or
set of exports keeps working. What changed:

- Editing `config.py` is no longer how you configure the service. Use
  environment variables or a `.env` file; run `pharmacy-ocr config` to print the
  effective configuration.
- Paths derive from a single `DATA_DIR` (default `./data`) instead of the
  package directory. `INVOICES_DIR`, `OUTPUTS_DIR`, `LOG_DIR`, `OCR_TEMP_DIR`,
  and `OCR_CACHE_DIR` still override individual locations.
- Invalid values now fail at startup with a clear message rather than producing
  odd behaviour later.

**Breaking in production only:** with `ENVIRONMENT=production` the service
refuses to start unless `API_KEYS` is set. If authentication is handled by a
gateway in front of it, set `ALLOW_UNAUTHENTICATED=true` to say so explicitly.
For local work use `ENVIRONMENT=development`.

## 3. Commands

| Old | New |
|---|---|
| `python scripts/batch_process.py` | `pharmacy-ocr batch` |
| `python scripts/batch_process.py --limit 5 --mode deterministic` | `pharmacy-ocr batch --limit 5 --mode deterministic` |
| `python scripts/cache_ocr.py --retry-profile` | `pharmacy-ocr cache warm --retry-profiles` |
| `python scripts/audit_outputs.py --output outputs` | `pharmacy-ocr audit --output outputs` |
| `uvicorn app.main:app` | `pharmacy-ocr serve` (or `uvicorn pharmacy_ocr.asgi:app`) |
| *(none)* | `pharmacy-ocr cache prune`, `pharmacy-ocr config` |

`scripts/llm_cache_experiment.py` was not ported. It was a research tool, and
the same experiment is now a few lines against `LLMMapper` with a warmed cache.

## 4. HTTP API

- `POST /extract` still works, unversioned and undocumented, as a compatibility
  shim. New clients should use `POST /v1/extract`.
- `GET /health` still returns a configuration snapshot. `/health/live` and
  `/health/ready` are the ones to wire into an orchestrator.
- **Error responses changed shape.** Previously FastAPI's `{"detail": "..."}`.
  Now every non-2xx response is:

  ```json
  {
    "error": { "code": "unsupported_file_type", "message": "...", "detail": {...} },
    "request_id": "0f9c..."
  }
  ```

  If you parsed `detail` as a string, read `error.message` instead.
- New: `POST /v1/jobs` + `GET /v1/jobs/{id}` for asynchronous extraction. Prefer
  these for anything batch-shaped — a synchronous request that runs for a minute
  will be cut off by most proxies.

## 5. The OCR cache is rebuilt once

`OCR_CACHE_VERSION` moved from `v2` to `v3` and the key no longer includes the
file's mtime. Your first run after switching re-OCRs everything; after that the
hit rate is *better* than before, because copying or resyncing a file no longer
invalidates its entry.

The old `.ocr_cache/` directory can be deleted.

## 6. Two parser behaviours changed deliberately

Both were bugs, but they change output on some invoices, so verify against your
own corpus before switching:

1. **Repeated rows are kept.** Previously any two rows with identical text were
   collapsed anywhere in the document, which deleted genuine repeated medicine
   lines. Now only *adjacent* duplicates collapse. If an invoice really lists the
   same product twice, you get two items and the checksum confirms it. If the
   OCR stuttered, the checksum fails and the invoice is flagged `needs_review`
   rather than silently under-counted.

2. **The invoice total is chosen by marker specificity, not magnitude.**
   "Net a payer" and "Mnt Total Facture" outrank a bare "Total"; a pre-tax
   "Total HTVA" is used only when the invoice prints nothing else. This changes
   `invoice_total` — and therefore `checksum_ok` — on invoices that print
   several totals.

## 7. Suggested cutover

```bash
# 1. Install alongside the old project
cd pharmacy-ocr-service
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt && pip install -e . --no-deps

# 2. Point it at your existing invoices
export INVOICES_DIR=../pharmacy_ocr/invoices        # PowerShell: $env:INVOICES_DIR=...
export ENVIRONMENT=development

# 3. Re-extract into a fresh folder and compare against the old outputs
pharmacy-ocr batch --output ./data/outputs_v3 --mode deterministic
pharmacy-ocr audit --output ./data/outputs_v3
```

Compare the `ok` / `needs_review` / `failed` counts against your v2 outputs
before switching production traffic. Invoices that move from `ok` to
`needs_review` are worth reading individually — that is where the two behaviour
changes above will show up.
