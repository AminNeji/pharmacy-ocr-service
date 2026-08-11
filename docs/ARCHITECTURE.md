# How it works, and why it is built this way

For running the service, see the [README](../README.md). This document is for
whoever has to change the code.

---

## The extraction pipeline

**1. Preprocessing** (OpenCV). Clean scans get CLAHE contrast equalization and a
resize toward an A4-at-300-DPI equivalent. Phone photos (`photo_mode=true`)
additionally get perspective correction, denoising, sharpening, and optional
adaptive thresholding. PDF pages are rendered, processed, and released one at a
time so a five-page document never holds five full-resolution frames in memory.

**2. Page orientation.** Text boxes that are taller than they are wide, in bulk,
mean the page is sideways. When that fraction crosses a threshold the page is
re-OCR'd upright. If a page still parses nothing, a bounded recovery sweep tries
180° first — upside-down is by far the common case — capped by
`MAX_ORIENTATION_PROBES` so one bad page cannot cost five minutes.

**3. OCR** (PaddleOCR PP-OCRv5). Returns text boxes with coordinates and
confidence scores. The geometry is the point — it is what lets the parser
reconstruct rows deterministically instead of asking a model to guess which
number belongs to which medicine.

**4. Line reconstruction.** Skew is estimated from the detection polygons
themselves — no extra OCR pass — and rows are clustered along the tilted
baseline rather than along absolute `y`. On a photographed invoice with a 3°
tilt, clustering on raw `y` splits one row into two.

**5. Column mapping.** Heading tokens are matched to semantic roles
(`code`, `name`, `quantity`, `unit_price`, `line_total`, …) through a synonym
table plus fuzzy matching, so `Réf`, `Code PCT`, `CIP`, and `Barcode` all resolve
to the same role. Column x-bands are then derived from the heading positions,
with open-ended outer bands so a value that overflows its column is still
attributed correctly.

**6. Row parsing.** For each line: pull the leading product code (normalizing
`O`→`0`, `I`→`1`, and similar recognizer confusions), find every money value,
decide which one is the line total, infer the quantity — explicitly (`2 x`) or by
arithmetic (`total / unit`) — and take the medicine name from the text before the
first price. Headers, addresses, phone numbers, totals, and stamps are rejected
by a marker vocabulary.

**7. Checksum.** The printed invoice total is compared against the sum of
extracted line totals. This is the only independent evidence the service has
that an extraction is right, and it is what `status: ok` means.

**8. LLM fallback** (`hybrid` only). If the parser found no rows, or its checksum
failed, the OCR rows are sent to a local Ollama model. Its output is accepted
only if it makes the checksum pass when the parser's did not. Ties go to the
parser, because the parser is reproducible and the model is not.

**9. OCR retry** (PDFs only). If the first pass looks suspicious, the PDF is
re-rendered at a higher DPI and re-parsed. The better-scoring attempt wins, where
"better" weights checksum agreement far above row count.

---

## Layout

```
src/pharmacy_ocr/
├── settings.py           validated, immutable configuration
├── pipeline.py           orchestration; never raises
├── evaluation.py         scoring against hand-labelled ground truth
├── asgi.py               ASGI entry point
├── domain/               models and the error hierarchy
├── ocr/                  engine protocol, PaddleOCR impl, content-addressed cache
├── imaging/              upload validation, preprocessing, orientation
├── parsing/              text primitives, numbers, columns, row rules, LLM fallback
├── jobs/                 in-process job queue and batches
├── api/                  routers, dependencies, middleware, error handlers
├── observability/        structured logging, Prometheus metrics
└── cli/                  the pharmacy-ocr command
```

Two decisions shape everything else:

**Dependencies are injected, not imported.** `InvoicePipeline` receives its
engine, cache, preprocessor, parser, and mapper. Nothing reads `os.environ` at
import time, and no module holds mutable global state. This is what makes the
system testable and what lets two differently-configured pipelines coexist.

**The parser owns the truth; the model is a fallback that must prove itself.**
Invoice arithmetic is deterministic. Treating it as a language-modelling problem
produces answers that look right and are wrong — which, on a medical invoice, is
worse than an obvious failure.

---

## Development

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
pre-commit install

make check          # ruff + mypy + pytest
make cov            # coverage report in htmlcov/
```

The suite runs against a fake OCR engine, so it needs no model weights and
finishes in a few seconds. That is deliberate: the `OCREngine` protocol exists so
the parser, cache, pipeline, job queue, and HTTP layer are all testable without
loading half a gigabyte of weights.

```
tests/
├── conftest.py       fake engine and preprocessor, shared fixtures
├── test_parsing.py   prices, quantities, row classification, totals, clustering
├── test_columns.py   heading synonyms, column bands, role assignment
├── test_imaging.py   preprocessing, orientation evidence
├── test_domain.py    settings validation, model normalization
├── test_intake.py    extensions, size ceilings, magic bytes, bomb guards
├── test_cache.py     cache identity, atomicity, corruption recovery, eviction
├── test_pipeline.py  end-to-end behaviour, failure handling, LLM policy
├── test_api.py       HTTP contracts, auth, limits, jobs, batches
└── test_cli.py       argument wiring, batch behaviour, audit
```

### Changing the parser

Iterate against a warm OCR cache, or every edit costs a full OCR pass:

```bash
pharmacy-ocr cache warm --limit 20 --retry-profiles
pharmacy-ocr batch --limit 20 --output data/outputs_test --force
pharmacy-ocr audit --output data/outputs_test
```

The cache is keyed on file content plus OCR settings, so changing an OCR setting
invalidates it correctly while renaming or re-copying a file does not.

The header, skip, and total marker vocabularies live in
[`parsing/text.py`](../src/pharmacy_ocr/parsing/text.py) and the column synonyms
in [`parsing/columns.py`](../src/pharmacy_ocr/parsing/columns.py), both kept
separate from the logic so a new supplier layout is a data change. Add a test
alongside any change.

Then confirm you did not regress accuracy:

```bash
pharmacy-ocr evaluate
```

---

## Observability

**Logs** are JSON by default, one object per line, with a `request_id` that
follows one invoice across the API thread, the preprocessing worker, and the OCR
thread. Use `LOG_FORMAT=text` locally.

**Metrics** at `/metrics`:

| Metric | Why you care |
|---|---|
| `pharmacy_ocr_invoices_processed_total{status}` | The `needs_review` rate is your quality signal |
| `pharmacy_ocr_pipeline_duration_seconds` | End-to-end latency per invoice |
| `pharmacy_ocr_stage_duration_seconds{stage}` | Where the time actually goes |
| `pharmacy_ocr_ocr_cache_events_total{event}` | Cache hit rate |
| `pharmacy_ocr_llm_calls_total{outcome}` | How often the LLM is used and accepted |
| `pharmacy_ocr_extractions_in_flight` | Whether you are at the concurrency ceiling |
| `pharmacy_ocr_jobs_queued` | Queue depth — sustained growth means under-provisioned |

The alert worth having: a rising ratio of `needs_review` to `ok`. That usually
means a supplier changed their invoice layout.

---

## Known ceilings

The job queue is in-memory and per-process. Jobs are lost on restart, and two
replicas share neither a queue nor a rate-limit budget. That is the correct size
for a single pharmacy or clinic. Durability or horizontal scale means replacing
`JobManager` with a real broker (Redis/RQ, Celery) and moving the rate limiter to
shared storage — the interfaces are small and isolated for exactly this reason.
