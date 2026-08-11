# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.0] - 2026-08-10

### Added

- **Folder upload.** `POST /v1/batches` accepts many invoices in one request and
  returns a single id. `GET /v1/batches/{id}` reports progress without the
  extraction payloads so polling a 50-invoice batch stays cheap;
  `GET /v1/batches/{id}/results` returns everything. Files the service cannot
  read are reported in `rejected` rather than failing the upload — a real folder
  contains a `Thumbs.db`, and rejecting the batch over it would force every
  client to reimplement the allow-list. Bounded by `MAX_BATCH_FILES` (50) and
  `MAX_BATCH_UPLOAD_BYTES` (200 MB); a batch too large for the queue is refused
  whole, never half-accepted.
- **`examples/upload_folder.py`** — a standalone client that chunks a folder,
  polls, and writes one JSON per invoice. Depends only on `httpx`, so it can be
  handed to a customer as-is.
- **`docs/ARCHITECTURE.md`** — pipeline internals, layout, and development
  workflow, moved out of the README so that document stays about running the
  service.

### Fixed

- **Every 5xx error returned a bare 500 instead of its error envelope.** The
  domain-error handler passed `message` through `extra=`, which is a reserved
  `LogRecord` attribute; `logging` raised `KeyError` inside the handler and the
  response fell through to the generic 500 path. Affected `service_overloaded`,
  `ocr_failed`, and `llm_mapping_failed` — the errors a client most needs to
  distinguish. Found by a test that first exercised a 503.

### Changed

- README rewritten around installing on a new machine and making HTTP requests.

## [3.2.0] - 2026-07-31

Driven by reading the invoice images and comparing them against the JSON, which
exposed three faults the invoice-total checksum is structurally blind to.

### Added

- **Ground-truth labels and a scoring harness.** `pharmacy-ocr evaluate` scores
  extraction against hand-transcribed invoices, reporting row recall/precision,
  per-field accuracy for REFCODE/MED/QT/prix, false positives and negatives, and
  per-invoice time. Rows are matched before fields are compared, so one dropped
  row no longer scores every later row wrong. This is the project's first real
  accuracy measurement; the checksum was only ever a proxy.
- **Page-orientation correction.** Box aspect ratio after the first OCR pass
  detects a sideways page, which is then re-OCR'd upright. Only pages that
  actually look rotated pay the extra pass.
- **Column-aware table reconstruction** (`parsing/columns.py`). The header row
  is located, each heading mapped to a semantic role, and its x-extent turned
  into a column band that later rows are cut against.
- **Semantic heading roles.** `Référence`/`Code PCT`/`CIP`/`Article` all resolve
  to `code`; `Produit`/`Libellé`/`Désignation`/`Médicament` to `name`;
  `PU`/`PUV`/`Prix Unit.` to `unit_price`, and so on. A new supplier layout
  usually needs a synonym rather than new logic.

### Fixed

- **Sideways invoices produced nothing.** Three corpus invoices came back with
  80%+ of their text boxes taller than wide; two extracted zero rows. Row recall
  on the labelled set went from 78.6% to 100%.
- **The dosage-form column was appended to the medicine name** —
  `ECOREX 1% CREME 30GR` became `ECOREX 1% CREME 30GR CREME`. MED accuracy went
  from 50% to 92.9%.
- **`10*10` was truncated to `10*`**: the trailing half of a 10x10 cm dimension
  was stripped as though it were a quantity.
- **Header grouping was too greedy.** Preferring the longest matching run of
  header tokens merged `Code PCT | Produit | Forme` into a single `code` column
  under the synonym "code pct", losing the product column entirely.

### Known gaps

- **The 10-15s SLA is not met on this CPU.** A single OCR pass costs ~25-32s;
  invoices needing an orientation probe or a DPI retry cost more. The machine
  has an RTX 3050 but PaddlePaddle is the CPU-only build. See docs/ACCURACY.md.
- The labelled set is five invoices. It is enough to catch systematic faults,
  not to claim a corpus-wide accuracy figure.

## [3.1.0] - 2026-07-29

Found by running the full 84-invoice corpus and inspecting every result. Each
parser change below was measured against the whole corpus, not just the invoice
that exposed it.

### Added

- **`pharmacy-ocr diagnose <invoice>`** — prints the parser's decision for every
  reconstructed line: what became an item, what was dropped by validation, and
  which rule rejected the rest. Runs off the OCR cache, so iterating on parser
  rules costs seconds. `--show-all` includes skipped lines.
- **`pharmacy-ocr audit` now groups warnings into categories**, so a systematic
  problem (one supplier's layout changed) stands out from one-off noise, and
  reports the signed difference on every failed checksum.

### Fixed

- **Patient copay was extracted as the line price.** On the CNAM layout
  (`Prix Pub | Tot. Fact | T.Ref | Tx | Perçu`) the parser took the last figure
  on the row, which is what the patient pays, not what the line cost — reading
  19.160 instead of 63.865. This was the single largest source of wrong prices
  in the corpus.
- **Package leaflets were parsed as invoice rows.** Where a product insert was
  photographed next to the invoice, prose containing numbers produced phantom
  items; on one invoice three of them added exactly the 3.200 by which the
  checksum failed. An invoice row ends at its rightmost column, so a line with
  a sentence's worth of text after its last figure is no longer a row.
- **Rows written `NAME 1 = 63.865` lost their quantity**, and the `1 =` stayed
  glued to the medicine name. `=` is now recognised as a quantity separator.
- **Invoice-total selection by marker specificity was a regression** introduced
  in 3.0.0 and is reverted. OCR routinely detaches a label like "Total Facture"
  from its own column and re-attaches it to the CNAM share, so the most
  authoritative-looking label can sit beside the wrong number. Magnitude cannot
  be scrambled that way, and the gross is never smaller than a share.
- **Every failed checksum re-OCR'd the whole document twice.** Any soft warning
  triggered the higher-DPI retry path, roughly tripling the cost of a typical
  PDF. A passing checksum is arithmetic proof the extraction is right, so those
  documents are no longer re-rendered.
- **`diagnose` did not apply the pipeline's item quality gate**, so it reported
  rows that never reach the output.

### Notes on what was *not* changed

Selecting the largest figure on a row as the line total was tried and measured:
it reads a single garbled number as the price, inflating one invoice's
extracted sum from 56.604 to 423.279. The narrower rules above were kept
instead.

Several remaining `needs_review` results are OCR-quality-bound rather than
parser bugs — low-resolution CamScanner captures where the medicine name or the
price column was never legible. The service flags these rather than guessing,
which is the intended behaviour.

## [3.0.0] - 2026-07-29

First release intended for production deployment. The extraction logic is the
same proven approach as v2; everything around it was rebuilt.

### Added

- **Asynchronous job API** — `POST /v1/jobs` returns `202` with a job id;
  `GET /v1/jobs/{id}` polls for the result. A synchronous request that runs for
  30-60 seconds fails against the idle timeouts most load balancers ship with.
- **API-key authentication** (`X-API-Key`) with constant-time comparison, plus
  a per-key sliding-window rate limiter.
- **Upload hardening** — streamed size ceiling, extension allow-list, magic-byte
  verification, and a megapixel cap that rejects decompression bombs.
- **`status` field** on every result (`ok` / `needs_review` / `failed`) so
  downstream systems have one field to route on instead of re-deriving intent
  from `checksum_ok`, `warnings`, and `error`.
- **Split health probes** — `/health/live` and `/health/ready`, so an
  orchestrator holds traffic back during model warmup instead of restarting a
  process that is loading normally.
- **Prometheus metrics** at `/metrics`: HTTP counters, pipeline and per-stage
  latency, cache hit rate, items per invoice, LLM accept/reject counts.
- **Structured JSON logging** with a request id propagated through every worker
  thread, so one invoice is traceable end to end.
- **Unified `pharmacy-ocr` CLI** (`serve`, `batch`, `cache warm|prune`, `audit`,
  `config`) replacing four ad-hoc scripts.
- **`unit_price` and `confidence`** on line items, and `page_count`,
  `ocr_engine`, `extraction_mode`, `service_version`, `processed_at` on results.
- **Cache eviction** by TTL and entry ceiling; the cache previously grew forever.
- **Test suite** — 146 tests covering parsing, models, settings, cache, intake,
  pipeline, CLI, and the full HTTP surface, running against a fake OCR engine.
- **Docker image**, compose stack, GitHub Actions CI, pre-commit hooks, Makefile.

### Fixed

- **`parsing_s` timing measured nothing.** The stopwatch started after parsing
  had already finished, so the metric was always ~0.
- **OCR cache missed on identical files.** The key included the file's mtime, so
  a copied, re-downloaded, or resynced invoice re-ran OCR for no reason. The key
  is now the content hash plus the OCR settings.
- **Cache writes were not atomic.** A crash mid-write left truncated JSON that
  every later run had to fail to parse. Writes now go through a temp file and an
  atomic replace, and an unreadable entry is discarded rather than raised.
- **Line reconstruction was O(n²).** Each token was compared against every open
  cluster, recomputing that cluster's mean every time. It is now a single linear
  sweep over y-sorted tokens.
- **Invoice-total detection picked the largest number** from any line containing
  "total", which selected a pre-tax "Total HTVA" or a stray header value over
  the real payable total. Markers are now ranked by specificity.
- **Identical invoice rows were silently deleted.** Global text deduplication
  removed legitimately repeated medicine lines. Only adjacent duplicates are
  collapsed now; the arithmetic checksum decides whether a repeat was real.
- **`except (asyncio.TimeoutError, Exception)`** in the LLM mapper swallowed
  genuine bugs and retried them. Retries are now bounded with backoff.
- **The Ollama client was never closed**, leaking its connection pool at
  shutdown.
- **Blocking I/O ran on the event loop.** Uploads were written with a
  synchronous copy inside an async handler, stalling every concurrent request.
- **PDF rendering held every page in memory at once** — roughly 26 MB per A4
  page at 300 DPI. Pages are now rendered, processed, and released one at a time.

### Changed

- Configuration is a validated, immutable `Settings` object instead of module
  globals mutated at runtime. In production the service refuses to start without
  `API_KEYS` (or an explicit `ALLOW_UNAUTHENTICATED=true`), with `DEBUG=true`,
  or with wildcard CORS.
- All dependencies are pinned. `requirements.txt` previously specified no
  versions at all, so two installs a week apart could produce different stacks.
- `opencv-python` was dropped in favour of `opencv-contrib-python`; installing
  both shadows the same `cv2` package.
- The OCR engine is behind an `OCREngine` protocol, which is what makes the
  pipeline testable without model weights.
- Endpoints are versioned under `/v1`. The unversioned `POST /extract` still
  works.

### Security

- Uploads are stored under generated names, never a caller-supplied filename.
- API docs (`/docs`, `/openapi.json`) are disabled by default.
- Security headers (`X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`) on every response.
- Internal tracebacks are never returned to clients; errors carry a stable code
  and the request id instead.
- The container runs as a non-root user.

## [2.0.0] - Prior work

- PaddleOCR general OCR with box geometry, deterministic row parser, invoice
  checksum, Ollama fallback, OCR cache, FastAPI `/extract`, batch script.
