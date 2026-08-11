# Runbook — installing and running the extraction engine

For whoever operates this in production. It covers installing, running a batch,
reading the output, and knowing when a result must not be trusted.

The engine takes an invoice (PDF / JPG / PNG) and writes one JSON file per
invoice. There is no database, no web service, and nothing to keep running
between batches.

---

## 1. Docker or a normal install?

**Install it normally. Do not use Docker for this.**

A Dockerfile is included and it works, but for this workload it costs more than
it gives:

| | Normal install | Docker |
|---|---|---|
| GPU access | Works as soon as the GPU wheel is installed | Needs NVIDIA Container Toolkit; on Windows that means WSL2 with GPU passthrough — awkward and easy to get wrong |
| Image size | ~4 GB in a venv | ~4 GB image *plus* the venv layers, rebuilt on every change |
| OCR model weights | Downloaded once to `~/.paddlex`, reused forever | Must be volume-mounted or re-downloaded per container |
| Reading results | Files land in `data/outputs/` | Needs a bind mount to get files out |

Speed on this workload depends almost entirely on the GPU, and GPU-in-Docker on
Windows is the single most fragile part of the whole setup. A plain virtualenv
avoids that entirely.

**Use Docker only if** the enterprise mandates containers for everything, or the
engine has to run on many machines that must stay byte-identical — and in that
case run it on Linux, where GPU passthrough is straightforward.

---

## 2. Install (one time, ~15 minutes)

Requires **Python 3.11 or newer**. Check with `python --version`.

```powershell
cd C:\path\to\pharmacy-ocr-service

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e . --no-deps
```

If PowerShell refuses to run the activate script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Optional but strongly recommended: the GPU

Without it, expect ~30 seconds per invoice. With it, expect several times
faster. See [GPU.md](GPU.md) — no code or config change is needed afterwards,
the engine detects the GPU on its own.

### Verify the install

```powershell
pharmacy-ocr --version
pharmacy-ocr config          # prints the effective settings, secrets redacted
```

The **first** extraction downloads the OCR models (a few hundred MB) into
`~/.paddlex`. That happens once, not per run.

---

## 3. Run a batch

Put invoices in a folder, point the engine at it, and collect the JSON:

```powershell
pharmacy-ocr batch --input C:\invoices --output C:\results
```

Useful flags:

| Flag | Why |
|---|---|
| `--limit 10` | Try a handful first before committing to thousands |
| `--concurrency 2` | Parallel invoices. 2 suits a 4-core / 8 GB machine |
| `--force` | Re-extract invoices that already have output (default is to skip) |
| `--photo-mode` | Force heavy preprocessing for phone photos |
| `--mode deterministic` | Never use the LLM. Fully reproducible — recommended |

**Resuming is automatic.** Invoices that already have an output file are
skipped, so an interrupted run over thousands of invoices picks up where it
stopped. Each file is written atomically — an interrupted run never leaves half
a JSON.

Exit codes: `0` success, `1` no input found, `2` at least one invoice failed.

---

## 4. Reading the output

One JSON per invoice, plus a `_batch_summary.json`.

**Route on `status`. It is the only field an operator needs to look at.**

| `status` | Meaning | Action |
|---|---|---|
| `ok` | Items extracted **and** their prices sum to the invoice total | Safe to process automatically |
| `needs_review` | Plausible but unverified — checksum failed, no total printed, or a row was dropped | A human must look before it is booked |
| `failed` | No result. `error` says why | Retry or investigate |

`checksum_ok: true` is the strongest guarantee available: the extracted line
totals independently reproduce the total printed on the invoice.

**`needs_review` is not a bug.** It is the engine refusing to vouch for
something it could not verify. On a medical invoice a confidently wrong price is
far worse than a flagged one. Expect a real share of invoices to land here and
staff a review queue accordingly.

### Reviewing a batch

```powershell
pharmacy-ocr audit --output C:\results
```

Prints how many are clean, groups the warnings into categories (a category
spiking usually means one supplier changed their layout), and lists every
invoice needing attention.

### Investigating one invoice

```powershell
pharmacy-ocr diagnose C:\invoices\problem.pdf --show-all
```

Shows every line the OCR read, what became an item, and which rule rejected the
rest.

---

## 5. Measuring speed honestly

The engine caches OCR results by file content. A second run over the same
invoices is roughly a hundred times faster because it re-reads JSON instead of
re-running OCR — **so a plain re-run tells you nothing about real speed.**

To measure genuine per-invoice cost, disable the cache for the run:

```powershell
$env:FORCE_OCR_REFRESH = "true"
pharmacy-ocr batch --input C:\invoices --output C:\results_speed `
                   --limit 10 --concurrency 1 --force
Remove-Item Env:\FORCE_OCR_REFRESH
```

Read `seconds_per_invoice` from `_batch_summary.json`. Use `--concurrency 1` for
a clean per-invoice figure; raise it to measure throughput instead.

Each result's `timings` block splits the cost across preprocessing, OCR, and
parsing — OCR is essentially all of it.

### Measured baseline (CPU, no GPU)

8 invoices, cache disabled, `--concurrency 1`, on an 8-core CPU:

| | seconds |
|---|---:|
| Typical invoice (6 of 8) | **25–33** |
| Invoice needing an orientation probe | 61 |
| Worst case observed | **273** |
| **Mean** | **68** |

The mean is far above the typical because cost is driven by **how many OCR
passes an invoice needs**, and a difficult one can stack up to four:

1. `primary` — the first pass
2. `retry` — re-render at higher DPI when the parse looks wrong
3. `retry_deep` — re-render higher still
4. `rot180` — orientation recovery when nothing parsed

Each pass is a full re-OCR of every page. One corpus invoice hit all four and
took 273 seconds.

**Preprocessing and parsing are noise** — under a second combined. All of the
time is OCR.

### Bounding the worst case

If predictable latency matters more than squeezing out the last few percent of
accuracy, cap the escalation:

```powershell
$env:ENABLE_DEEP_OCR_RETRY = "false"      # drops the third pass
$env:MAX_ORIENTATION_PROBES = "0"         # drops the orientation recovery pass
$env:RETRY_OCR_ON_REVIEW_FLAGS = "false"  # drops all DPI retries: 1 pass, always
```

With all three off, every invoice costs exactly one pass (~25–33 s) and the
worst case disappears. It costs accuracy: on the 84-invoice corpus the DPI
retries were worth about +9 verified invoices, and orientation recovery rescued
invoices that otherwise returned nothing at all. Measure both ways on your own
invoices before choosing.

---

## 6. Configuration

Settings come from environment variables or a `.env` file in the project root.
Copy `.env.example` and edit. `pharmacy-ocr config` prints what is actually in
effect.

The ones that matter in practice:

| Variable | Default | Effect |
|---|---|---|
| `EXTRACTION_MODE` | `hybrid` | `deterministic` never calls the LLM and is fully reproducible |
| `PHOTO_MODE` | `false` | `true` for phone photos; slower but handles shadows and folds |
| `OCR_DEVICE` | `auto` | Uses the GPU when available, else CPU |
| `BATCH_CONCURRENCY` | `2` | Parallel invoices |
| `USE_OCR_CACHE` | `true` | Leave on; makes re-runs near-instant |
| `LOG_FORMAT` | `json` | `text` is easier to read by eye |

---

## 7. Troubleshooting

**First run is very slow.** It is downloading OCR models. Once only.

**"no supported files found"** — check `--input`. Supported: `.pdf .png .jpg
.jpeg .bmp .tif .tiff .webp`.

**Everything comes back `needs_review`** — run `pharmacy-ocr audit` and read the
warning categories. If most say "no invoice total detected", the scans are
likely cropping off the totals line.

**A whole supplier extracts badly** — run `diagnose` on one of their invoices.
If the header row is not being recognised, that supplier's column labels
probably need adding to the synonym table in
`src/pharmacy_ocr/parsing/columns.py`.

**Out of memory** — lower `MAX_LONG_SIDE` (default 3300) or
`BATCH_CONCURRENCY`.

---

## 8. What this engine does not do

- **Handwriting** is not supported.
- **PDFs are capped** at `PDF_MAX_PAGES` (5) pages.
- Text that OCR never *detects* cannot be recovered by any setting. Creased,
  low-resolution, or shadowed captures lose whole columns. Scanning flat at 300
  DPI beats every software knob here.
- Accuracy figures in [ACCURACY.md](ACCURACY.md) come from a 7-invoice
  hand-labelled set. They are sound for catching systematic faults, but they are
  not a corpus-wide guarantee.
