# Accuracy: what is measured, what it costs, and what 99% would take

Everything here is measured on the 84-invoice reference corpus (74 PDFs, 10
phone photos) in `deterministic` mode on CPU. Reproduce with:

```bash
pharmacy-ocr batch --mode deterministic --output data/outputs
pharmacy-ocr audit --output data/outputs
```

## Measured against ground truth (the number that matters)

Since v3.2.0 there is a hand-labelled set, transcribed by reading the invoice
images, and a scorer that matches rows before comparing fields:

```bash
pharmacy-ocr evaluate --mode deterministic
```

| Metric | v3.1.0 | v3.2.0 |
|---|---:|---:|
| Invoices extracted exactly right | 1 / 7 | **6 / 7** |
| Row recall (rows found) | 78.6% | **100%** |
| Row precision (nothing invented) | 100% | **100%** |
| REFCODE accuracy | 71.4% | **100%** |
| MED accuracy | 50.0% | **94.4%** |
| QT accuracy | 71.4% | **100%** |
| prix accuracy | 78.6% | **100%** |

The labelled set is seven invoices — enough to catch the systematic faults
below, not enough to claim a corpus-wide figure. Expanding it is the single
highest-value next task; see "What 99% would require".

### Corpus-wide effect (84 invoices, checksum proxy)

The faults the labels exposed were systematic, so fixing them moved the whole
corpus, not just the labelled invoices:

| | v3.1.0 | v3.2.0 |
|---|---:|---:|
| `status=ok` | 45 | **63** |
| Invoices yielding no rows at all | 13 | **4** |
| Checksum passed / failed / no total | 45 / 27 / 12 | **66 / 15 / 3** |
| Line items extracted | 210 | **241** |
| Items carrying a REFCODE | 158 (75%) | **192 (80%)** |
| Failures | 0 | **0** |

The drop from 13 unreadable invoices to 4 is almost entirely the skew and
orientation work: those documents were not illegible, they were misread as a
column of vertical strips or split down the middle of every row.

### What the labels exposed that the checksum could not

Three faults were invisible to the invoice-total checksum because they do not
change the arithmetic:

1. **Sideways pages were never corrected.** PaddleOCR's own orientation
   classifier missed three corpus invoices; their text boxes came back 80%+
   taller than wide, so a "row" was a vertical strip of one character and two
   of the three yielded nothing at all. The pipeline now measures box aspect
   ratio after the first pass and re-OCRs upright when the page is sideways.
   This alone took row recall from 78.6% to 100%.
2. **The dosage-form column was glued onto the product name.**
   `ECOREX 1% CREME 30GR` came out as `ECOREX 1% CREME 30GR CREME`. Flattening
   a table row to a string discards the x-coordinates that say where a column
   ends. Extraction is now column-aware.
3. **`10*10` lost its second `10`.** A 10x10 cm dressing had its trailing
   dimension stripped as if it were a quantity.

### Why label matching cannot work, and what replaced it

The same heading means different things on different invoices. On one CNAM
invoice the line total is `Tot. Fact` and `Perçu` is the patient's copay; on
another the line total *is* `Mt. Percu`. Reading the label alone gets one of
them wrong whichever rule you pick.

Headings are therefore mapped to *semantic roles* — `Référence`, `Code PCT`,
`CIP`, and `Article` all mean `code`; `Produit`, `Libellé`, `Désignation`, and
`Médicament` all mean `name` — and the geometry does the rest. A new supplier
usually needs a synonym, not new logic.

## Where it stands

| Metric | v3.0.0 | v3.1.0 |
|---|---:|---:|
| Invoices processed without error | 84 / 84 | 84 / 84 |
| `status=ok` (extracted **and** arithmetically verified) | 42 (50%) | 43 (51%) |
| Checksum passed / failed / not checkable | 44 / 27 / 13 | 45 / 27 / 12 |
| Line items extracted | 212 | 210 |
| Items carrying a REFCODE | 153 (72%) | 158 (75%) |
| Obvious junk rows in output | 1 | 0 |

Throughput, measured with a cold OCR cache:

| Configuration | Per invoice | 84 invoices, 2 concurrent |
|---|---:|---:|
| Mobile models, no retry | ~32 s | ~22 min |
| Mobile models, retry enabled (default) | ~40 s | ~55 min |
| Server recognizer | 60–101 s | ~2.5 h |

## Running the benchmarks yourself

All commands are run from the project root with the venv's Python. On Windows
PowerShell:

```powershell
cd c:\Users\moham\OneDrive\Desktop\Pharamacy-IA\pharmacy-ocr-service
$env:PYTHONPATH = "$PWD\src"
$env:ENVIRONMENT = "development"
$py = ".\.venv\Scripts\python.exe"
```

**Accuracy against the hand-labelled invoices** (seconds, and exact per-field
scores):

```powershell
& $py -m pharmacy_ocr evaluate --mode deterministic
```

**True cold-start speed.** This is the only honest SLA measurement. The OCR
cache makes a second run look ~100x faster than it is, so it has to be disabled
— otherwise you are timing a JSON read, not OCR:

```powershell
$env:FORCE_OCR_REFRESH = "true"
& $py -m pharmacy_ocr batch --mode deterministic --limit 10 `
      --concurrency 1 --force --output data\outputs_speed
Remove-Item Env:\FORCE_OCR_REFRESH
```

Read `seconds_per_invoice` from the printed summary (also written to
`data/outputs_speed/_batch_summary.json`). Use `--concurrency 1` for a clean
per-invoice number; raise it for throughput.

**One invoice, with the per-stage breakdown** — `timings` in the output JSON
splits preprocessing, OCR, and parsing, which is how you see where the time
actually goes:

```powershell
& $py -m pharmacy_ocr batch --mode deterministic --limit 1 --force `
      --output data\outputs_speed
Get-Content data\outputs_speed\*.json | Select-String "timings" -Context 0,12
```

**Why one invoice reads badly** — every line, and which rule rejected it:

```powershell
& $py -m pharmacy_ocr diagnose data\invoices\invoice1.jpeg --show-all
```

## The measurement problem, stated plainly

**There is no ground truth for this corpus.** Every number above is the
*checksum* proxy: does the sum of extracted line totals equal the printed
invoice total. That proxy is not accuracy, and it is wrong in both directions:

- **It under-reports.** 12 of 84 invoices print no total the OCR can find.
  Those can never reach `status=ok` however perfect the extraction is. That
  alone caps checksum-verified at 72/84 = **86%**.
- **It over-reports.** A passing checksum proves the *prices* add up. It says
  nothing about whether the medicine names or REFCODEs are right. An invoice can
  pass with a null REFCODE on every row.

So "99% accuracy" is not currently a measurable claim. The first step toward it
is not more parser rules — it is **labels**.

### The REFCODE example

REFCODE is null on 25% of extracted items. Three real causes were found and
fixed in v3.1.0 (orphan code lines, letter-prefixed references like `PFC034`,
and a mangled `Tota!` line becoming an item). That moved REFCODE coverage 72% →
75%.

The remaining 25% cannot be attributed without labels, because two very
different situations look identical in the output:

1. The invoice printed a code and OCR missed it — a **defect**.
2. The product genuinely has no PCT code. Parapharmacy items (`PHYSIOMER JET
   NORMAL`) legitimately have none, and the reference examples this parser was
   built against show `REFCODE: null` for exactly that product.

An earlier estimate in this project claimed 71% of missing codes were
"recoverable because a code is visible nearby". That estimate was wrong: it
counted the code belonging to the *previous product row* as recoverable. Only
about 10% were genuinely orphaned codes, which is what the measured 3-point gain
reflects.

## Why the OCR model upgrade does not help

The obvious lever — swap PaddleOCR's `mobile` models for `server` models — was
tested and does not work here:

| Combination | Result |
|---|---|
| `PP-OCRv5_server_det` (detection) | **Segfaults** during inference (exit 139) on this PaddlePaddle 3.3.1 CPU build. Not a memory problem — 0.6 GB resident when it dies. |
| `PP-OCRv5_server_rec` (recognition) | Runs, but produced **byte-identical output** on the three test invoices at 60–101 s versus 22–44 s. |

That result is more informative than it looks. The better *recognizer* changing
nothing means the failures are not misread characters — they are text regions
that were **never detected**. On a low-resolution CamScanner capture the price
column simply is not found, so there is nothing for a better reader to read. And
the better *detector*, which is the component that would help, crashes.

This is the ceiling: **detection quality on poor source scans**, not parsing and
not recognition.

## What 99% would actually require

Ordered by expected gain per unit of effort.

| # | Action | Expected gain | Cost |
|---|---|---|---|
| 1 | **Label 25–30 invoices by hand** (items, REFCODE, QT, prix) and add a scored regression harness | None directly — but nothing below can be verified without it | ~4 h of a pharmacist's time, ~4 h to build the harness |
| 2 | **Fix source capture**: scan at 300 DPI flatbed instead of phone/CamScanner for the worst ~20% | Large. This is where the losses are | Process change, no code |
| 3 | **GPU PaddlePaddle build** so `server_det` is usable | Unknown, plausibly large — it is the component that matters | ~1 day, needs a CUDA GPU |
| 4 | **Enable `hybrid` mode** (Ollama fallback) | Small. Helps where OCR read the text but the parser failed; useless where text was never detected | ~2 h to set up and validate |
| 5 | **Auto-detect scanned-photo PDFs** and apply `photo_mode` per document | Small and two-sided — measured on one invoice it recovered the medicine name but mangled the price | ~1 day |
| 6 | Continue parser rule work | Diminishing. The last three fixes bought 1 checksum and 5 REFCODEs | ongoing |

**Realistic targets:**

- **~60–65% verified** is reachable with items 1 and 4, plus finishing the
  parser work the labels would then expose. Estimate ~1 week.
- **~85%** is the structural ceiling of checksum verification on this corpus
  (12 invoices print no checkable total). Reaching it needs item 2 or 3.
- **99% of invoices auto-approved with no human review is not achievable** on
  input of this quality, by this or any pipeline. Roughly a fifth of these
  documents are photographs of printouts in which a human also cannot reliably
  read the price column.

**99% is achievable on a different metric**, and it is probably the one worth
committing to: *99% of invoices are either correct or correctly flagged for
review.* That is already close — 0 of 84 failed silently, and the checksum is
what makes the flagging trustworthy. The lever there is not accuracy, it is
keeping the review queue small.

## The failure mode this design protects

`needs_review` is not a defect. It is the service declining to vouch for a
result it cannot verify. On a medical invoice, a confidently wrong price is far
worse than a flagged one, which is why the checksum gates `status` rather than
being reported as an afterthought. Any push toward a higher `ok` percentage
should be checked against that: raising the number by loosening the check would
make the output less trustworthy, not more.
