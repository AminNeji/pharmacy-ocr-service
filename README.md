# Pharmacy OCR Service

This project extracts invoice data from pharmacy invoice PDFs/images and returns
JSON with an arithmetic self-check. Everything runs locally — no invoice leaves
the machine.

Commands are given for **Windows PowerShell** and **Linux/macOS**. Run them from
the project folder (the one containing `requirements.txt`).

> **Windows only:** use `curl.exe`, not `curl`. In PowerShell, `curl` is an alias
> for `Invoke-WebRequest` and takes different flags.

---

## 1. Install

### Step 1: install Python

**Python 3.11 or newer.** Check it:

```powershell
python --version
```

### Step 2: get the project

```powershell
git clone https://github.com/AminNeji/pharmacy-ocr-service.git
cd pharmacy-ocr-service
```

### Step 3: create the virtual environment

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### Step 5: check the install

```bash
pharmacy-ocr --version
pharmacy-ocr config
```

If both run, the install is good.

---

## 2. Start the server

**Windows**

```powershell
$env:ENVIRONMENT = "development"
$env:API_KEYS = "sk_local_dev"
$env:DOCS_ENABLED = "true"
$env:LOG_FORMAT = "text"

pharmacy-ocr serve
```

**Linux / macOS**

```bash
ENVIRONMENT=development API_KEYS=sk_local_dev DOCS_ENABLED=true LOG_FORMAT=text \
  pharmacy-ocr serve
```

Leave this terminal open. The API is at `http://127.0.0.1:8000`.

**The first run downloads the OCR models (~200 MB).** Wait for it to finish.

### Browser testing

Open `http://127.0.0.1:8000/docs`, click **Authorize**, and enter only:

```text
sk_local_dev
```

Do not type `X-API-Key:` there — Swagger adds the header for you.

### Production note

`ENVIRONMENT=production` is the default and **refuses to start without
`API_KEYS`**. That is deliberate: an open endpoint that accepts file uploads and
burns CPU on OCR is the easiest way this service gets abused. Generate a real key
with:

```bash
python -c "import secrets; print('sk_' + secrets.token_urlsafe(32))"
```

Set it in a `.env` file (copy `.env.example`) or as an environment variable.
Never commit a real key.

---

## 3. Check the server is working

Open a second terminal while the server runs.

```powershell
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/health/ready
```

If `/health/ready` returns `503`, wait — OCR models are still loading.

---

## 4. HTTP requests

If you get `401 unauthorized`, the API key is missing. It is `sk_local_dev` in
the examples above.

Replace `path\to\invoice.pdf` with a real file path.

### One invoice, wait for the answer

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/extract" `
  -H "X-API-Key: sk_local_dev" `
  -F "file=@path\to\invoice.pdf"
```

Works the same for `.jpg`, `.png`, `.jpeg`, `.tif`, `.bmp`, `.webp`.

Takes 25–60 seconds. Fine for one file from a browser or script.

### One invoice as a job

Better when OCR takes a long time, and required behind any proxy — most cut
requests off at 60 seconds.

Submit:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/jobs" `
  -H "X-API-Key: sk_local_dev" `
  -F "file=@path\to\invoice.pdf"
```

Copy the `id` from the response, then poll until `status` is `succeeded` or
`failed`:

```powershell
curl.exe "http://127.0.0.1:8000/v1/jobs/PASTE_JOB_ID_HERE" `
  -H "X-API-Key: sk_local_dev"
```

### Many invoices at once

The API does not receive a folder as one object. It receives many files, all
with the form name `files` (plural).

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/batches" `
  -H "X-API-Key: sk_local_dev" `
  -F "files=@path\to\invoice1.pdf" `
  -F "files=@path\to\invoice2.jpg" `
  -F "files=@path\to\invoice3.png"
```

Files the service cannot read (a stray `Thumbs.db`, a `.txt`) are listed under
`rejected` and the rest still run.

Copy the batch `id`, then poll progress:

```powershell
curl.exe "http://127.0.0.1:8000/v1/batches/PASTE_BATCH_ID_HERE" `
  -H "X-API-Key: sk_local_dev"
```

When `status` is `completed`, get the results:

```powershell
curl.exe "http://127.0.0.1:8000/v1/batches/PASTE_BATCH_ID_HERE/results" `
  -H "X-API-Key: sk_local_dev"
```

Limits per request: 50 files, 200 MB total, 20 MB per file.

### Upload a whole folder the easy way

The helper script reads a folder, uploads in batches, polls, and writes one JSON
per invoice.

```powershell
python examples\upload_folder.py "path\to\invoice-folder" "path\to\results" `
  --url http://127.0.0.1:8000 `
  --key sk_local_dev
```

It needs only `httpx`. If you see `No module named httpx`:

```bash
pip install httpx
```

If the script crashes *after* printing `batch ... queued`, do not upload again.
Resume that same batch:

```powershell
python examples\upload_folder.py "path\to\invoice-folder" "path\to\results" `
  --url http://127.0.0.1:8000 `
  --key sk_local_dev `
  --resume-batch PASTE_BATCH_ID_HERE
```

### Recheck only `needs_review` with the LLM

After a normal run, rerun only invoices whose previous JSON says
`"status": "needs_review"`:

```powershell
python examples\upload_folder.py "path\to\invoice-folder" "path\to\llm-results" `
  --url http://127.0.0.1:8000 `
  --key sk_local_dev `
  --mode llm `
  --only-needs-review-from "path\to\results"
```

Use a separate output folder so you can compare the normal result against the
LLM result before trusting it. This mode needs Ollama running
(`ollama pull qwen2.5:3b`).

---

## 5. HTTPS

For local testing, plain HTTP is normally what you want. Use HTTPS only when you
start the server with a certificate.

### Create a test certificate

**Windows** (needs Git for Windows, which ships openssl):

```powershell
mkdir certs -Force
$openssl = "C:\Program Files\Git\mingw64\bin\openssl.exe"

& $openssl req -x509 -newkey rsa:2048 -nodes `
  -keyout certs\server.key `
  -out certs\server.crt `
  -days 365 `
  -subj "/CN=localhost" `
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

**Linux / macOS**

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/server.key -out certs/server.crt \
  -days 365 -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

### Start with HTTPS

```powershell
$env:ENVIRONMENT = "development"
$env:API_KEYS = "sk_local_dev"
$env:SSL_CERTFILE = "$PWD\certs\server.crt"
$env:SSL_KEYFILE = "$PWD\certs\server.key"

pharmacy-ocr serve
```

The API is now `https://127.0.0.1:8000`. Setting only one of the two SSL
variables is refused at startup rather than silently serving plaintext.

### HTTPS requests

`-k` is needed only because this certificate is self-signed. **Drop `-k` in
production** — if a real certificate needs it, something is wrong.

```powershell
curl.exe -k https://127.0.0.1:8000/health/live

curl.exe -k -X POST "https://127.0.0.1:8000/v1/extract" `
  -H "X-API-Key: sk_local_dev" `
  -F "file=@path\to\invoice.pdf"
```

In production, terminating TLS at a reverse proxy (nginx, Caddy) is usually
easier to keep renewed.

---

## 6. Run without the server

Put invoices in a folder and run batch mode directly:

```powershell
mkdir data\invoices -Force

pharmacy-ocr batch --input data\invoices --output data\outputs
pharmacy-ocr audit --output data\outputs
```

`batch` resumes by default — invoices that already have output are skipped.

Other useful commands:

```bash
pharmacy-ocr batch --photo-mode        # phone photos, heavier preprocessing
pharmacy-ocr batch --force             # re-extract everything
pharmacy-ocr diagnose path/to/one.pdf  # why did this invoice extract badly?
pharmacy-ocr config                    # effective settings, secrets hidden
```

---

## 7. Measure real speed

To time it like a fresh machine, with no help from the OCR cache:

Stop the server (`Ctrl + C`), then restart it with:

```powershell
$env:ENVIRONMENT = "development"
$env:API_KEYS = "sk_local_dev"
$env:LOG_FORMAT = "text"
$env:FORCE_OCR_REFRESH = "true"
$env:MAX_CONCURRENT_EXTRACTIONS = "1"

pharmacy-ocr serve
```

From another terminal:

```powershell
python examples\upload_folder.py "path\to\invoice-folder" "path\to\results_speed" `
  --url http://127.0.0.1:8000 `
  --key sk_local_dev `
  --chunk 1
```

The script prints `elapsed` and `seconds per invoice`.

Afterwards, turn the cache back on:

```powershell
Remove-Item Env:\FORCE_OCR_REFRESH -ErrorAction SilentlyContinue
Remove-Item Env:\MAX_CONCURRENT_EXTRACTIONS -ErrorAction SilentlyContinue
```

Then restart the server normally.

### Measured baseline

41 invoices on an Intel i7-11370H (4 cores), CPU only, one at a time:

| | |
|---|---|
| Median | **40 s** |
| Mean | 67 s |
| Fastest | 19 s |
| Slowest | 276 s |

A GPU is roughly 5× faster — see [docs/GPU.md](docs/GPU.md). It needs no code or
config change; the service detects the GPU by itself.

---

## 8. Run tests

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps

pytest                    # all tests, a few seconds
pytest tests/test_api.py  # only the API tests
```

The suite uses a fake OCR engine, so it needs no model weights.

---

## 9. Reading the results

`status` tells you what to do:

| status | meaning | action |
|---|---|---|
| `ok` | extracted **and** totals verified | safe to process automatically |
| `needs_review` | extracted but unverified | human check |
| `failed` | extraction failed | retry or investigate |

Main output fields:

| field | meaning |
|---|---|
| `items` | extracted medicine rows |
| `REFCODE` | medicine/product code |
| `MED` | medicine name |
| `QT` | quantity |
| `prix` | line total price (not unit price) |
| `checksum_ok` | `true` only when line totals match the printed invoice total |

`checksum_ok` is the field to trust — it is independent arithmetic evidence that
the extraction is right. `null` means no total was found, which is **not** the
same as agreement.

**Never book a `needs_review` invoice without a human looking at it.**

---

## More documentation

| Document | For |
|---|---|
| [docs/GPU.md](docs/GPU.md) | Running on a GPU server |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Day-to-day operation |
| [docs/API.md](docs/API.md) | Every endpoint, HTTP and HTTPS |
| [docs/ACCURACY.md](docs/ACCURACY.md) | What the accuracy numbers mean |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How extraction works; changing the code |

---

MIT — see [LICENSE](LICENSE). History in [CHANGELOG.md](CHANGELOG.md).
