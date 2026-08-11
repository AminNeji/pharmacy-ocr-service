# Testing the HTTP / HTTPS endpoints

Everything below is copy-pasteable. Commands are PowerShell; `curl.exe` is the
real curl on Windows (plain `curl` is an alias for `Invoke-WebRequest`, which
takes different flags).

---

## 1. Start the server (HTTP)

```powershell
cd c:\Users\moham\OneDrive\Desktop\Pharamacy-IA\pharmacy-ocr-service
$env:PYTHONPATH = "$PWD\src"
$env:ENVIRONMENT = "development"   # 'production' refuses to start without API_KEYS
$env:DOCS_ENABLED = "true"
$env:API_KEYS = "sk_test_123"
.\.venv\Scripts\python.exe -m pharmacy_ocr serve
```

It prints the URL it is serving on. Leave this window open and use a second one
for the requests below.

**Interactive docs:** open <http://127.0.0.1:8000/docs>. Click *Authorize*,
paste the API key, and you can upload an invoice from the browser — the fastest
way to show someone the service working.

---

## 2. Check it is alive

```powershell
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/health/ready
```

`ready` returns 503 while the OCR models are still loading. That is intended:
the process is alive but cannot serve an extraction yet.

---

## 3. Upload an invoice

**Synchronous** — one invoice, wait for the answer:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/extract" `
         -H "X-API-Key: sk_test_123" `
         -F "file=@data\invoices\invoice2.jpeg"
```

**Asynchronous** — the right call for anything batch-shaped, because an
extraction takes tens of seconds and most clients and proxies give up first:

```powershell
# 1. submit -> 202 with a job id
curl.exe -X POST "http://127.0.0.1:8000/v1/jobs" `
         -H "X-API-Key: sk_test_123" `
         -F "file=@data\invoices\invoice2.jpeg"

# 2. poll until status is 'succeeded' or 'failed'
curl.exe "http://127.0.0.1:8000/v1/jobs/PASTE_JOB_ID" -H "X-API-Key: sk_test_123"
```

Optional query parameters on both: `?photo_mode=true` and
`?mode=deterministic|hybrid|llm`.

---

## 3b. Upload a whole folder

One request, one id to poll. Repeat the `files` part per invoice — note the
field name is `files`, plural, unlike the single-invoice endpoints:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/batches" `
         -H "X-API-Key: sk_test_123" `
         -F "files=@data\invoices\invoice1.jpeg" `
         -F "files=@data\invoices\invoice2.jpeg"
```

To send a real folder without typing every name:

```powershell
$parts = @()
Get-ChildItem data\invoices -Include *.pdf,*.jpg,*.jpeg,*.png -Recurse |
    Select-Object -First 20 |
    ForEach-Object { $parts += "-F"; $parts += "files=@$($_.FullName)" }
curl.exe -X POST "http://127.0.0.1:8000/v1/batches" -H "X-API-Key: sk_test_123" @parts
```

The response tells you what was queued and what was skipped:

```json
{
  "id": "948ce442cf08487283c4c09dd4c01079",
  "accepted": 2,
  "rejected": [
    { "filename": "Thumbs.db", "code": "unsupported_file_type",
      "message": "Unsupported file type '.db'." }
  ],
  "jobs": [{ "id": "...", "filename": "invoice1.jpeg", "status": "queued" }]
}
```

Unreadable files are reported rather than failing the upload — a real folder has
a `Thumbs.db` in it, and failing the whole batch over one would force every
client to reimplement this service's allow-list.

Then poll, and collect when it says `completed`:

```powershell
curl.exe "http://127.0.0.1:8000/v1/batches/BATCH_ID" -H "X-API-Key: sk_test_123"
# {"status":"running","counts":{"queued":1,"running":1,"succeeded":0,"total":2}}

curl.exe "http://127.0.0.1:8000/v1/batches/BATCH_ID/results" -H "X-API-Key: sk_test_123"
```

The two are split so polling stays cheap: progress carries no extraction
payloads, which matters when you are checking a 50-invoice batch every few
seconds.

Ceilings are `MAX_BATCH_FILES` (50) and `MAX_BATCH_UPLOAD_BYTES` (200 MB); each
file is still individually capped by `MAX_UPLOAD_BYTES`. Exceeding either
returns `413 batch_too_large`. If the queue lacks room for the whole batch you
get `503 service_overloaded` and **nothing** is queued — partial acceptance
would leave you reconciling which half got in.

Rather than scripting curl, use
[examples/upload_folder.py](../examples/upload_folder.py), which chunks, polls,
and writes one JSON per invoice:

```powershell
python examples\upload_folder.py data\invoices results --key sk_test_123
```

---

## 4. Check the failure paths

These are worth demonstrating, because "it works" is only half the story:

```powershell
# no key -> 401
curl.exe -X POST "http://127.0.0.1:8000/v1/extract" -F "file=@data\invoices\invoice2.jpeg"

# wrong file type -> 415 with a machine-readable code
curl.exe -X POST "http://127.0.0.1:8000/v1/extract" `
         -H "X-API-Key: sk_test_123" -F "file=@README.md"
```

Every non-2xx response has the same shape:

```json
{
  "error": {"code": "unsupported_file_type", "message": "...", "detail": {...}},
  "request_id": "0f9c1a2b3c4d5e6f"
}
```

`request_id` is echoed in the `X-Request-ID` header and appears in every log
line for that request.

---

## 5. HTTPS

### Make a certificate

For local testing, self-signed is fine:

```powershell
$openssl = "C:\Program Files\Git\mingw64\bin\openssl.exe"
mkdir certs -Force
& $openssl req -x509 -newkey rsa:2048 -nodes `
    -keyout certs\server.key -out certs\server.crt `
    -days 365 -subj "/CN=localhost" `
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

For production, use a real certificate — Let's Encrypt via certbot, or whatever
your enterprise issues. The service does not care which, it just needs the two
PEM files.

### Serve HTTPS

```powershell
$env:SSL_CERTFILE = "$PWD\certs\server.crt"
$env:SSL_KEYFILE  = "$PWD\certs\server.key"
.\.venv\Scripts\python.exe -m pharmacy_ocr serve
```

It prints `serving HTTPS on https://...`. Setting only one of the two variables
is rejected at startup rather than silently serving plaintext.

### Call it

`-k` tells curl to accept the self-signed certificate. **Drop `-k` in
production** — if a real certificate needs `-k`, something is wrong.

```powershell
curl.exe -k https://127.0.0.1:8000/health/live

curl.exe -k -X POST "https://127.0.0.1:8000/v1/extract" `
         -H "X-API-Key: sk_test_123" `
         -F "file=@data\invoices\invoice2.jpeg"
```

Verified behaviour on this setup:

| Check | Result |
|---|---|
| HTTPS health probe | 200 |
| Plain HTTP against the TLS port | refused |
| Client that verifies certificates, self-signed cert | rejected |
| Invoice upload over TLS | 200, items returned |
| Missing API key over TLS | 401 |

### TLS here, or at a reverse proxy?

Both work. Terminating TLS at nginx/Caddy/a load balancer is the more common
production choice — it handles certificate renewal, HTTP/2, and redirects for
you, and the service then listens on plain HTTP on a private interface. Serving
TLS directly is simpler when the service is the only thing on the box.

If you do put a proxy in front, set `TRUST_PROXY_HEADERS=true` **only** if that
proxy overwrites `X-Forwarded-For`. If it merely appends, clients can forge the
header and slip the rate limiter.

---

## 6. Python client

```python
import httpx

client = httpx.Client(base_url="https://127.0.0.1:8000", verify=False)  # verify=True in production
headers = {"X-API-Key": "sk_test_123"}

with open("invoice.pdf", "rb") as handle:
    job = client.post("/v1/jobs", files={"file": handle}, headers=headers).json()

while True:
    state = client.get(f"/v1/jobs/{job['id']}", headers=headers).json()
    if state["status"] in ("succeeded", "failed"):
        break

for item in state["result"]["items"]:
    print(item["REFCODE"], item["MED"], item["QT"], item["prix"])
```

Route on `result["status"]`: `ok` is arithmetically verified, `needs_review`
needs a human, `failed` has an `error`.

---

## 7. Endpoint reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/extract` | key | Synchronous extraction |
| `POST` | `/v1/jobs` | key | Queue an invoice, returns `202` + id |
| `GET` | `/v1/jobs/{id}` | key | Poll a job and read its result |
| `GET` | `/v1/jobs` | key | Job counts by status |
| `POST` | `/v1/batches` | key | Queue a folder, returns `202` + batch id |
| `GET` | `/v1/batches/{id}` | key | Batch progress, no payloads |
| `GET` | `/v1/batches/{id}/results` | key | Every extraction in the batch |
| `GET` | `/health/live` | none | Liveness |
| `GET` | `/health/ready` | none | Readiness (503 while models load) |
| `GET` | `/health` | none | Health + non-secret config |
| `GET` | `/metrics` | key | Prometheus metrics |
| `GET` | `/docs` | none | Swagger UI (only when `DOCS_ENABLED=true`) |

`POST /extract` without the `/v1` prefix still works for older clients.
