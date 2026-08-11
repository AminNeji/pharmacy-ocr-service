"""Upload a folder of invoices and write one JSON file per invoice.

    python examples/upload_folder.py ./invoices ./results
    python examples/upload_folder.py ./invoices ./results --url https://ocr.example.com --key sk_live_x
    python examples/upload_folder.py ./invoices ./llm_results --key sk_live_x --mode llm
    python examples/upload_folder.py ./invoices ./llm_results --key sk_live_x --mode llm --only-needs-review-from ./results
    python examples/upload_folder.py ./invoices ./results --key sk_live_x --resume-batch BATCH_ID

Only needs `httpx`. Copy it to the client machine; it does not import the
service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
POLL_SECONDS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder containing invoices")
    parser.add_argument("output", type=Path, help="folder to write results into")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="service base URL")
    parser.add_argument("--key", default="", help="API key, if the service requires one")
    parser.add_argument("--chunk", type=int, default=50, help="files per request (server caps this)")
    parser.add_argument("--photo-mode", action="store_true", help="force phone-photo preprocessing")
    parser.add_argument("--resume-batch", help="poll and collect an already-submitted batch id")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "hybrid", "llm"),
        help="override extraction mode for this upload",
    )
    parser.add_argument(
        "--only-needs-review-from",
        type=Path,
        help="previous results folder; upload only invoices whose old JSON has status=needs_review",
    )
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed certificate")
    args = parser.parse_args()

    invoices = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in SUPPORTED)
    if args.only_needs_review_from:
        review_stems = _needs_review_stems(args.only_needs_review_from)
        invoices = [path for path in invoices if path.stem in review_stems]
        print(f"found {len(invoices)} invoice(s) needing review from {args.only_needs_review_from}")

    if not invoices:
        print(f"no invoices in {args.folder}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    headers = {"Connection": "close"}
    if args.key:
        headers["X-API-Key"] = args.key
    params = {"photo_mode": "true"} if args.photo_mode else {}
    if args.mode:
        params["mode"] = args.mode

    # No read timeout: a batch of 50 can occupy the server for the better part
    # of an hour, and only the poll responses are actually slow to arrive.
    client = httpx.Client(
        base_url=args.url,
        headers=headers,
        verify=not args.insecure,
        timeout=httpx.Timeout(60.0, read=None),
        limits=httpx.Limits(max_keepalive_connections=0),
    )

    started = time.perf_counter()
    if args.resume_batch:
        return _poll_and_write_results(client, args.resume_batch, args.output, started=started)

    failures = 0
    written = 0
    for start in range(0, len(invoices), args.chunk):
        group = invoices[start : start + args.chunk]
        print(f"\nuploading {len(group)} file(s)...")
        files = [("files", (path.name, path.read_bytes())) for path in group]
        response = client.post("/v1/batches", files=files, params=params)
        if response.status_code != 202:
            print(f"upload rejected: {response.status_code} {response.text}", file=sys.stderr)
            return 1

        batch = response.json()
        print(f"batch {batch['id']}: {batch['accepted']} queued")
        for item in batch["rejected"]:
            print(f"  skipped {item['filename']}: {item['message']}")

        while True:
            state = _get_json(client, f"/v1/batches/{batch['id']}")
            counts = state["counts"]
            done = counts["succeeded"] + counts["failed"]
            print(f"  {done}/{counts['total']} done", end="\r", flush=True)
            if state["status"] == "completed":
                break
            time.sleep(POLL_SECONDS)

        chunk_failures, chunk_written = _write_batch_results(
            _get_json(client, f"/v1/batches/{batch['id']}/results"), args.output
        )
        failures += chunk_failures
        written += chunk_written

    print(f"\nwrote results to {args.output}")
    _print_timing(started, written, failures)
    return 1 if failures else 0


def _get_json(client: httpx.Client, path: str, *, attempts: int = 10) -> dict:
    """GET JSON with retries for safe polling requests."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(path)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt == attempts:
                raise
            wait_s = min(30, 2 * attempt)
            print(f"\n  poll failed ({exc}); retrying in {wait_s}s...")
            time.sleep(wait_s)
    raise RuntimeError("unreachable")


def _poll_and_write_results(
    client: httpx.Client, batch_id: str, output_dir: Path, *, started: float
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        state = _get_json(client, f"/v1/batches/{batch_id}")
        counts = state["counts"]
        done = counts["succeeded"] + counts["failed"]
        print(f"  {done}/{counts['total']} done", end="\r", flush=True)
        if state["status"] == "completed":
            break
        time.sleep(POLL_SECONDS)

    failures, written = _write_batch_results(_get_json(client, f"/v1/batches/{batch_id}/results"), output_dir)
    print(f"\nwrote results to {output_dir}")
    _print_timing(started, written, failures)
    return 1 if failures else 0


def _write_batch_results(results: dict, output_dir: Path) -> tuple[int, int]:
    print(f"  {len(results['results'])} finished" + " " * 20)
    failures = 0
    written = 0
    for job in results["results"]:
        name = Path(job["filename"]).stem
        if job["result"] is None:
            print(f"  FAILED {name}: {job['error']}", file=sys.stderr)
            failures += 1
            continue
        destination = output_dir / f"{name}.json"
        destination.write_text(json.dumps(job["result"], ensure_ascii=False, indent=2), encoding="utf-8")
        result = job["result"]
        written += 1
        print(f"  {result['status']:<12} {len(result['items']):>3} items  {destination.name}")
    return failures, written


def _print_timing(started: float, written: int, failures: int) -> None:
    elapsed = time.perf_counter() - started
    processed = written + failures
    print(f"elapsed: {elapsed:.1f}s")
    if processed:
        print(f"seconds per invoice: {elapsed / processed:.1f}s")


def _needs_review_stems(results_dir: Path) -> set[str]:
    stems: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped result {path.name}: {exc}", file=sys.stderr)
            continue
        if data.get("status") == "needs_review":
            stems.add(path.stem)
    return stems


if __name__ == "__main__":
    sys.exit(main())
