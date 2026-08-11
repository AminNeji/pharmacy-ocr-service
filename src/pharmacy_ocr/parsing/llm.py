"""Ollama mapper used as a fallback for layouts the row parser cannot handle.

The deterministic parser stays the source of truth whenever it can parse a row.
This mapper receives row-oriented text rather than raw detector order, because
invoice line structure is the difference between correct extraction and
plausible-looking fabrication.

Changes from the previous implementation: the client is owned and closed rather
than leaked in a module global, retries back off instead of hammering, and
`except (asyncio.TimeoutError, Exception)` — which swallowed genuine bugs and
retried them — is gone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from pharmacy_ocr.domain.errors import LLMMappingError
from pharmacy_ocr.observability import metrics
from pharmacy_ocr.ocr.types import OCRPage
from pharmacy_ocr.parsing.rules import extract_lines
from pharmacy_ocr.settings import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You extract ONLY medicine/product line items from Tunisian pharmacy invoices.
Return exactly one strict JSON object and no prose:
{"items":[{"REFCODE":string|null,"MED":string,"QT":integer|null,"prix":number|null}]}

Hard rules:
- Use the OCR ROWS exactly as evidence. Do not use outside knowledge.
- REFCODE is the product/PCT code at the beginning of a product row. It is usually 5-13 digits.
- If a product row starts with digits, copy that full digit sequence as REFCODE. Never shorten it.
- Return REFCODE as null only when the product row truly has no printed code.
- MED must be verbatim from the product row, excluding REFCODE, quantity, unit price, line total, refund columns, and trailing markers like x or =.
- QT is always an integer quantity. A decimal/millime value is never QT.
- Tunisian prices use 3 millimes digits. OCR may print 5 320, 5.320, or 5,320; all mean 5.320.
- prix is the LINE TOTAL, not the unit price and not the CNAM refund.
- If a row has "QT unit total", use total. Example: "2 17.340 34.680" -> QT 2, prix 34.680.
- If a row has "1 x 5 320 = 5 320 1 596", use QT 1 and prix 5.320. Ignore the final refund/copay number.
- Ignore all totals/checksums: Total, Total Facture, Total HTVA, Total TVA, Mnt Total, CNAM, Percu, remise, timbre.
- Ignore pharmacy address, phone/fax, dates, patient/assure/matricule, stamps, signatures, barcode numbers.
- Never merge two product rows. Never split one product row into multiple items.
- If the OCR is ambiguous, still return the visible medicine row with null only for the ambiguous field.

Few-shot examples:

OCR ROWS:
1. 303154 COPRED ODT 20 MG CP ORODISP B110...X. 5 320
2. = 5 320 1 596
3. 300194 SALBUTAMOL SIMED SIROP 150 ML.....X 5 150
4. = 5 150 1 545
5. Mnt Total Facture 10.470 Dinars Dont
JSON:
{"items":[{"REFCODE":"303154","MED":"COPRED ODT 20 MG CP ORODISP B/10","QT":1,"prix":5.320},{"REFCODE":"300194","MED":"SALBUTAMOL SIMED SIROP 150 ML","QT":1,"prix":5.150}]}

OCR ROWS:
1. 303616 OMNICEF 125 MG SUSP ORAL 80 ML 2 17.340 34.680
2. PHYSIOMER JET NORMAL 1 23.300 23.300
3. 301034 MUCOLARE 2% ENF FL/125 ML 1 2.875 2.875
4. Total 66.175
JSON:
{"items":[{"REFCODE":"303616","MED":"OMNICEF 125 MG SUSP ORAL 80 ML","QT":2,"prix":34.680},{"REFCODE":null,"MED":"PHYSIOMER JET NORMAL","QT":1,"prix":23.300},{"REFCODE":"301034","MED":"MUCOLARE 2% ENF FL/125 ML","QT":1,"prix":2.875}]}

OCR ROWS:
1. 300883 TERCYD 250MG BT/30 CP 1 63.865 63.865 46.560 70 19.160
2. Total HTVA = 63.865
JSON:
{"items":[{"REFCODE":"300883","MED":"TERCYD 250MG BT/30 CP","QT":1,"prix":63.865}]}
"""

RETRY_SUFFIX = """

Your previous answer was invalid. Return JSON only, no prose, no markdown.
Re-check that every item with a leading product code keeps that REFCODE.
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_RETRY_BASE_DELAY_S = 0.5


def parse_items(content: str) -> list[dict[str, Any]] | None:
    """Pull an items list out of a model response, tolerating fences and prose."""
    text = _FENCE_RE.sub("", content.strip()).strip()

    candidates = [text]
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    bracket = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if bracket:
        candidates.append(bracket.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return None


def rows_text_from_pages(pages: list[OCRPage]) -> str:
    """Numbered row text — the prompt format the few-shot examples establish."""
    lines = [line.text for line in extract_lines(pages)]
    if not lines:
        lines = [line.strip() for page in pages for line in page.text.splitlines() if line.strip()]
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


class LLMMapper:
    """Maps OCR rows to invoice items with a local Ollama model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise LLMMappingError(
                    "The 'ollama' package is not installed; use EXTRACTION_MODE=deterministic "
                    "or install the 'llm' extra."
                ) from exc
            self._client = AsyncClient(
                host=self._settings.ollama_host, timeout=self._settings.ollama_timeout_s
            )
        return self._client

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        client = self._client
        self._client = None
        if client is None:
            return
        inner = getattr(client, "_client", None)
        if inner is not None and hasattr(inner, "aclose"):
            try:
                await inner.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("ignoring error while closing the Ollama client", exc_info=True)

    async def map_pages(
        self,
        pages: list[OCRPage],
        *,
        model: str | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.map_text(
            rows_text_from_pages(pages),
            model=model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            text_label="OCR ROWS",
        )

    async def map_text(
        self,
        ocr_text: str,
        *,
        model: str | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        text_label: str = "OCR TEXT",
    ) -> list[dict[str, Any]]:
        if not ocr_text.strip():
            return []

        settings = self._settings
        client = self._get_client()
        chosen_model = model or settings.ollama_model
        chosen_num_ctx = num_ctx or settings.ollama_num_ctx
        chosen_num_predict = num_predict or settings.ollama_num_predict
        attempts = 1 + settings.ollama_max_retries
        last_error = "unknown"

        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))

            start = time.perf_counter()
            try:
                response = await client.chat(
                    model=chosen_model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT + (RETRY_SUFFIX if attempt else ""),
                        },
                        {"role": "user", "content": f"{text_label}:\n\n{ocr_text}"},
                    ],
                    format="json",
                    options={
                        "temperature": settings.ollama_temperature,
                        "num_predict": chosen_num_predict,
                        "num_ctx": chosen_num_ctx,
                        "repeat_penalty": 1.05,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
                last_error = f"ollama call failed: {exc}"
                logger.warning(
                    "LLM mapping attempt failed",
                    extra={"attempt": attempt + 1, "attempts": attempts, "error": str(exc)},
                )
                continue

            items = parse_items(_response_content(response))
            if items is not None:
                logger.info(
                    "LLM mapping succeeded",
                    extra={
                        "attempt": attempt + 1,
                        "items": len(items),
                        "seconds": round(time.perf_counter() - start, 2),
                        "model": chosen_model,
                        "num_ctx": chosen_num_ctx,
                        "prompt_tokens": _response_field(response, "prompt_eval_count"),
                        "output_tokens": _response_field(response, "eval_count"),
                    },
                )
                return items

            last_error = "model returned unparseable JSON"
            logger.warning(
                "LLM mapping returned unparseable JSON",
                extra={"attempt": attempt + 1, "attempts": attempts},
            )

        metrics.LLM_CALLS.labels(outcome="failed").inc()
        raise LLMMappingError(f"LLM mapping failed after {attempts} attempts: {last_error}")


def _response_content(response: Any) -> str:
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message", {})
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    return content or ""


def _response_field(response: Any, name: str) -> Any:
    value = getattr(response, name, None)
    if value is None and isinstance(response, dict):
        value = response.get(name)
    return value
