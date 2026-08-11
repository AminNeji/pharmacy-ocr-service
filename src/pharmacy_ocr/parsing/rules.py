"""Deterministic parser for Tunisian pharmacy invoice rows.

The OCR model's job is to read characters. This module decides what is an
invoice row — and that separation is the whole design. Row structure,
quantities, unit prices, and invoice arithmetic are deterministic enough that a
3B-parameter LLM should never be the primary source of truth for them.

Two behaviours changed from the previous implementation:

* Line reconstruction is a single linear sweep over y-sorted tokens instead of
  an all-pairs search that recomputed each cluster's mean on every comparison.
  On a dense invoice that was tens of thousands of redundant averages.
* Repeated rows are only collapsed when they are adjacent. Globally deduping
  identical text silently deleted legitimately repeated medicine lines; letting
  them through means the invoice-total checksum decides whether the repeat was
  real or an OCR stutter.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from pharmacy_ocr.domain.models import ParsedInvoice
from pharmacy_ocr.ocr.types import OCRPage, OCRToken
from pharmacy_ocr.parsing import columns, numbers
from pharmacy_ocr.parsing.text import (
    SKIP_MARKERS,
    ascii_fold,
    clean_line,
    has_letters,
    has_phone_marker,
    has_total_marker,
    looks_like_header,
    normalize_key,
)
from pharmacy_ocr.settings import Settings

logger = logging.getLogger(__name__)

#: Names that are only a dosage form or status marker, never a medicine.
GENERIC_MED_ONLY: Final[frozenset[str]] = frozenset(
    {"b", "bt", "cp", "creme", "fl", "forme", "gel", "lot", "pr/lot", "sirop", "susp", "v", "x"}
)

_MIN_MED_ALPHA_CHARS = 3

#: An invoice row is a table row: the line total sits in the rightmost column,
#: so almost nothing follows it. Prose that happens to contain a number — a
#: product leaflet photographed next to the invoice, dosage instructions, a
#: concentration like "0,2%" — has the number embedded mid-sentence with the
#: rest of the sentence after it. Allow a short tail for trailing column
#: markers ("V", "N.I.O", "Dinars") and reject anything wordier.
_MAX_TRAILING_LETTERS = 15

#: Longest plausible medicine designation. Real names run about 20-40
#: characters ("OMNICEF 125 MG SUSP ORAL 80 ML"); anything far past that is a
#: sentence, not a product.
_MAX_MED_LENGTH = 70


@dataclass(frozen=True, slots=True)
class TextLine:
    """One reconstructed row of the invoice.

    The source tokens are retained because their x-coordinates are what make
    column-aware extraction possible; the flattened `text` alone cannot tell a
    product name from the dosage-form column printed beside it.
    """

    text: str
    page_index: int = 0
    top: float = 0.0
    confidence: float | None = None
    tokens: tuple[OCRToken, ...] = ()


@dataclass(frozen=True, slots=True)
class LineVerdict:
    """What the parser decided about one line, and why.

    `skip_reason` is what makes a bad extraction debuggable: it names the rule
    that rejected a row instead of leaving it silently absent from the output.
    """

    text: str
    page_index: int = 0
    confidence: float | None = None
    item: dict[str, Any] | None = None
    warning: str | None = None
    skip_reason: str | None = None

    @property
    def is_item(self) -> bool:
        return self.item is not None


# --------------------------------------------------------------------------- #
# Line reconstruction
# --------------------------------------------------------------------------- #
#: Skew below this is not worth correcting; above it, a wide table row drifts
#: further than the row-grouping threshold and gets split in half.
_MIN_SKEW_DEGREES = 0.3
#: Beyond this, the estimate is picking up something other than page skew.
_MAX_SKEW_DEGREES = 20.0


def estimate_skew(tokens: list[OCRToken]) -> float:
    """Median tilt of the recognized text lines, in degrees.

    Read off the top edge of each detection polygon, which the recognizer
    already gives us — so correcting for skew costs no extra OCR pass.
    """
    angles: list[float] = []
    for token in tokens:
        polygon = token.polygon
        if not polygon or len(polygon) < 2:
            continue
        (x0, y0), (x1, y1) = polygon[0], polygon[1]
        if abs(x1 - x0) < 1.0:
            continue
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if abs(angle) <= _MAX_SKEW_DEGREES:
            angles.append(angle)
    if not angles:
        return 0.0
    skew = statistics.median(angles)
    return skew if abs(skew) >= _MIN_SKEW_DEGREES else 0.0


def cluster_tokens_into_lines(tokens: list[OCRToken]) -> list[TextLine]:
    """Group text boxes into visual rows.

    Grouping is done on the token's vertical position *measured along the
    text's own baseline*, not on raw y. On a hand-held photo the page is
    slightly tilted, so across the width of a table a row can drift further
    than the grouping threshold: the code and product name land in one cluster
    and the prices in another, and neither half parses as an invoice row. One
    corpus invoice lost its only line item exactly this way.

    Tokens are sorted by that corrected position, so a single pass comparing
    each token against the running mean of the open cluster is enough.
    """
    boxed = [token for token in tokens if token.box and token.text.strip()]
    if not boxed:
        return []

    median_height = statistics.median([token.height for token in boxed])
    threshold = max(8.0, median_height * 0.45)

    skew = estimate_skew(boxed)
    slope = math.tan(math.radians(skew)) if skew else 0.0

    def baseline_y(token: OCRToken) -> float:
        """Vertical position with the page tilt taken out."""
        return token.cy - token.cx * slope

    ordered = sorted(boxed, key=lambda token: (token.page_index, baseline_y(token), token.x0))
    clusters: list[list[OCRToken]] = []
    current: list[OCRToken] = []
    current_sum = 0.0

    for token in ordered:
        if current and token.page_index == current[0].page_index:
            centre = current_sum / len(current)
            if abs(baseline_y(token) - centre) <= threshold:
                current.append(token)
                current_sum += baseline_y(token)
                continue
        if current:
            clusters.append(current)
        current = [token]
        current_sum = baseline_y(token)

    if current:
        clusters.append(current)

    return [line for line in (_cluster_to_line(cluster) for cluster in clusters) if line]


def _cluster_to_line(cluster: list[OCRToken]) -> TextLine | None:
    cluster = sorted(cluster, key=lambda token: token.x0)
    median_width = statistics.median([token.width for token in cluster])

    parts: list[str] = []
    previous_x1: float | None = None
    for token in cluster:
        # A gap wider than a third of a typical box is a column separator, not
        # kerning: emit a space so column values do not fuse together.
        if previous_x1 is not None and token.x0 - previous_x1 > median_width * 0.35:
            parts.append(" ")
        parts.append(token.text)
        previous_x1 = token.x1

    text = clean_line("".join(parts))
    if not text:
        return None

    scores = [token.score for token in cluster if token.score is not None]
    return TextLine(
        text=text,
        page_index=cluster[0].page_index,
        top=min(token.y0 for token in cluster),
        confidence=round(statistics.fmean(scores), 4) if scores else None,
        tokens=tuple(cluster),
    )


def _lines_from_raw_text(pages: Iterable[OCRPage]) -> list[TextLine]:
    """Fallback for engines that return text without geometry (PaddleOCR-VL)."""
    lines: list[TextLine] = []
    for page in pages:
        for raw in page.text.splitlines():
            text = clean_line(raw)
            if text:
                lines.append(TextLine(text=text, page_index=page.page_index))
    return lines


def _drop_adjacent_duplicates(lines: list[TextLine]) -> list[TextLine]:
    deduped: list[TextLine] = []
    previous_key: str | None = None
    for line in lines:
        key = normalize_key(line.text)
        if key and key != previous_key:
            deduped.append(line)
        previous_key = key
    return deduped


def _attach_orphan_codes(lines: list[TextLine]) -> list[TextLine]:
    """Reunite a product code with the row it belongs to.

    The code column is narrow and sits far to the left, so the recognizer
    frequently emits it as its own token cluster that never merges with the rest
    of its row. That orphan was the largest single cause of a null REFCODE on
    invoices where the code is plainly legible.

    A code-only line is attached to whichever neighbour looks like a product row
    that is itself missing a code — preferring the following line, since the
    code column is read before the designation.
    """
    merged: list[TextLine] = []
    pending: TextLine | None = None

    for line in lines:
        code = numbers.only_code(line.text)
        if code is not None:
            if pending is not None:
                merged.append(pending)  # two codes in a row: keep the first as-is
            pending = line
            continue

        if pending is not None:
            # Only attach to a row that has prices and no code of its own;
            # anything else (a header, a total) keeps the orphan separate.
            has_own_code, _ = numbers.leading_code(line.text)
            if has_own_code is None and numbers.find_prices(line.text):
                line = TextLine(
                    text=f"{numbers.only_code(pending.text)} {line.text}",
                    page_index=line.page_index,
                    top=line.top,
                    confidence=line.confidence,
                )
            else:
                merged.append(pending)
            pending = None

        merged.append(line)

    if pending is not None:
        merged.append(pending)
    return merged


def extract_lines(pages: list[OCRPage]) -> list[TextLine]:
    """Reconstruct invoice rows from every page."""
    geometric: list[TextLine] = []
    text_only: list[OCRPage] = []
    for page in pages:
        if page.tokens:
            geometric.extend(cluster_tokens_into_lines(page.tokens))
        else:
            text_only.append(page)
    lines = _drop_adjacent_duplicates([*geometric, *_lines_from_raw_text(text_only)])
    return _attach_orphan_codes(lines)


# --------------------------------------------------------------------------- #
# Invoice-level arithmetic
# --------------------------------------------------------------------------- #
def extract_invoice_total(lines: Iterable[TextLine | str]) -> float | None:
    """Find the gross invoice total: the largest value on a total-marked line.

    Taking the maximum is not a heuristic, it follows from what these numbers
    are. A CNAM pharmacy invoice prints the gross alongside its parts — the
    reimbursed share, the patient's share, the VAT component, a pre-tax
    subtotal — and every one of those is by definition no larger than the
    gross. The line items sum to the gross, so that is what the checksum must
    compare against.

    Selecting by marker specificity instead was tried and was worse: on real
    invoices OCR routinely detaches a label like "Total Facture" from its own
    column and re-attaches it to the CNAM share, so the most authoritative-
    looking label can sit next to the wrong number. Magnitude cannot be
    scrambled that way.

    Column headers are excluded because a header row often absorbs the first
    product row's figures during line reconstruction.
    """
    best: float | None = None
    for line in lines:
        text = line if isinstance(line, str) else line.text
        if not has_total_marker(ascii_fold(text)) or looks_like_header(text):
            continue
        for price in numbers.find_prices(text):
            if best is None or price.value > best:
                best = price.value
    return best


def items_total(items: Iterable[dict[str, Any]]) -> float:
    return round(sum(float(item.get("prix") or 0) for item in items), 3)


def checksum_ok(items_sum: float | None, invoice_total: float | None, tolerance: float = 0.05) -> bool | None:
    """True/False when both totals are known, None when the check is impossible."""
    if items_sum is None or invoice_total is None:
        return None
    return numbers.nearly_equal(items_sum, invoice_total, tolerance=tolerance)


def item_quality_issue(refcode: str | None, med: str, prix: float | None) -> str | None:
    """Reason this row should not be trusted as a medicine line, if any."""
    normalized = re.sub(r"\s+", " ", med).strip()
    if sum(1 for char in normalized if char.isalpha()) < _MIN_MED_ALPHA_CHARS:
        return "medicine name is too short to be reliable"
    if normalized.casefold().strip(" .:-") in GENERIC_MED_ONLY:
        return "medicine name is only a form/status marker"
    if prix is None or prix <= 0:
        return "missing non-zero line total"
    return None


# --------------------------------------------------------------------------- #
# Row parser
# --------------------------------------------------------------------------- #
class RowParser:
    """Turns reconstructed lines into candidate invoice items."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def parse(self, pages: list[OCRPage]) -> ParsedInvoice:
        lines = extract_lines(pages)
        items: list[dict[str, Any]] = []
        warnings: list[str] = []

        # Column geometry, when the invoice prints a header. Without it the
        # parser falls back to the regex path, which several corpus invoices
        # (free-form `CODE NAME Qte x unit = total`) rely on.
        layout = columns.detect_layout(lines)

        for line in lines:
            item, warning = self.parse_line(line, layout=layout)
            if item is not None:
                items.append(item)
            if warning:
                warnings.append(warning)

        items = _drop_adjacent_duplicate_items(items)
        invoice_total = extract_invoice_total(lines)
        if not items:
            warnings.append("deterministic parser found no invoice rows")

        return ParsedInvoice(
            items=items,
            warnings=warnings,
            lines=[line.text for line in lines],
            invoice_total=invoice_total,
        )

    def parse_line(
        self, line: TextLine | str, *, layout: columns.ColumnLayout | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        verdict = self.explain_line(line, layout=layout)
        return verdict.item, verdict.warning

    def explain_line(
        self, line: TextLine | str, *, layout: columns.ColumnLayout | None = None
    ) -> LineVerdict:
        """Parse one line and record *why* it was or was not an invoice row.

        `parse_line` is the hot path and only needs the outcome; this variant
        keeps the reason so `pharmacy-ocr diagnose` can show a human which rule
        rejected a row that should have been extracted.
        """
        source = line if isinstance(line, TextLine) else TextLine(text=line)
        text = clean_line(source.text)

        def verdict(
            *,
            item: dict[str, Any] | None = None,
            warning: str | None = None,
            skip_reason: str | None = None,
        ) -> LineVerdict:
            return LineVerdict(
                text=text,
                page_index=source.page_index,
                confidence=source.confidence,
                item=item,
                warning=warning,
                skip_reason=skip_reason,
            )

        code, rest = numbers.leading_code(text)
        prices = numbers.find_prices(rest)

        skip_reason = self._non_item_reason(text, has_code=code is not None)
        if skip_reason:
            return verdict(skip_reason=skip_reason)
        if not prices:
            return verdict(skip_reason="no price found on the line")

        trailing_letters = sum(1 for char in rest[prices[-1].end :] if char.isalpha())
        if trailing_letters > _MAX_TRAILING_LETTERS:
            return verdict(
                skip_reason=f"prose, not a table row ({trailing_letters} letters follow the last price)"
            )

        provisional_qty = numbers.explicit_qty(rest[: prices[0].start])
        total_price = numbers.choose_total_price(rest, prices, provisional_qty)
        quantity = numbers.infer_qty(rest, prices, total_price)

        # Values as read from the flattened text, before the table geometry
        # gets a say.
        final_total = round(total_price.value, 3) if total_price else None
        final_unit = round(prices[0].value, 3) if len(prices) >= 2 and prices[0] is not total_price else None

        # The line-total column, when the table declares one, beats every
        # positional heuristic: it is stated rather than inferred, and it is
        # immune to two adjacent figures being recognized as one run
        # ("16.2380.00" -> a nonsense 2380).
        column_total = self._column_price(source, layout, columns.ColumnRole.line_total)
        if column_total is not None:
            final_total = column_total
            final_unit = self._column_price(source, layout, columns.ColumnRole.unit_price)
            column_qty = self._column_quantity(source, layout)
            if column_qty is not None:
                quantity = column_qty

        # Always derive the text-based name too: it is what reveals prose. A
        # column cell is bounded by the table's width, so a sentence read out
        # of one looks like a short, plausible product name.
        text_name = self._medicine_name(rest, prices[0])
        if len(text_name) > _MAX_MED_LENGTH:
            return verdict(
                skip_reason=f"row text is {len(text_name)} chars before the price, reads as a sentence"
            )

        med = self._column_name(source, layout) or text_name

        if not med:
            return verdict(skip_reason="nothing left as a medicine name before the price")
        if not has_letters(med):
            return verdict(skip_reason=f"medicine name has no letters ({med!r})")
        if len(med) > _MAX_MED_LENGTH:
            return verdict(skip_reason=f"medicine name is {len(med)} chars long, reads as a sentence")

        med_folded = ascii_fold(med)
        if has_total_marker(med_folded):
            return verdict(skip_reason="name reads as a summary/total label")
        if has_phone_marker(med_folded) or "pharmacie" in med_folded:
            return verdict(skip_reason="name reads as pharmacy contact details")

        # A row that carries a printed product code and a total is a product
        # row; an absent quantity on such a row means one unit.
        if quantity is None and code is not None and final_total is not None:
            quantity = 1

        item: dict[str, Any] = {
            "REFCODE": code,
            "MED": med,
            "QT": quantity,
            "prix": final_total,
            "unit_price": final_unit,
            "confidence": source.confidence,
        }
        warning = None if quantity is not None else f"quantity could not be determined for: {text}"
        return verdict(item=item, warning=warning)

    @staticmethod
    def _non_item_reason(line: str, *, has_code: bool) -> str | None:
        folded = ascii_fold(line)
        if looks_like_header(folded):
            return "column header"
        if has_phone_marker(folded):
            return "phone/fax line"
        if "matricule" in folded:
            return "patient identifier line"
        if has_code:
            # A printed product code outweighs every soft skip marker; invoices
            # legitimately contain words like "date" inside a medicine name.
            return None
        marker = next((marker for marker in SKIP_MARKERS if marker in folded), None)
        if marker:
            return f"document furniture (matched {marker!r})"
        if has_total_marker(folded):
            return "summary/total line"
        return None

    @staticmethod
    def _row_uses_layout(line: TextLine, layout: columns.ColumnLayout | None) -> bool:
        """Whether this line is a data row of the table `layout` describes."""
        if layout is None or not line.tokens:
            return False
        return line.page_index == layout.page_index and line.top >= layout.header_top

    @classmethod
    def _column_price(
        cls, line: TextLine, layout: columns.ColumnLayout | None, role: columns.ColumnRole
    ) -> float | None:
        """The money value printed in one column of this row.

        Normally the cell holds one figure and there is nothing to decide. The
        exception is the last declared column, whose band is open-ended: if the
        recognizer fused two headings into one token, that band also covers an
        unnamed column to its right, and the value we want is the rightmost.
        Anywhere else the band is closed on both sides, so a second figure
        means a neighbour has leaked in and the first one is ours.
        """
        if not cls._row_uses_layout(line, layout):
            return None
        assert layout is not None
        column = layout.role(role)
        if column is None:
            return None
        found = numbers.find_prices(columns.cell_text(line, column))
        if not found:
            return None
        chosen = found[-1] if layout.is_rightmost(column) else found[0]
        return round(chosen.value, 3)

    @classmethod
    def _column_quantity(cls, line: TextLine, layout: columns.ColumnLayout | None) -> int | None:
        if not cls._row_uses_layout(line, layout):
            return None
        assert layout is not None
        column = layout.role(columns.ColumnRole.quantity)
        if column is None:
            return None
        cell = columns.cell_text(line, column).strip()
        if not cell.isdigit():
            return None
        quantity = int(cell)
        return quantity if 0 <= quantity <= numbers.MAX_QUANTITY else None

    @staticmethod
    def _column_name(line: TextLine, layout: columns.ColumnLayout | None) -> str | None:
        """Read the designation straight out of its column, when we have one.

        This is what stops the neighbouring dosage-form column from being glued
        onto the product name. Returns None when there is no layout, when the
        row sits above the header, or when the cell is empty — the caller then
        falls back to the text-based rules.
        """
        if layout is None or not line.tokens:
            return None
        if line.page_index != layout.page_index or line.top < layout.header_top:
            return None
        name_column = layout.role(columns.ColumnRole.name)
        if name_column is None:
            return None

        # Trust the geometry only for rows that actually populate a money
        # column. Prose printed beside the table spans the page and would
        # otherwise be sliced into a plausible-looking short "name".
        money = [layout.role(role) for role in (columns.ColumnRole.line_total, columns.ColumnRole.unit_price)]
        has_money_cell = any(
            column is not None and numbers.find_prices(columns.cell_text(line, column)) for column in money
        )
        if not has_money_cell:
            return None

        cell = columns.cell_text(line, name_column)
        # Header extents are narrower than their data, so a band can reach into
        # the code column. A code at the head of the name cell is never part of
        # the product name.
        code, remainder = numbers.leading_code(cell)
        if code:
            cell = remainder
        cell = re.sub(r"\s+", " ", cell).strip(" .:-")
        # A cell holding only digits is the code column bleeding leftwards, not
        # a product; let the text rules handle those rows.
        return cell if cell and has_letters(cell) else None

    @staticmethod
    def _medicine_name(rest: str, first_price: numbers.PriceMatch | None) -> str:
        segment = rest[: first_price.start] if first_price else rest
        segment = re.sub(r"\bN[°o]?\s*Ord\w*\b", " ", segment, flags=re.IGNORECASE)
        segment = re.sub(r"\bTotal\b.*$", " ", segment, flags=re.IGNORECASE)
        segment = re.sub(r"\bHTVA\b.*$", " ", segment, flags=re.IGNORECASE)
        segment = numbers.strip_quantity_tail(segment)
        # Recover the "B/30" pack notation from common OCR misreads.
        segment = re.sub(r"\bB1(\d{1,3})\b", r"B/\1", segment, flags=re.IGNORECASE)
        segment = re.sub(r"\bB\s*[/I1]\s*(\d{1,3})\b", r"B/\1", segment, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", segment).strip(" .:-")


def _drop_adjacent_duplicate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    previous_key: tuple[Any, ...] | None = None
    for item in items:
        key = (
            item.get("REFCODE"),
            re.sub(r"\s+", " ", str(item.get("MED", ""))).strip().upper(),
            item.get("QT"),
            item.get("prix"),
        )
        if key != previous_key:
            deduped.append(item)
        previous_key = key
    return deduped
