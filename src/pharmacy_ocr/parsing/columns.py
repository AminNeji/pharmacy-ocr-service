"""Column-aware table reconstruction.

Flattening a table row to a string throws away the one thing that says where a
product name ends: the x-coordinate. Every remaining medicine-name error in the
reference corpus came from that loss — the dosage-form column printed beside the
designation got appended to it, turning "ECOREX 1% CREME 30GR" into
"ECOREX 1% CREME 30GR CREME".

This module recovers the geometry. It locates the header row, converts each
heading into a semantic role, and turns the headings' x-extents into column
bands that later rows are cut against.

Roles are semantic, not literal. Suppliers label the same column
"Produit", "Libellé", "Désignation", "Médicament", or "Article", and the price
column may be "PU", "PUV", "Prix Unit.", "Montant", or "Mt". Matching is done on
a folded, punctuation-free form of the heading against a synonym table, so a new
supplier usually needs a synonym rather than new logic.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pharmacy_ocr.ocr.types import OCRToken
from pharmacy_ocr.parsing.text import ascii_fold

logger = logging.getLogger(__name__)


class ColumnRole(StrEnum):
    """What a table column means, independent of how it is labelled."""

    code = "code"
    name = "name"
    form = "form"
    quantity = "quantity"
    unit_price = "unit_price"
    line_total = "line_total"
    reference_price = "reference_price"
    rate = "rate"
    refund = "refund"
    other = "other"


#: Folded heading text -> role. Longer keys are matched first so that
#: "prix unit" wins over "prix" and "code pct" over "code".
_HEADING_SYNONYMS: dict[str, ColumnRole] = {
    # identifiers
    "code pct": ColumnRole.code,
    "codepct": ColumnRole.code,
    "pct nom": ColumnRole.code,
    "reference": ColumnRole.code,
    "ref": ColumnRole.code,
    "refcode": ColumnRole.code,
    "code": ColumnRole.code,
    "cip": ColumnRole.code,
    "barcode": ColumnRole.code,
    "pct": ColumnRole.code,
    # designation
    "nom de specialite": ColumnRole.name,
    "nom de specialit": ColumnRole.name,
    "specialite": ColumnRole.name,
    "designation": ColumnRole.name,
    "desiqnation": ColumnRole.name,
    "libelle": ColumnRole.name,
    "medicament": ColumnRole.name,
    "produit": ColumnRole.name,
    "article": ColumnRole.name,
    "nom": ColumnRole.name,
    # dosage form: its own column, and the one that kept polluting names
    "forme": ColumnRole.form,
    "form": ColumnRole.form,
    # quantity
    "quantite": ColumnRole.quantity,
    "qte": ColumnRole.quantity,
    "qty": ColumnRole.quantity,
    "qt": ColumnRole.quantity,
    "q": ColumnRole.quantity,
    # money
    "prix unit": ColumnRole.unit_price,
    "prix u": ColumnRole.unit_price,
    "prix pub": ColumnRole.unit_price,
    "prix": ColumnRole.unit_price,
    "puv": ColumnRole.unit_price,
    "pu": ColumnRole.unit_price,
    "tot ttc": ColumnRole.line_total,
    "tot fact": ColumnRole.line_total,
    "total ttc": ColumnRole.line_total,
    "total ligne": ColumnRole.line_total,
    "total": ColumnRole.line_total,
    "montant": ColumnRole.line_total,
    "mt percu": ColumnRole.line_total,
    "mtpercu": ColumnRole.line_total,
    "percu": ColumnRole.refund,
    "mt": ColumnRole.line_total,
    # reimbursement detail
    "t ref": ColumnRole.reference_price,
    "tref": ColumnRole.reference_price,
    "tx": ColumnRole.rate,
    "n i o": ColumnRole.other,
    "nio": ColumnRole.other,
    "pr lot": ColumnRole.other,
    "prlot": ColumnRole.other,
}

#: A header must supply at least this many recognised roles before its geometry
#: is trusted; two stray words that happen to match are not a table header.
MIN_HEADER_ROLES = 3

#: Header rows appear above the products, never at the very bottom.
MAX_HEADER_LINE_FRACTION = 0.6

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", ascii_fold(text))).strip()


#: Headings are short, so one wrong character is a large proportional error;
#: 0.82 accepts "Artides" for "Articles" and "Desiqnation" for "Designation"
#: without pulling in genuinely different words.
_FUZZY_HEADING_CUTOFF = 0.82
#: Too short to match fuzzily without collisions ("Tx" vs "Tva").
_MIN_FUZZY_LENGTH = 5


def role_for_heading(text: str) -> ColumnRole | None:
    """Map one heading cell to a role, longest synonym first.

    Falls back to approximate matching because headings are printed small and
    are among the first things the recognizer gets wrong: one corpus invoice
    reads "Artides" for "Articles", which lost the whole table layout and with
    it the product column.
    """
    folded = normalize_heading(text)
    if not folded:
        return None
    for synonym in sorted(_HEADING_SYNONYMS, key=len, reverse=True):
        if synonym in folded:
            return _HEADING_SYNONYMS[synonym]

    if len(folded) >= _MIN_FUZZY_LENGTH:
        for word in folded.split():
            if len(word) < _MIN_FUZZY_LENGTH:
                continue
            candidates = [key for key in _HEADING_SYNONYMS if len(key) >= _MIN_FUZZY_LENGTH]
            close = difflib.get_close_matches(word, candidates, n=1, cutoff=_FUZZY_HEADING_CUTOFF)
            if close:
                return _HEADING_SYNONYMS[close[0]]
    return None


@dataclass(frozen=True, slots=True)
class Column:
    role: ColumnRole
    label: str
    x0: float
    x1: float

    @property
    def centre(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """The table's column bands, derived from its header row."""

    columns: tuple[Column, ...]
    header_top: float
    page_index: int

    def role(self, role: ColumnRole) -> Column | None:
        for column in self.columns:
            if column.role is role:
                return column
        return None

    def is_rightmost(self, column: Column) -> bool:
        """Whether `column` is the last one the header declared.

        Only the last band is open-ended, so only there can a cell legitimately
        pick up figures belonging to a column the header never named.
        """
        return bool(self.columns) and self.columns[-1] is column

    def column_at(self, x: float) -> Column | None:
        """The column whose band contains `x`, else the nearest by centre."""
        for column in self.columns:
            if column.x0 <= x <= column.x1:
                return column
        if not self.columns:
            return None
        return min(self.columns, key=lambda column: abs(column.centre - x))


def _bands(entries: list[tuple[ColumnRole, str, float, float]]) -> tuple[Column, ...]:
    """Widen each heading's extent to meet its neighbours.

    A heading is usually narrower than the data beneath it ("Qté" over "1"), so
    the raw heading box would leave most cells outside every band. Splitting the
    gap between adjacent headings gives bands that tile the row.
    """
    entries = sorted(entries, key=lambda item: item[2])
    columns: list[Column] = []
    for index, (role, label, x0, x1) in enumerate(entries):
        # The outermost bands are open-ended. A heading is narrower than its
        # data, and the recognizer sometimes fuses two headings into one token
        # ("Tva Total"), leaving the rightmost real column unlabelled - its
        # values then fell outside every band and were lost.
        left = float("-inf") if index == 0 else (entries[index - 1][3] + x0) / 2.0
        right = float("inf") if index == len(entries) - 1 else (x1 + entries[index + 1][2]) / 2.0
        columns.append(Column(role=role, label=label, x0=left, x1=right))
    return tuple(columns)


def _group_heading_tokens(tokens: list[OCRToken]) -> list[tuple[ColumnRole, str, float, float]]:
    """Turn a header line's tokens into (role, label, x0, x1) entries.

    The shortest run that resolves wins. Preferring longer runs merged headings
    that are each a column in their own right: "Code PCT | Produit | Forme"
    matched the synonym "code pct" across all three tokens and swallowed the
    product column, which is the one the name is read from.

    Longer runs are still tried when no single token resolves, because some
    headings genuinely span tokens — "Tot." + "TTC" means nothing apart.
    """
    entries: list[tuple[ColumnRole, str, float, float]] = []
    index = 0
    while index < len(tokens):
        best: tuple[ColumnRole, str, float, float] | None = None
        best_span = 0
        for span in (1, 2, 3):
            if index + span > len(tokens):
                continue
            # Never let a longer span swallow a heading that stands on its own.
            # "Artides" does not resolve, but merging it into the neighbouring
            # "Oté Prix Ht" produced a single price column spanning both and
            # erased the designation column. Extend only when the next token
            # cannot be a heading by itself.
            if span > 1 and role_for_heading(tokens[index + 1].text) is not None:
                break
            group = tokens[index : index + span]
            label = " ".join(token.text for token in group)
            role = role_for_heading(label)
            if role is not None:
                best = (role, label, group[0].x0, group[-1].x1)
                best_span = span
                break
        if best is not None:
            entries.append(best)
            index += best_span
        else:
            # Keep the geometry of headings we cannot name. They still bound
            # their neighbours, and one of them may be the designation column
            # under a misread label - see `_promote_name_column`.
            token = tokens[index]
            entries.append((ColumnRole.other, token.text, token.x0, token.x1))
            index += 1
    return entries


def _promote_name_column(
    entries: list[tuple[ColumnRole, str, float, float]],
) -> list[tuple[ColumnRole, str, float, float]]:
    """Adopt an unnamed column as the designation when none was recognised.

    Header text is small and is among the first things the recognizer garbles:
    one corpus invoice prints "Articles" but reads as "Artides", which is too
    far off to match safely by edit distance. The structure still gives it
    away — a table that declares a code, a quantity and a price is a product
    table, and the widest unnamed column sitting between the code and the
    numbers is its designation.
    """
    if any(role is ColumnRole.name for role, _, _, _ in entries):
        return entries

    ordered = sorted(entries, key=lambda item: item[2])
    code_right = max((x1 for role, _, _, x1 in ordered if role is ColumnRole.code), default=float("-inf"))
    numeric_left = min(
        (
            x0
            for role, _, x0, _ in ordered
            if role in (ColumnRole.quantity, ColumnRole.unit_price, ColumnRole.line_total)
        ),
        default=float("inf"),
    )

    candidates = [
        entry
        for entry in ordered
        if entry[0] is ColumnRole.other and entry[2] >= code_right and entry[3] <= numeric_left
    ]
    if not candidates:
        return entries

    widest = max(candidates, key=lambda entry: entry[3] - entry[2])
    logger.info("adopted unnamed column as the designation", extra={"label": widest[1]})
    return [
        (ColumnRole.name, entry[1], entry[2], entry[3]) if entry is widest else entry for entry in ordered
    ]


def detect_layout(lines: list[Any]) -> ColumnLayout | None:
    """Find the table header and derive column bands from it.

    Returns None when the invoice prints no header at all, which is common —
    several corpus invoices use a free-form `CODE NAME Qte x unit = total`
    layout. Callers must keep working without a layout.
    """
    limit = max(1, int(len(lines) * MAX_HEADER_LINE_FRACTION))
    for line in lines[:limit]:
        tokens = [token for token in getattr(line, "tokens", ()) if token.box]
        if len(tokens) < MIN_HEADER_ROLES:
            continue
        entries = _group_heading_tokens(sorted(tokens, key=lambda t: t.x0))
        # Only recognised roles count towards "is this a header"; the unnamed
        # placeholders carry geometry, not evidence.
        roles = {entry[0] for entry in entries if entry[0] is not ColumnRole.other}
        if len(roles) < MIN_HEADER_ROLES:
            continue

        entries = _promote_name_column(entries)
        if not any(entry[0] is ColumnRole.name for entry in entries):
            continue

        layout = ColumnLayout(
            columns=_bands(entries),
            header_top=line.top,
            page_index=line.page_index,
        )
        logger.info(
            "table header detected",
            extra={
                "columns": [f"{c.label}->{c.role.value}" for c in layout.columns],
                "page": line.page_index,
            },
        )
        return layout
    return None


def cell_text(line: Any, column: Column) -> str:
    """The text of `line` that falls inside `column`'s band."""
    parts = [
        token.text
        for token in sorted(getattr(line, "tokens", ()), key=lambda t: t.x0)
        if token.box and column.x0 <= token.cx <= column.x1
    ]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()
