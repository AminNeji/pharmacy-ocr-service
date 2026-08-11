"""Text normalization and the marker vocabulary of Tunisian pharmacy invoices.

Kept separate from the row rules so the vocabulary can be reviewed and extended
by someone who knows the paperwork without touching parsing logic.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"[•·]+")
_LETTER_RE = re.compile(r"[A-Za-z]")


def ascii_fold(text: str) -> str:
    """Lowercase and strip accents so 'Désignation' matches 'designation'."""
    folded = unicodedata.normalize("NFKD", text)
    return "".join(char for char in folded if not unicodedata.combining(char)).lower()


def clean_line(text: str) -> str:
    """Collapse whitespace and drop separator glyphs the OCR emits for rules."""
    # `\s` matches U+00A0, so the collapse below also normalizes the
    # non-breaking spaces OCR emits inside amounts like "5 320".
    text = text.replace("|", " ")
    text = _BULLET_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def has_letters(text: str) -> bool:
    return bool(_LETTER_RE.search(text))


def normalize_key(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


# --------------------------------------------------------------------------- #
# Marker vocabulary (all entries must be ascii-folded lowercase)
# --------------------------------------------------------------------------- #
#: Column headings. A line containing one of these describes the table, it is
#: never a product row.
HEADER_MARKERS: Final[tuple[str, ...]] = (
    "pct nom",
    "nom de specialite",
    "code pct",
    "designation",
    "desiqnation",  # frequent OCR misread of "designation"
    "qte",
    "prix unit",
    "prix pub",
    "puv",
    "tot ttc",
    "tot fact",
    "n ord",
)

#: Document furniture: addresses, identifiers, stamps. Only skipped when the
#: line does not start with a product code.
SKIP_MARKERS: Final[tuple[str, ...]] = (
    "pharmacie",
    "date",
    "patient",
    "assure",
    "matricule",
    "identif",
    "facture",
    "ordonnance",
    "arreter",
    "arret",
    "tva",
    "tel",
    "cam scanner",
    "camscanner",
    "cachet",
    "signature",
    "rib",
    "adresse",
)

#: Any summary line. Never a product row.
TOTAL_MARKERS: Final[tuple[str, ...]] = (
    "total",
    "mnt total",
    "montant total",
    "mnt percu",
    "percu assure",
    "cnam",
    "timbre",
    "remise",
    "net a payer",
)

_PHONE_RE = re.compile(r"\bt.?l\b")

#: "total" as the recognizer actually renders it. A dropped or substituted final
#: letter ("Tota!", "Tota1", "T0tal") made the literal match fail, and the
#: summary line was then extracted as a medicine called "Tota!".
_TOTAL_FUZZY_RE = re.compile(r"t[o0]t[a4][l1i!|]")


def looks_like_header(line: str) -> bool:
    lower = ascii_fold(line)
    return any(marker in lower for marker in HEADER_MARKERS)


def has_phone_marker(folded: str) -> bool:
    return bool(_PHONE_RE.search(folded)) or "fax" in folded or "f1x" in folded


def has_total_marker(folded: str) -> bool:
    return any(marker in folded for marker in TOTAL_MARKERS) or bool(_TOTAL_FUZZY_RE.search(folded))
