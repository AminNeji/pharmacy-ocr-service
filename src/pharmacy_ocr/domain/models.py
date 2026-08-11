"""Pydantic models describing invoices and extraction results.

Field names `REFCODE`, `MED`, `QT`, and `prix` are part of the public output
contract and are kept exactly as they were in earlier versions. Everything else
is additive, so existing consumers keep working.

Enforced rules:
- REFCODE has spaces and dashes removed; it becomes null when the cleaned value
  is shorter than `refcode_min_length`, which filters OCR fragments.
- QT is always an integer. A decimal found in QT is a price, not a quantity, so
  it is swapped into `prix`.
- prix is the LINE TOTAL in dinars. French decimal commas ("12,500") are
  accepted on input.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Default kept as a module constant so model validation stays independent of
#: the settings singleton; the pipeline overrides it via `clean_refcode`.
DEFAULT_REFCODE_MIN_LENGTH: Final[int] = 4

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_NULLISH = frozenset({"", "null", "none", "n/a", "na", "-", "--"})

#: Accepts whatever key casing/language an LLM or a parser produced.
#: Grouped by target field on purpose — one key per line would make it much
#: harder to see which spellings map where.
# fmt: off
_FIELD_ALIASES: Final[dict[str, str]] = {
    "refcode": "REFCODE", "ref": "REFCODE", "code": "REFCODE",
    "codepct": "REFCODE", "pct": "REFCODE", "pctnom": "REFCODE",
    "med": "MED", "medicament": "MED", "medicine": "MED",
    "designation": "MED", "libelle": "MED", "produit": "MED", "nom": "MED",
    "qt": "QT", "qte": "QT", "quantite": "QT", "quantity": "QT", "qty": "QT",
    "prix": "prix", "price": "prix", "montant": "prix", "total": "prix",
    "linetotal": "prix", "prixtotal": "prix", "totalttc": "prix",
    "pu": "unit_price", "prixunitaire": "unit_price", "unitprice": "unit_price",
    "prixunit": "unit_price", "puv": "unit_price",
    "confidence": "confidence", "score": "confidence",
}
# fmt: on


class ReviewStatus(StrEnum):
    """Single field downstream systems can route on.

    ok
        Items were extracted and the arithmetic checksum passed.
    needs_review
        Something is uncertain — no printed total to check against, a failed
        checksum, or dropped rows. A human should look before this is booked.
    failed
        No result at all; `error` explains why.
    """

    ok = "ok"
    needs_review = "needs_review"
    failed = "failed"


def to_number(value: Any) -> float | None:
    """Coerce int, float, '12,500', '3 x', or None into a float or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Strip every whitespace kind first: OCR writes "5 320" and
        # "5\u00a0320" interchangeably for 5.320.
        match = _NUMBER_RE.search(re.sub(r"\s+", "", value))
        if match:
            return float(match.group(0).replace(",", "."))
    return None


def is_integral(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def clean_refcode(value: Any, min_length: int = DEFAULT_REFCODE_MIN_LENGTH) -> str | None:
    """Strip spaces and dashes; drop codes too short to be trustworthy."""
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in _NULLISH:
        return None
    cleaned = re.sub(r"[\s\-]+", "", text)
    if len(cleaned) < min_length:
        return None
    return cleaned


def _normalize_key(key: Any) -> str:
    return re.sub(r"[\s_./-]+", "", str(key)).casefold()


class InvoiceItem(BaseModel):
    """One medicine line on the invoice."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    REFCODE: str | None = Field(default=None, description="Printed product/PCT code.")
    MED: str = Field(min_length=1, description="Medicine name as printed on the row.")
    QT: int | None = Field(default=None, ge=0, description="Quantity; always an integer.")
    prix: float | None = Field(default=None, ge=0, description="Line total in dinars.")
    unit_price: float | None = Field(
        default=None, ge=0, description="Unit price when the invoice prints one."
    )
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Mean OCR confidence for the source row."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = _FIELD_ALIASES.get(_normalize_key(key))
            if canonical and canonical not in normalized:
                normalized[canonical] = value

        qt = to_number(normalized.get("QT"))
        prix = to_number(normalized.get("prix"))
        unit_price = to_number(normalized.get("unit_price"))

        # A decimal quantity is always a misread price.
        if qt is not None and not is_integral(qt):
            if prix is None:
                qt, prix = None, qt
            elif is_integral(prix):
                qt, prix = prix, qt
            else:
                qt = None  # both decimal: trust the declared prix, drop bad QT

        # Some mappers only emit a unit price. Deriving the line total by
        # arithmetic is safe — the invoice-level checksum still validates it.
        if prix is None and unit_price is not None:
            prix = unit_price * qt if qt else unit_price

        normalized["QT"] = int(round(qt)) if qt is not None and qt >= 0 else None
        normalized["prix"] = round(prix, 3) if prix is not None and prix >= 0 else None
        normalized["unit_price"] = (
            round(unit_price, 3) if unit_price is not None and unit_price >= 0 else None
        )
        return normalized

    @field_validator("REFCODE", mode="before")
    @classmethod
    def _clean_refcode(cls, value: Any) -> str | None:
        return clean_refcode(value)

    @field_validator("MED", mode="before")
    @classmethod
    def _clean_med(cls, value: Any) -> str:
        if value is None:
            raise ValueError("MED is required")
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text:
            raise ValueError("MED must not be empty")
        return text

    @property
    def line_total_matches_unit(self) -> bool | None:
        """True when QT x unit_price reproduces prix, None when unknowable."""
        if self.prix is None or self.unit_price is None or self.QT is None:
            return None
        return abs(self.unit_price * self.QT - self.prix) <= 0.01


class ParsedInvoice(BaseModel):
    """Intermediate output of the deterministic parser."""

    model_config = ConfigDict(frozen=True)

    items: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    invoice_total: float | None = None


class InvoiceResult(BaseModel):
    """Full pipeline output for one invoice.

    The pipeline never raises past this point: failures are reported through
    `error` and `warnings` so a batch run survives a bad invoice.
    """

    source_file: str
    status: ReviewStatus = ReviewStatus.needs_review
    items: list[InvoiceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    invoice_total: float | None = Field(default=None, description="Total printed on the invoice.")
    items_total: float | None = Field(default=None, description="Sum of extracted line totals.")
    checksum_ok: bool | None = Field(
        default=None, description="Whether items_total matches invoice_total; null if unknown."
    )

    page_count: int = 0
    extraction_mode: str | None = None
    ocr_engine: str | None = None
    service_version: str | None = None
    processed_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    timings: dict[str, float] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def needs_review(self) -> bool:
        return self.status is not ReviewStatus.ok

    @staticmethod
    def derive_status(
        *, error: str | None, items: list[InvoiceItem], checksum_ok: bool | None, warnings: list[str]
    ) -> ReviewStatus:
        if error is not None:
            return ReviewStatus.failed
        if not items or checksum_ok is not True or warnings:
            return ReviewStatus.needs_review
        return ReviewStatus.ok


class ExtractionEnvelope(BaseModel):
    """Schema the LLM is asked to produce."""

    items: list[dict[str, Any]] = Field(default_factory=list)
