"""Column detection: semantic heading roles and cell extraction."""

from __future__ import annotations

import pytest

from pharmacy_ocr.parsing.columns import (
    ColumnRole,
    detect_layout,
    role_for_heading,
)
from pharmacy_ocr.parsing.rules import RowParser, TextLine, extract_lines
from pharmacy_ocr.settings import Settings
from tests.conftest import make_token


# --------------------------------------------------------------------------- #
# Semantic heading -> role
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("heading", "role"),
    [
        # The brief lists these as interchangeable across suppliers.
        ("Reference", ColumnRole.code),
        ("Référence", ColumnRole.code),
        ("Réf", ColumnRole.code),
        ("Code", ColumnRole.code),
        ("Code PCT", ColumnRole.code),
        ("CIP", ColumnRole.code),
        ("Produit", ColumnRole.name),
        ("Libellé", ColumnRole.name),
        ("Médicament", ColumnRole.name),
        ("Designation", ColumnRole.name),
        ("Nom de spécialité", ColumnRole.name),
        ("QTE", ColumnRole.quantity),
        ("Qté", ColumnRole.quantity),
        ("Qty", ColumnRole.quantity),
        ("Quantité", ColumnRole.quantity),
        ("PU", ColumnRole.unit_price),
        ("PUV", ColumnRole.unit_price),
        ("Prix Unit.", ColumnRole.unit_price),
        ("Montant", ColumnRole.line_total),
        ("Total Ligne", ColumnRole.line_total),
        ("Tot. TTC", ColumnRole.line_total),
        ("Forme", ColumnRole.form),
    ],
)
def test_heading_maps_to_semantic_role(heading: str, role: ColumnRole) -> None:
    assert role_for_heading(heading) is role


def test_unknown_heading_has_no_role() -> None:
    assert role_for_heading("Observations") is None
    assert role_for_heading("") is None


# --------------------------------------------------------------------------- #
# Layout detection
# --------------------------------------------------------------------------- #
def _header_line(headings: list[str], *, top: float = 100.0) -> TextLine:
    tokens = tuple(
        make_token(text, 40 + 200 * index, top, width=150, height=18) for index, text in enumerate(headings)
    )
    return TextLine(text=" ".join(headings), page_index=0, top=top, confidence=0.95, tokens=tokens)


def test_detects_the_cnam_header() -> None:
    header = _header_line(["Code PCT", "Produit", "Forme", "Qte", "PUV", "Mt. Percu"])
    layout = detect_layout([header])

    assert layout is not None
    assert [column.role for column in layout.columns] == [
        ColumnRole.code,
        ColumnRole.name,
        ColumnRole.form,
        ColumnRole.quantity,
        ColumnRole.unit_price,
        ColumnRole.line_total,
    ]


def test_multi_token_heading_is_grouped_only_when_needed() -> None:
    """`Code PCT | Produit | Forme` are three columns, not one.

    Preferring the longest matching run merged them under the synonym
    "code pct" and lost the product column entirely.
    """
    layout = detect_layout([_header_line(["Code PCT", "Produit", "Forme", "Qte"])])
    assert layout is not None
    assert layout.role(ColumnRole.name) is not None
    assert layout.role(ColumnRole.form) is not None


def test_header_needs_a_name_column() -> None:
    assert detect_layout([_header_line(["Qte", "PUV", "Tx"])]) is None


def test_two_matching_words_are_not_a_header() -> None:
    assert detect_layout([_header_line(["Code", "Produit"])]) is None


def test_no_header_returns_none() -> None:
    """Free-form invoices print no header; the parser must still work."""
    line = TextLine(text="102474 TIORFAN 30 SA 1 x 22.440 = 22.440", tokens=())
    assert detect_layout([line]) is None


# --------------------------------------------------------------------------- #
# End-to-end: the dosage-form column must not pollute the name
# --------------------------------------------------------------------------- #
def test_form_column_is_not_appended_to_the_medicine_name() -> None:
    """Regression: 'ECOREX 1% CREME 30GR' came out as '... 30GR CREME'."""
    from pharmacy_ocr.ocr.types import OCRPage

    def row(values: list[str], top: float) -> list:
        return [
            make_token(text, 40 + 200 * index, top, width=150, height=18) for index, text in enumerate(values)
        ]

    tokens = [
        *row(["Code PCT", "Produit", "Forme", "Qte", "PUV", "Mt. Percu"], 100),
        *row(["307540", "ECOREX 1% CREME 30GR", "CREME", "1", "5.395", "5.395"], 140),
    ]
    parsed = RowParser(Settings(environment="development")).parse([OCRPage(text="", tokens=tokens)])

    assert len(parsed.items) == 1
    assert parsed.items[0]["MED"] == "ECOREX 1% CREME 30GR"
    assert parsed.items[0]["REFCODE"] == "307540"


def test_skewed_row_is_not_split_in_half() -> None:
    """A hand-held photo tilts the page, so a wide row drifts as it crosses it.

    One corpus invoice lost its only line item this way: the code and product
    name grouped into one row and the prices into another, and neither half
    parsed — one had no price, the other no letters.
    """
    from pharmacy_ocr.ocr.types import OCRPage, OCRToken

    def tilted(text: str, x0: float, y0: float, width: float = 150) -> OCRToken:
        # ~1.7 degrees of tilt: small, but 25px of drift across the row.
        drift = x0 * 0.03
        top, bottom = y0 + drift, y0 + drift + 18
        return OCRToken(
            text=text,
            score=0.95,
            box=(x0, top, x0 + width, bottom),
            polygon=((x0, top), (x0 + width, top + width * 0.03), (x0 + width, bottom), (x0, bottom)),
        )

    tokens = [
        tilted("301108", 30, 300, width=120),
        tilted("NOVACLAV 500 COMP B/24", 270, 300, width=570),
        tilted("16.236", 1100, 300, width=115),
        tilted("16.236", 1420, 300, width=115),
    ]
    (line,) = extract_lines([OCRPage(text="", tokens=tokens)])
    assert "301108" in line.text and "16.236" in line.text


def test_skew_estimate_ignores_upright_pages() -> None:
    from pharmacy_ocr.ocr.types import OCRToken
    from pharmacy_ocr.parsing.rules import estimate_skew

    upright = [OCRToken(text="A", box=(0, 0, 100, 20), polygon=((0, 0), (100, 0), (100, 20), (0, 20)))]
    assert estimate_skew(upright) == 0.0


def test_lines_retain_their_source_tokens() -> None:
    """Column extraction needs the x-coordinates, not just the flat text."""
    from pharmacy_ocr.ocr.types import OCRPage

    tokens = [make_token("OMNICEF", 10, 100), make_token("34.680", 400, 100)]
    (line,) = extract_lines([OCRPage(text="", tokens=tokens)])
    assert len(line.tokens) == 2
