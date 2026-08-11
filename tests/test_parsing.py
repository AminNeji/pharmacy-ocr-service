"""Parser behaviour: prices, quantities, row classification, totals."""

from __future__ import annotations

import pytest

from pharmacy_ocr.parsing import numbers
from pharmacy_ocr.parsing.rules import (
    RowParser,
    cluster_tokens_into_lines,
    extract_invoice_total,
    item_quality_issue,
)
from pharmacy_ocr.settings import Settings
from tests.conftest import make_token


@pytest.fixture
def parser(settings: Settings) -> RowParser:
    return RowParser(settings)


# --------------------------------------------------------------------------- #
# Price detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5.320", [5.320]),
        ("5,320", [5.320]),
        ("5 320", [5.320]),
        ("2 17.340 34.680", [17.340, 34.680]),
        ("63.865 46.560 19.160", [63.865, 46.560, 19.160]),
    ],
)
def test_find_prices_accepts_every_separator(text: str, expected: list[float]) -> None:
    assert [match.value for match in numbers.find_prices(text)] == expected


def test_find_prices_ignores_long_identifiers() -> None:
    """A 5+ digit integer part is a code or a year, not a dinar amount."""
    assert numbers.find_prices("303154 12345.678") == []


def test_leading_code_normalizes_letter_lookalikes() -> None:
    code, rest = numbers.leading_code("3O3I54 COPRED ODT")
    assert code == "303154"
    assert rest == "COPRED ODT"


def test_letter_prefixed_code_is_recognized() -> None:
    """Some suppliers print references like `PFC034`, not bare digits."""
    code, rest = numbers.leading_code("PFC034 ZINC PLUS 25MG GELLULE B/30")
    assert code == "PFC034"
    assert rest == "ZINC PLUS 25MG GELLULE B/30"


@pytest.mark.parametrize(
    "line",
    [
        "OMNICEF 125 MG SUSP ORAL 80 ML",  # a name, not a code
        "VIT D3 BON AMP IM BUV",
        "CP 500 MG PARACETAMOL",
    ],
)
def test_medicine_names_are_not_read_as_codes(line: str) -> None:
    code, _ = numbers.leading_code(line)
    assert code is None


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("301108", "301108"),
        ("  301108  ", "301108"),
        ("3O11O8", "301108"),  # recognizer letter/digit confusion
        ("PFC034", "PFC034"),
        ("301108 NOVAFEN 600 MG", None),  # not a bare code
        ("Total", None),
        ("123", None),  # too short to be a product code
    ],
)
def test_only_code_detects_orphan_code_lines(line: str, expected: str | None) -> None:
    assert numbers.only_code(line) == expected


def test_leading_code_absent() -> None:
    code, rest = numbers.leading_code("PHYSIOMER JET NORMAL 1 23.300")
    assert code is None
    assert rest.startswith("PHYSIOMER")


# --------------------------------------------------------------------------- #
# Row parsing
# --------------------------------------------------------------------------- #
def test_parses_quantity_unit_total_row(parser: RowParser) -> None:
    item, warning = parser.parse_line("303616 OMNICEF 125 MG SUSP ORAL 80 ML 2 17.340 34.680")
    assert warning is None
    assert item == {
        "REFCODE": "303616",
        "MED": "OMNICEF 125 MG SUSP ORAL 80 ML",
        "QT": 2,
        "prix": 34.680,
        "unit_price": 17.340,
        "confidence": None,
    }


def test_cnam_row_uses_total_not_patient_copay(parser: RowParser) -> None:
    """`Prix Pub | Tot Fact | T.Ref | Tx | Perçu` — the trailing 19.160 is the
    patient's copay. Taking the last figure charged the copay as the price."""
    item, _ = parser.parse_line("300883 TERCYD 250MG BT/30 CP - 63.865 63.865 46.560 70 19.160")

    assert item is not None
    assert item["prix"] == 63.865
    assert item["QT"] == 1
    assert item["MED"] == "TERCYD 250MG BT/30 CP"


def test_cnam_row_when_ocr_merges_middle_columns(parser: RowParser) -> None:
    """Real OCR fuses `63.86546.56070`, leaving only the total and the copay."""
    item, _ = parser.parse_line("300883 TERCYD 250MG BT/30 CP - 63.865 63.86546.56070 19.160")

    assert item is not None
    assert item["prix"] == 63.865
    assert item["unit_price"] is None, "19.160 is a copay, not a unit price"


def test_quantity_times_unit_still_wins_when_printed(parser: RowParser) -> None:
    """An explicit quantity makes the total unambiguous by arithmetic."""
    item, _ = parser.parse_line("303616 OMNICEF 125 MG SUSP ORAL 80 ML 2 17.340 34.680")
    assert item is not None
    assert item["prix"] == 34.680
    assert item["unit_price"] == 17.340


def test_uses_line_total_not_refund(parser: RowParser) -> None:
    """`1 x 5 320 = 5 320 1 596` — the trailing number is a CNAM refund."""
    item, _ = parser.parse_line("303154 COPRED ODT 20 MG CP ORODISP B110...X. 5 320")
    assert item is not None
    assert item["prix"] == 5.320
    assert item["QT"] == 1


def test_recovers_pack_notation(parser: RowParser) -> None:
    item, _ = parser.parse_line("303154 COPRED ODT 20 MG CP ORODISP B110...X. 5 320")
    assert item is not None
    assert item["MED"].endswith("B/10")


def test_equals_separated_row(parser: RowParser) -> None:
    """`... CP 1 = 63.865` — the `1 =` used to end up inside the medicine name
    and the quantity was reported as unknown."""
    item, warning = parser.parse_line("TERCYD 250MG BT/30 CP 1 = 63.865")

    assert item is not None
    assert item["MED"] == "TERCYD 250MG BT/30 CP"
    assert item["QT"] == 1
    assert item["prix"] == 63.865
    assert warning is None


def test_equals_separated_row_with_code(parser: RowParser) -> None:
    item, _ = parser.parse_line("300883 TERCYD 250MG BT/30 CP 2 = 127.730")
    assert item is not None
    assert item["REFCODE"] == "300883"
    assert item["MED"] == "TERCYD 250MG BT/30 CP"
    assert item["QT"] == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TERCYD 250MG BT/30 CP 1 =", "TERCYD 250MG BT/30 CP"),
        ("PARACETAMOL 500MG 2 x", "PARACETAMOL 500MG"),
        ("PARACETAMOL 500MG =", "PARACETAMOL 500MG"),
        ("PARACETAMOL 500MG", "PARACETAMOL 500MG"),
        ("MUCOLARE 2% ENF FL/125 ML", "MUCOLARE 2% ENF FL/125 ML"),
    ],
)
def test_strip_quantity_tail(text: str, expected: str) -> None:
    assert numbers.strip_quantity_tail(text) == expected


def test_row_without_code_still_parsed(parser: RowParser) -> None:
    item, _ = parser.parse_line("PHYSIOMER JET NORMAL 1 23.300 23.300")
    assert item is not None
    assert item["REFCODE"] is None
    assert item["QT"] == 1
    assert item["prix"] == 23.300


@pytest.mark.parametrize(
    "line",
    [
        "Designation Qte Prix Unit Tot TTC",
        "Pharmacie Centrale Tel 71 123 456",
        "Total 66.175",
        "Mnt Total Facture 10.470 Dinars",
        "Matricule 1234567 A",
    ],
)
def test_non_product_rows_are_rejected(parser: RowParser, line: str) -> None:
    item, _ = parser.parse_line(line)
    assert item is None


@pytest.mark.parametrize(
    "line",
    [
        # Package-leaflet prose photographed alongside the invoice.
        "BACTIGRAS Dressings act as a wound contact layer between 0,2%+1% Creme dermique for use",
        "wounds where there is a risk of infection, or on already infected 1.500 wounds in conjunction",
        "Chaque compresse de 10 X 10 cm est imprégnée de 4.500 g de pommade au chlorhexidine",
    ],
)
def test_prose_is_not_mistaken_for_a_row(parser: RowParser, line: str) -> None:
    """A leaflet in the photo added phantom items that broke the checksum.

    An invoice row ends at its rightmost column; prose keeps going after the
    number.
    """
    item, _ = parser.parse_line(line)
    assert item is None


def test_short_column_marker_after_the_price_is_allowed(parser: RowParser) -> None:
    """Real rows do carry a trailing marker such as a dispensing code."""
    item, _ = parser.parse_line("307540 ECOREX 1% CREME 30GR 1 5.395 V")
    assert item is not None
    assert item["MED"] == "ECOREX 1% CREME 30GR"


def test_sentence_length_name_is_rejected(parser: RowParser) -> None:
    long_name = "BACTIGRAS Dressings are indicated for a wide range of wounds where risk 12.500"
    item, _ = parser.parse_line(long_name)
    assert item is None


def test_quantity_inferred_from_arithmetic(parser: RowParser) -> None:
    item, warning = parser.parse_line("300000 PARACETAMOL 500MG 3.000 9.000")
    assert item is not None
    assert item["QT"] == 3
    assert warning is None


def test_missing_quantity_produces_warning(parser: RowParser) -> None:
    item, warning = parser.parse_line("ASPIRINE 500 MG COMPRIME 12.345")
    assert item is not None
    assert item["QT"] is None
    assert warning is not None and "quantity" in warning


# --------------------------------------------------------------------------- #
# Invoice total selection
# --------------------------------------------------------------------------- #
def test_total_is_the_gross_not_a_subtotal() -> None:
    lines = ["Total HTVA = 63.865", "Total TVA 2.310", "Mnt Total Facture 66.175 Dinars"]
    assert extract_invoice_total(lines) == 66.175


def test_pretax_total_used_when_it_is_all_there_is() -> None:
    assert extract_invoice_total(["Total HTVA = 63.865"]) == 63.865


def test_total_beats_cnam_and_patient_shares() -> None:
    """A CNAM invoice prints the gross next to its two shares.

    OCR frequently re-attaches the "Total Facture" label to the reimbursed
    share, so choosing by label picked 44.706 here. The gross is never smaller
    than a share, so magnitude gets it right.
    """
    lines = [
        "Total TVA 63.865",
        "Total Facture 44.706",
        "Total a la charge de la CNAM =- 19.159",
        "- Total Percu du Client",
    ]
    assert extract_invoice_total(lines) == 63.865


def test_total_ignores_column_headers() -> None:
    assert extract_invoice_total(["Designation Qte Tot TTC 1.000"]) is None


def test_no_total_returns_none() -> None:
    assert extract_invoice_total(["ASPIRINE 500 MG 12.345"]) is None


def test_zero_total_line_never_wins() -> None:
    assert extract_invoice_total(["Total HTVA - 0.000", "Total 66.175"]) == 66.175


# --------------------------------------------------------------------------- #
# Line reconstruction
# --------------------------------------------------------------------------- #
def test_orphan_code_line_is_reattached_to_its_row() -> None:
    """The code column is often emitted as its own cluster.

    Leaving it orphaned was the largest single cause of a null REFCODE on
    invoices where the code is plainly legible.
    """
    from pharmacy_ocr.ocr.types import OCRPage

    text = "\n".join(
        [
            "Code PCT Designation Qte Prix Unit Tot TTC",
            "301108",
            "NOVAFEN 600 MG COMP. PELL. BT 20 COMPRI 6.410 6.410",
            "Total 6.410",
        ]
    )
    settings = Settings(environment="development")
    parsed = RowParser(settings).parse([OCRPage(text=text)])

    assert len(parsed.items) == 1
    assert parsed.items[0]["REFCODE"] == "301108"
    assert parsed.items[0]["MED"] == "NOVAFEN 600 MG COMP. PELL. BT 20 COMPRI"


def test_orphan_code_is_not_attached_to_a_total_line() -> None:
    """A stray code before a summary line must not manufacture an item."""
    from pharmacy_ocr.ocr.types import OCRPage

    text = "\n".join(["301235", "Total : 43.125"])
    parsed = RowParser(Settings(environment="development")).parse([OCRPage(text=text)])
    assert parsed.items == []


def test_ocr_mangled_total_is_not_an_item() -> None:
    """`Tota! : 43.125` was extracted as a medicine called "Tota!"."""
    parser = RowParser(Settings(environment="development"))
    for variant in ("Tota! : 43.125", "Tota1 : 43.125", "T0tal : 43.125", "Totai 43.125"):
        item, _ = parser.parse_line(variant)
        assert item is None, variant


def test_tokens_on_the_same_row_become_one_line() -> None:
    tokens = [
        make_token("303616", 10, 100),
        make_token("OMNICEF", 90, 102),
        make_token("34.680", 400, 101),
        make_token("301034", 10, 140),
        make_token("MUCOLARE", 90, 141),
    ]
    lines = cluster_tokens_into_lines(tokens)
    assert len(lines) == 2
    assert lines[0].text.startswith("303616")
    assert "34.680" in lines[0].text
    assert lines[1].text.startswith("301034")


def test_line_confidence_is_the_token_mean() -> None:
    tokens = [make_token("A", 10, 10, score=0.9), make_token("B", 80, 10, score=0.7)]
    (line,) = cluster_tokens_into_lines(tokens)
    assert line.confidence == pytest.approx(0.8)


def test_tokens_from_different_pages_never_merge() -> None:
    tokens = [make_token("PAGE1", 10, 100, page=0), make_token("PAGE2", 10, 100, page=1)]
    assert len(cluster_tokens_into_lines(tokens)) == 2


def test_clustering_is_linear_on_large_inputs() -> None:
    """Guards the O(n^2) regression: 4000 tokens must not take seconds."""
    import time

    tokens = [
        make_token(f"t{index}", 10 + (index % 8) * 70, 100 + (index // 8) * 40) for index in range(4000)
    ]
    start = time.perf_counter()
    lines = cluster_tokens_into_lines(tokens)
    assert len(lines) == 500
    assert time.perf_counter() - start < 2.0


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("med", "prix", "expected_issue"),
    [
        ("OMNICEF 125 MG", 5.0, None),
        ("B/", 5.0, "medicine name is too short to be reliable"),
        ("sirop", 5.0, "medicine name is only a form/status marker"),
        ("OMNICEF 125 MG", 0.0, "missing non-zero line total"),
        ("OMNICEF 125 MG", None, "missing non-zero line total"),
    ],
)
def test_item_quality_issue(med: str, prix: float | None, expected_issue: str | None) -> None:
    assert item_quality_issue(None, med, prix) == expected_issue


def test_repeated_rows_survive_when_not_adjacent(parser: RowParser) -> None:
    """Two genuine occurrences of the same medicine must both be kept."""
    from pharmacy_ocr.ocr.types import OCRPage

    text = "\n".join(
        [
            "300000 ASPIRINE 500MG 1 5.000 5.000",
            "300001 DOLIPRANE 1000MG 1 3.000 3.000",
            "300000 ASPIRINE 500MG 1 5.000 5.000",
            "Mnt Total Facture 13.000",
        ]
    )
    parsed = parser.parse([OCRPage(text=text)])
    assert len(parsed.items) == 3
    assert parsed.invoice_total == 13.000
