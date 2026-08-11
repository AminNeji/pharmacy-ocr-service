"""Numeric primitives for invoice rows.

Tunisian pharmacy invoices quote dinars with three millime digits. OCR renders
the separator inconsistently — "5.320", "5,320", and "5 320" all mean 5.320 —
so price detection has to accept all three while refusing to read product codes
and years as money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Either `1234.567` / `1234,567` (2-3 decimals) or `1234 567` (space + exactly
#: 3 digits). Lookarounds keep the match from starting or ending inside a longer
#: alphanumeric run such as a product code.
PRICE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<int>\d{1,5})\s*(?P<sep>[.,])\s*(?P<dec>\d{2,3})(?!\d)"
    r"|(?<![A-Za-z0-9])(?P<int2>\d{1,5})\s+(?P<dec2>\d{3})(?!\d)"
)

#: A product code at the start of a row. Characters the recognizer confuses with
#: digits are allowed here and normalized below.
#: A product code at the start of a row. Characters the recognizer confuses with
#: digits are allowed here and normalized below.
LEADING_CODE_RE = re.compile(r"^\s*(?P<code>[0-9OQIlS]{5,14})(?=$|[^0-9OQIlS])")

#: Some suppliers print alphanumeric references — `PFC034`, `80000H3`. Requiring
#: at least three digits keeps this from swallowing a medicine name that merely
#: starts with capitals ("OMNICEF", "VIT D3").
LEADING_ALNUM_CODE_RE = re.compile(r"^\s*(?P<code>(?=[A-Z0-9]{5,14}(?:$|\s))[A-Z0-9]*\d[A-Z0-9]*)(?=$|\s)")
_MIN_ALNUM_CODE_DIGITS = 3

#: A line that is nothing but a product code. OCR puts the code column in its
#: own token cluster often enough that these orphans account for a large share
#: of missing REFCODEs.
ONLY_CODE_RE = re.compile(r"^\s*(?P<code>[0-9OQIlS]{5,14}|[A-Z]{1,4}\d{3,8}[A-Z0-9]*)\s*$")

TRAILING_QTY_RE = re.compile(r"(?<![/A-Za-z0-9])(?P<qty>\d{1,3})(?![/A-Za-z0-9])\s*$")
QTY_BEFORE_X_RE = re.compile(r"(?<![/A-Za-z0-9])(?P<qty>\d{1,3})(?![/A-Za-z0-9])\s*(?:[. ]+)?[xX*]\s*$")
QTY_AFTER_LETTER_RE = re.compile(r"(?<=[A-Za-z])(?P<qty>[1-9])\s*(?:[. ]+)?[xX*]\s*$")
#: `TERCYD 250MG BT/30 CP 1 = 63.865` puts the quantity before an equals sign
#: instead of before an `x`. Without this the `1 =` stayed glued to the medicine
#: name and the quantity was reported as unknown.
QTY_BEFORE_EQUALS_RE = re.compile(r"(?<![/A-Za-z0-9])(?P<qty>\d{1,3})(?![/A-Za-z0-9])\s*=+\s*$")

#: Whatever `=` separator is left once the price has been split off the row.
TRAILING_EQUALS_RE = re.compile(r"[\s=]+$")

DIGIT_TRANSLATION = str.maketrans({"O": "0", "Q": "0", "I": "1", "l": "1", "S": "5"})

#: Above four integer digits a "price" is far more likely to be a code, a year,
#: or a phone fragment than a line total on a pharmacy invoice.
MAX_PRICE_INTEGER_DIGITS = 4
MAX_QUANTITY = 999


@dataclass(frozen=True, slots=True)
class PriceMatch:
    value: float
    start: int
    end: int
    raw: str


def find_prices(line: str) -> list[PriceMatch]:
    """All money-looking values in `line`, in reading order."""
    matches: list[PriceMatch] = []
    for match in PRICE_RE.finditer(line):
        integer = match.group("int") or match.group("int2")
        decimal = match.group("dec") or match.group("dec2")
        if integer is None or decimal is None:
            continue
        if len(integer) > MAX_PRICE_INTEGER_DIGITS:
            continue
        matches.append(
            PriceMatch(
                value=float(f"{integer}.{decimal}"),
                start=match.start(),
                end=match.end(),
                raw=match.group(0),
            )
        )
    return matches


def leading_code(line: str) -> tuple[str | None, str]:
    """Split a row into (product code, remainder).

    Digit-only codes are normalized through `DIGIT_TRANSLATION`, since the
    recognizer routinely returns `O` for `0` and `I` for `1`. Alphanumeric
    references are kept verbatim — their letters are part of the code.
    """
    match = LEADING_CODE_RE.match(line)
    if match:
        code = re.sub(r"\D+", "", match.group("code").translate(DIGIT_TRANSLATION))
        return (code or None), line[match.end() :].strip(" -:;")

    match = LEADING_ALNUM_CODE_RE.match(line)
    if match:
        code = match.group("code")
        if sum(1 for char in code if char.isdigit()) >= _MIN_ALNUM_CODE_DIGITS:
            return code, line[match.end() :].strip(" -:;")

    return None, line.strip()


def only_code(line: str) -> str | None:
    """Return the product code when `line` contains nothing else."""
    match = ONLY_CODE_RE.match(line)
    if not match:
        return None
    code = match.group("code")
    if code[0].isdigit():
        digits = re.sub(r"\D+", "", code.translate(DIGIT_TRANSLATION))
        return digits or None
    return code


def explicit_qty(text: str) -> int | None:
    """Read a quantity printed before the first price ("2 x", "1 =", trailing "3")."""
    match = (
        QTY_BEFORE_X_RE.search(text)
        or QTY_BEFORE_EQUALS_RE.search(text)
        or QTY_AFTER_LETTER_RE.search(text)
        or TRAILING_QTY_RE.search(text)
    )
    if not match:
        return None
    try:
        quantity = int(match.group("qty"))
    except ValueError:
        return None
    return quantity if 0 <= quantity <= MAX_QUANTITY else None


def nearly_equal(left: float, right: float, tolerance: float = 0.02) -> bool:
    return abs(left - right) <= tolerance


def choose_total_price(line: str, prices: list[PriceMatch], quantity: int | None) -> PriceMatch | None:
    """Pick the line total from all money values on the row.

    Two layouts dominate, and they put the total in opposite places:

        name QT unit total                    -> total is last
        code name QT prixPub totFact tRef tx percu -> total is second

    The CNAM layout appends reimbursement columns, so taking the last number
    charged the patient's copay as the line total. That was the single largest
    source of wrong prices in the corpus.

    Resolved in order of how much the row actually tells us:

    1. An explicit `=` points straight at the total.
    2. If a quantity is printed, arithmetic decides: the total is the figure
       equal to quantity x unit price.
    3. Exactly two figures with the second smaller than the first means the
       second is a deduction column, not a total — a line total is never below
       its own unit price, since quantity is at least one.
    4. Otherwise the total is the last figure, which is where the rightmost
       column sits in the common layout.

    Taking the largest figure instead of the last was measured across the
    corpus and was worse: it turns a single garbled number into the line total,
    which on one invoice inflated the extracted sum from 56.604 to 423.279.
    """
    if not prices:
        return None
    if len(prices) == 1:
        return prices[0]

    equals_position = line.find("=")
    if equals_position >= 0:
        for price in prices:
            if price.start > equals_position:
                return price

    if quantity is not None and quantity > 0:
        expected = prices[0].value * quantity
        for price in prices[1:]:
            if nearly_equal(price.value, expected):
                return price

    # `Prix Pub` and `Tot. Fact` printing the same figure is the signature of
    # the CNAM layout, where everything after them is reimbursement detail.
    if len(prices) >= 3 and nearly_equal(prices[0].value, prices[1].value):
        return prices[1]

    if len(prices) >= 3 and "x" in line.lower() and prices[1].value >= prices[-1].value:
        return prices[1]

    if len(prices) == 2 and prices[1].value < prices[0].value:
        return prices[0]

    return prices[-1]


def infer_qty(line: str, prices: list[PriceMatch], total_price: PriceMatch | None) -> int | None:
    """Read the quantity, or recover it from unit-price arithmetic."""
    if prices:
        quantity = explicit_qty(line[: prices[0].start])
        if quantity is not None:
            return quantity

    if len(prices) >= 2 and total_price is not None:
        unit = prices[0].value
        if unit > 0:
            rounded = round(total_price.value / unit)
            if 0 <= rounded <= MAX_QUANTITY and nearly_equal(unit * rounded, total_price.value):
                return int(rounded)
    return None


#: `10*10`, `10x10`, `2.5ML/5` — a number joined to the name by a dimension or
#: pack separator belongs to the product, not to the quantity column.
DIMENSION_TAIL_RE = re.compile(r"[0-9]\s*[*xX/]\s*\d{1,3}\s*$")


def strip_quantity_tail(text: str) -> str:
    """Remove a trailing quantity marker so it does not pollute the name."""
    # Drop the `=` separator first, so the quantity in `... CP 1 = ` is exposed
    # to the trailing-quantity rule below instead of being shielded by it.
    text = TRAILING_EQUALS_RE.sub("", text)
    text = re.sub(r"(?<=[A-Za-z])[1-9]\s*(?:[. ]+)?[xX*][. ]*$", "", text)
    text = re.sub(r"[. ]*[xX*][. ]*$", "", text).strip()

    # `BACTIGRAS TULLE GRAS 10*10` is a 10x10 cm dressing. Stripping the second
    # `10` as a quantity truncated the product name.
    if DIMENSION_TAIL_RE.search(text):
        return text.strip(" .:-")

    match = TRAILING_QTY_RE.search(text)
    if match and match.start() > 0:
        text = text[: match.start()].rstrip(" .-")
    return text.strip(" .:-")
