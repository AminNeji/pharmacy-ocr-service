"""Scoring extraction output against hand-labelled ground truth.

The invoice-total checksum is a proxy: it can only tell you whether the prices
add up. It cannot see a wrong medicine name, a missing REFCODE, or a row that
was invented. This module scores against labels transcribed from the invoice
images, which is the only measurement that reflects what a pharmacist would
call correct.

Rows are matched between prediction and truth before fields are compared,
because a missing or extra row would otherwise shift every later comparison and
report a near-zero score for output that is mostly right.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Two prices are the same if they agree to the millime.
PRICE_TOLERANCE = 0.005


def normalize_name(value: str | None) -> str:
    """Fold a medicine name to its comparable core.

    OCR and humans disagree on punctuation, accents, and spacing far more often
    than on the actual product, so scoring on the raw string would understate
    accuracy for reasons nobody cares about.
    """
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", str(value))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = folded.upper().replace("*", "X")
    folded = re.sub(r"[^A-Z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def normalize_code(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def prices_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= PRICE_TOLERANCE


@dataclass
class FieldScore:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.correct += int(ok)


@dataclass
class InvoiceScore:
    name: str
    expected_rows: int = 0
    predicted_rows: int = 0
    matched_rows: int = 0
    missing_rows: list[str] = field(default_factory=list)
    spurious_rows: list[str] = field(default_factory=list)
    fields: dict[str, FieldScore] = field(default_factory=dict)
    #: (field, expected, got) for each wrong value, so a run says what to fix.
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)
    seconds: float | None = None

    @property
    def row_recall(self) -> float:
        return self.matched_rows / self.expected_rows if self.expected_rows else 1.0

    @property
    def row_precision(self) -> float:
        return self.matched_rows / self.predicted_rows if self.predicted_rows else 1.0

    @property
    def exact(self) -> bool:
        """Every expected row found, nothing invented, every field right."""
        return (
            self.matched_rows == self.expected_rows
            and self.predicted_rows == self.expected_rows
            and all(score.correct == score.total for score in self.fields.values())
        )


def _match_rows(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Pair predicted rows with expected rows, best candidate first.

    Matching is greedy on a similarity score rather than positional, so one
    dropped row does not cascade into every subsequent row being scored wrong.
    """
    unmatched_pred = list(predicted)
    pairs: list[tuple[dict, dict]] = []
    missing: list[dict] = []

    def similarity(want: dict, got: dict) -> float:
        score = 0.0
        want_code, got_code = normalize_code(want.get("REFCODE")), normalize_code(got.get("REFCODE"))
        if want_code and want_code == got_code:
            score += 3.0
        want_name, got_name = normalize_name(want.get("MED")), normalize_name(got.get("MED"))
        if want_name and got_name:
            if want_name == got_name:
                score += 3.0
            elif want_name.startswith(got_name[:8]) or got_name.startswith(want_name[:8]):
                score += 2.0
            else:
                shared = len(set(want_name.split()) & set(got_name.split()))
                score += min(1.5, shared * 0.5)
        if prices_match(want.get("prix"), got.get("prix")):
            score += 2.0
        return score

    for want in expected:
        best, best_score = None, 0.0
        for got in unmatched_pred:
            value = similarity(want, got)
            if value > best_score:
                best, best_score = got, value
        # Require more than a lone price coincidence before calling it a match.
        if best is not None and best_score >= 2.0:
            unmatched_pred.remove(best)
            pairs.append((want, best))
        else:
            missing.append(want)

    return pairs, missing, unmatched_pred


def score_invoice(
    name: str,
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    seconds: float | None = None,
) -> InvoiceScore:
    pairs, missing, spurious = _match_rows(expected, predicted)

    result = InvoiceScore(
        name=name,
        expected_rows=len(expected),
        predicted_rows=len(predicted),
        matched_rows=len(pairs),
        missing_rows=[str(row.get("MED", "?")) for row in missing],
        spurious_rows=[str(row.get("MED", "?")) for row in spurious],
        seconds=seconds,
        fields={key: FieldScore() for key in ("REFCODE", "MED", "QT", "prix")},
    )

    for want, got in pairs:
        checks = (
            ("REFCODE", normalize_code(want.get("REFCODE")) == normalize_code(got.get("REFCODE"))),
            ("MED", normalize_name(want.get("MED")) == normalize_name(got.get("MED"))),
            ("QT", want.get("QT") == got.get("QT")),
            ("prix", prices_match(want.get("prix"), got.get("prix"))),
        )
        for key, ok in checks:
            result.fields[key].add(ok)
            if not ok:
                result.mismatches.append((key, str(want.get(key)), str(got.get(key))))

    # A row we never found still counts against every field, otherwise dropping
    # a hard row would look like perfect field accuracy.
    for _ in missing:
        for score in result.fields.values():
            score.add(False)

    return result


@dataclass
class Report:
    invoices: list[InvoiceScore] = field(default_factory=list)

    def add(self, score: InvoiceScore) -> None:
        self.invoices.append(score)

    def field_accuracy(self, key: str) -> float:
        correct = sum(inv.fields[key].correct for inv in self.invoices if key in inv.fields)
        total = sum(inv.fields[key].total for inv in self.invoices if key in inv.fields)
        return correct / total if total else 1.0

    def summary(self) -> dict[str, Any]:
        expected = sum(inv.expected_rows for inv in self.invoices)
        predicted = sum(inv.predicted_rows for inv in self.invoices)
        matched = sum(inv.matched_rows for inv in self.invoices)
        times = [inv.seconds for inv in self.invoices if inv.seconds is not None]
        return {
            "invoices": len(self.invoices),
            "invoices_exact": sum(1 for inv in self.invoices if inv.exact),
            "rows_expected": expected,
            "rows_predicted": predicted,
            "rows_matched": matched,
            "row_recall": round(matched / expected, 4) if expected else 1.0,
            "row_precision": round(matched / predicted, 4) if predicted else 1.0,
            "false_negatives": expected - matched,
            "false_positives": predicted - matched,
            "field_accuracy": {
                key: round(self.field_accuracy(key), 4) for key in ("REFCODE", "MED", "QT", "prix")
            },
            "seconds_mean": round(sum(times) / len(times), 2) if times else None,
            "seconds_max": round(max(times), 2) if times else None,
        }

    def render(self) -> str:
        lines = [
            f"{'invoice':30} {'rows':>9} {'REF':>5} {'MED':>5} {'QT':>5} {'prix':>5} {'sec':>6}",
            "-" * 72,
        ]
        for inv in sorted(self.invoices, key=lambda i: i.name):
            flag = "OK " if inv.exact else "   "
            lines.append(
                f"{flag}{inv.name[:27]:27} {inv.matched_rows:>3}/{inv.expected_rows:<5} "
                f"{inv.fields['REFCODE'].accuracy:>5.0%} {inv.fields['MED'].accuracy:>5.0%} "
                f"{inv.fields['QT'].accuracy:>5.0%} {inv.fields['prix'].accuracy:>5.0%} "
                f"{inv.seconds if inv.seconds is None else round(inv.seconds, 1):>6}"
            )
            for med in inv.missing_rows:
                lines.append(f"      MISSING  {med[:60]}")
            for med in inv.spurious_rows:
                lines.append(f"      INVENTED {med[:60]}")
            for key, want, got in inv.mismatches:
                lines.append(f"      {key:<7}  want {want[:32]!r}  got {got[:32]!r}")

        summary = self.summary()
        lines += [
            "-" * 72,
            f"exact invoices : {summary['invoices_exact']}/{summary['invoices']}",
            f"row recall     : {summary['row_recall']:.1%}  " f"(missed {summary['false_negatives']})",
            f"row precision  : {summary['row_precision']:.1%}  " f"(invented {summary['false_positives']})",
            "field accuracy : " + "  ".join(f"{k}={v:.1%}" for k, v in summary["field_accuracy"].items()),
            f"time           : mean {summary['seconds_mean']}s  max {summary['seconds_max']}s",
        ]
        return "\n".join(lines)


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("invoices", {})
