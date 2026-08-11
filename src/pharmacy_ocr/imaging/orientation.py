"""Page-orientation detection from OCR geometry.

PaddleOCR's own document-orientation classifier is enabled, but it does not
catch every sideways scan: three invoices in the reference corpus come back
with their text boxes standing on end, and two of those produced no items at
all because a "row" was then a vertical strip of one character.

Detection uses the recognized boxes rather than the raw image because it is
both cheaper and more reliable than a Hough transform on a noisy phone photo:
upright text lines are far wider than they are tall, and that ratio inverts
when the page is turned. The cost is one extra OCR pass, paid only on the small
fraction of pages that are actually rotated.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from pharmacy_ocr.ocr.types import OCRPage

logger = logging.getLogger(__name__)

#: Above this share of taller-than-wide boxes, the page is read as sideways.
#: Upright invoices in the corpus sit near 0.05; rotated ones at 0.82-0.87, so
#: the threshold is nowhere near either population.
TALL_BOX_FRACTION = 0.5

#: Below this many boxes the fraction is noise, not evidence.
MIN_BOXES_FOR_DETECTION = 8


@dataclass(frozen=True, slots=True)
class OrientationEvidence:
    """What the OCR geometry says about which way up the page is."""

    tall_fraction: float
    token_count: int
    mean_confidence: float
    total_chars: int

    @property
    def looks_sideways(self) -> bool:
        return self.token_count >= MIN_BOXES_FOR_DETECTION and self.tall_fraction > TALL_BOX_FRACTION

    @property
    def quality(self) -> float:
        """A single number for comparing two candidate orientations.

        Characters recognized, weighted by how confident the recognizer was.
        Both matter: a rotated page can yield many low-confidence fragments, and
        a blank one yields few high-confidence ones.
        """
        return self.total_chars * max(self.mean_confidence, 0.01)


def assess(pages: list[OCRPage]) -> OrientationEvidence:
    tokens = [token for page in pages for token in page.tokens if token.box]
    if not tokens:
        return OrientationEvidence(0.0, 0, 0.0, sum(len(page.text) for page in pages))

    tall = sum(1 for token in tokens if token.height > token.width)
    scores = [token.score for token in tokens if token.score is not None]
    return OrientationEvidence(
        tall_fraction=tall / len(tokens),
        token_count=len(tokens),
        mean_confidence=statistics.fmean(scores) if scores else 0.0,
        total_chars=sum(len(token.text) for token in tokens),
    )


def candidate_rotations(evidence: OrientationEvidence) -> tuple[int, ...]:
    """Rotations worth re-running OCR at, given the first pass.

    A sideways page is either 90 or 270 and the box shapes cannot tell which,
    so both are tried and the better result kept. An upright page needs nothing.
    """
    if not evidence.looks_sideways:
        return ()
    return (270, 90)


#: Angles to sweep when a page produced no invoice rows at all. 180 is here and
#: not in `candidate_rotations` because an upside-down page keeps normal box
#: shapes: the aspect-ratio test cannot see it, and neither can PaddleOCR's
#: text-line orientation pass in practice. The only reliable signal is that
#: nothing parsed, so the sweep is gated on that outcome.
RECOVERY_ROTATIONS: tuple[int, ...] = (180, 270, 90)


def recovery_rotations(already_tried: set[int]) -> tuple[int, ...]:
    """Untried orientations, for a page that yielded no rows."""
    return tuple(angle for angle in RECOVERY_ROTATIONS if angle % 360 not in already_tried)
