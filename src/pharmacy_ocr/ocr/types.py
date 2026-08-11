"""Immutable OCR result types.

These are plain dataclasses rather than pydantic models: they are created once
per detected text box (hundreds per page) and never validated at a trust
boundary, so the validation overhead would buy nothing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]
Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class OCRToken:
    """One text box returned by the detector/recognizer."""

    text: str
    score: float | None = None
    box: Box | None = None
    polygon: Polygon | None = None
    page_index: int = 0

    @property
    def x0(self) -> float:
        return self.box[0] if self.box else 0.0

    @property
    def y0(self) -> float:
        return self.box[1] if self.box else 0.0

    @property
    def x1(self) -> float:
        return self.box[2] if self.box else 0.0

    @property
    def y1(self) -> float:
        return self.box[3] if self.box else 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass(frozen=True, slots=True)
class OCRPage:
    """All text recognized on a single rendered page."""

    text: str
    tokens: list[OCRToken] = field(default_factory=list)
    source_image: str = ""
    page_index: int = 0

    @property
    def mean_confidence(self) -> float | None:
        scores = [token.score for token in self.tokens if token.score is not None]
        return round(statistics.fmean(scores), 4) if scores else None

    @property
    def char_count(self) -> int:
        return len(self.text)


def pages_text(pages: list[OCRPage]) -> str:
    """Concatenate page text, keeping a blank line between pages."""
    return "\n\n".join(page.text for page in pages if page.text)
