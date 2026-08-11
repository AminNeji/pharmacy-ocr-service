"""OpenCV preprocessing.

photo_mode=True (phone photos)
    perspective correction -> denoise -> CLAHE -> sharpen -> optional adaptive
    threshold -> resize toward a 300-DPI-equivalent long side.
photo_mode=False (clean scans)
    light CLAHE + resize only, which is several times faster.

PaddleOCR reads file paths rather than arrays, so each prepared page is written
to a temp PNG and the path returned; callers must call `cleanup`.

PDF pages are rendered and written one at a time. The previous version built a
list of every decoded page first, which held roughly 26 MB of BGR pixels per A4
page at 300 DPI — five pages was 130 MB resident before OCR even started.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from pharmacy_ocr.domain.errors import CorruptDocumentError
from pharmacy_ocr.imaging.files import guard_megapixels, probe_image_dimensions
from pharmacy_ocr.settings import Settings

logger = logging.getLogger(__name__)


class Preprocessor:
    """Turns an input document into OCR-ready page images."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------- io ---
    def _read_image(self, path: Path) -> np.ndarray:
        dimensions = probe_image_dimensions(path)
        if dimensions is not None:
            guard_megapixels(*dimensions, self._settings.max_image_megapixels, source=path.name)
        # np.fromfile keeps non-ASCII Windows paths working, which cv2.imread
        # does not handle.
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise CorruptDocumentError(f"Could not decode image: {path.name}")
        if dimensions is None:
            height, width = image.shape[:2]
            guard_megapixels(width, height, self._settings.max_image_megapixels, source=path.name)
        return image

    def _write_temp(self, image: np.ndarray) -> Path:
        self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._settings.temp_dir / f"page_{uuid.uuid4().hex}.png"
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise CorruptDocumentError("PNG encoding failed for a prepared page")
        try:
            # numpy.tofile can intermittently write 0 bytes on OneDrive-backed
            # paths; a plain binary write costs a few ms and is reliable.
            out_path.write_bytes(buffer.tobytes())
        except OSError:
            out_path.unlink(missing_ok=True)
            raise
        return out_path

    def _iter_pdf_pages(self, path: Path, dpi: int) -> Iterator[np.ndarray]:
        """Yield rendered PDF pages one at a time, never holding two at once."""
        import fitz  # pymupdf

        max_pixels = self._settings.max_image_megapixels * 1_000_000
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open PDF: {exc}") from exc

        with document as doc:
            if doc.page_count == 0:
                raise CorruptDocumentError(f"PDF has no pages: {path.name}")
            if doc.needs_pass:
                raise CorruptDocumentError("PDF is password protected")

            for index, page in enumerate(doc):
                if index >= self._settings.pdf_max_pages:
                    logger.warning(
                        "PDF truncated",
                        extra={"file": path.name, "max_pages": self._settings.pdf_max_pages},
                    )
                    break

                # Clamp DPI so an unusually large page cannot allocate an
                # arbitrarily large pixmap.
                effective_dpi = _clamp_dpi(page.rect.width, page.rect.height, dpi, max_pixels)
                if effective_dpi < dpi:
                    logger.info(
                        "reduced PDF render DPI to stay within the pixel budget",
                        extra={"file": path.name, "page": index, "dpi": effective_dpi},
                    )
                try:
                    pix = page.get_pixmap(dpi=effective_dpi, colorspace=fitz.csRGB, alpha=False)
                    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    yield cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                except CorruptDocumentError:
                    raise
                except Exception as exc:
                    raise CorruptDocumentError(
                        f"Could not render page {index + 1} of {path.name}: {exc}"
                    ) from exc

    # -------------------------------------------------------------- steps -----
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Warp the invoice sheet in a phone photo to a flat rectangle.

        Returns the input unchanged when no convincing quadrilateral is found,
        which is the common case for already-cropped photos and busy backgrounds.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        frame_area = float(height * width)
        best_quad: np.ndarray | None = None
        best_area = 0.0
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area / frame_area < self._settings.perspective_min_area_ratio:
                break
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and area > best_area:
                best_quad = approx.reshape(4, 2).astype(np.float32)
                best_area = area

        if best_quad is None:
            return image

        rect = _order_points(best_quad)
        top_left, top_right, bottom_right, bottom_left = rect
        # Take the longer of each opposite pair so a tilted sheet is not cropped.
        out_w = int(
            max(
                float(np.linalg.norm(bottom_right - bottom_left)),
                float(np.linalg.norm(top_right - top_left)),
            )
        )
        out_h = int(
            max(
                float(np.linalg.norm(top_right - bottom_right)),
                float(np.linalg.norm(top_left - bottom_left)),
            )
        )
        if out_w < 100 or out_h < 100:
            return image

        destination = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(rect, destination)
        return cv2.warpPerspective(image, matrix, (out_w, out_h))

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        grid = self._settings.clahe_tile_grid
        clahe = cv2.createCLAHE(clipLimit=self._settings.clahe_clip_limit, tileGridSize=(grid, grid))
        merged = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        strength = self._settings.denoise_strength
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21
        )

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._settings.adaptive_block_size,
            self._settings.adaptive_c,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def resize_to_target_dpi(
        self,
        image: np.ndarray,
        target_long_side: int | None = None,
        max_long_side: int | None = None,
    ) -> np.ndarray:
        """Bring the long side into range, targeting an A4-at-300-DPI equivalent.

        Small phone photos get upscaled so glyph strokes survive detection;
        oversized scans get capped so memory and OCR time stay bounded.
        """
        target = target_long_side or self._settings.target_long_side
        minimum = self._settings.min_long_side
        maximum = max_long_side or self._settings.max_long_side
        long_side = max(image.shape[:2])

        if long_side < minimum:
            scale = target / long_side
            interpolation = cv2.INTER_CUBIC
        elif long_side > maximum:
            scale = maximum / long_side
            interpolation = cv2.INTER_AREA
        else:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    # ------------------------------------------------------------ pipeline ----
    def _prepare_page(self, image: np.ndarray, photo_mode: bool) -> np.ndarray:
        if photo_mode:
            image = self.correct_perspective(image)
            image = self.denoise(image)
            image = self.apply_clahe(image)
            image = self.sharpen(image)
            if self._settings.apply_adaptive_threshold:
                image = self.adaptive_threshold(image)
        else:
            image = self.apply_clahe(image)
        return image

    @staticmethod
    def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
        """Rotate by a right angle without resampling or cropping."""
        turns = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        code = turns.get(degrees % 360)
        return image if code is None else cv2.rotate(image, code)

    def preprocess(
        self,
        input_path: str | Path,
        photo_mode: bool | None = None,
        pdf_render_dpi: int | None = None,
        target_long_side: int | None = None,
        max_long_side: int | None = None,
        rotate_degrees: int = 0,
    ) -> list[Path]:
        """Prepare every page of `input_path`; returns temp PNG paths."""
        path = Path(input_path)
        if not path.is_file():
            raise CorruptDocumentError(f"Input file does not exist: {path.name}")
        if photo_mode is None:
            photo_mode = self._settings.photo_mode

        if path.suffix.lower() == ".pdf":
            sources = self._iter_pdf_pages(path, pdf_render_dpi or self._settings.pdf_render_dpi)
        else:
            sources = iter([self._read_image(path)])

        output_paths: list[Path] = []
        try:
            for image in sources:
                # Turn the page upright before anything else: perspective
                # correction and thresholding both assume horizontal text.
                if rotate_degrees % 360:
                    image = self.rotate(image, rotate_degrees)
                prepared = self._prepare_page(image, photo_mode)
                prepared = self.resize_to_target_dpi(prepared, target_long_side, max_long_side)
                output_paths.append(self._write_temp(prepared))
                del image, prepared  # release page memory before rendering the next
        except BaseException:
            self.cleanup(output_paths)
            raise

        if not output_paths:
            raise CorruptDocumentError(f"No renderable pages in {path.name}")
        return output_paths

    @contextmanager
    def prepared_pages(
        self,
        input_path: str | Path,
        photo_mode: bool | None = None,
        pdf_render_dpi: int | None = None,
        target_long_side: int | None = None,
        max_long_side: int | None = None,
    ) -> Iterator[list[Path]]:
        """`preprocess` with guaranteed cleanup."""
        paths = self.preprocess(input_path, photo_mode, pdf_render_dpi, target_long_side, max_long_side)
        try:
            yield paths
        finally:
            self.cleanup(paths)

    @staticmethod
    def cleanup(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not delete temp page", extra={"temp_file": str(path)})


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    rect[0] = points[np.argmin(total)]
    rect[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1).ravel()
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _clamp_dpi(width_pt: float, height_pt: float, dpi: int, max_pixels: float) -> int:
    """Lower `dpi` until the rendered page fits inside `max_pixels`."""
    scale = dpi / 72.0
    pixels = (width_pt * scale) * (height_pt * scale)
    if pixels <= max_pixels or pixels <= 0:
        return dpi
    factor = (max_pixels / pixels) ** 0.5
    return max(72, int(dpi * factor))
