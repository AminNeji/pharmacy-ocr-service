"""OpenCV preprocessing against real pixel data."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pharmacy_ocr.domain.errors import CorruptDocumentError
from pharmacy_ocr.imaging.preprocessing import Preprocessor, _clamp_dpi, _order_points
from pharmacy_ocr.settings import Settings


@pytest.fixture
def preproc(settings: Settings) -> Preprocessor:
    settings.ensure_directories()
    return Preprocessor(settings)


def make_image(width: int = 800, height: int = 600) -> np.ndarray:
    """A light page with a dark text-like band, as BGR."""
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (50, 80), (width - 50, 120), (30, 30, 30), -1)
    return image


def write_png(path: Path, image: np.ndarray) -> Path:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def test_clahe_preserves_shape_and_type(preproc: Preprocessor) -> None:
    image = make_image()
    result = preproc.apply_clahe(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_adaptive_threshold_returns_three_channels(preproc: Preprocessor) -> None:
    result = preproc.adaptive_threshold(make_image())
    assert result.shape == (600, 800, 3)
    assert set(np.unique(result)) <= {0, 255}


def test_sharpen_preserves_shape(preproc: Preprocessor) -> None:
    assert Preprocessor.sharpen(make_image()).shape == (600, 800, 3)


def test_small_images_are_upscaled(preproc: Preprocessor) -> None:
    """A phone photo below MIN_LONG_SIDE loses strokes unless it is enlarged."""
    result = preproc.resize_to_target_dpi(make_image(400, 300))
    assert max(result.shape[:2]) > 400


def test_oversized_images_are_capped(preproc: Preprocessor, settings: Settings) -> None:
    result = preproc.resize_to_target_dpi(make_image(6000, 4000))
    assert max(result.shape[:2]) <= settings.max_long_side


def test_in_range_images_are_untouched(preproc: Preprocessor) -> None:
    image = make_image(2000, 1500)
    assert preproc.resize_to_target_dpi(image) is image


def test_perspective_correction_falls_back_on_a_flat_page(preproc: Preprocessor) -> None:
    """No convincing quadrilateral means return the input, not a bad crop."""
    image = make_image()
    assert preproc.correct_perspective(image).shape == image.shape


# --------------------------------------------------------------------------- #
# Full document pipeline
# --------------------------------------------------------------------------- #
def test_preprocess_writes_one_page_for_an_image(
    preproc: Preprocessor, tmp_path: Path, settings: Settings
) -> None:
    source = write_png(tmp_path / "scan.png", make_image())
    pages = preproc.preprocess(source, photo_mode=False)
    try:
        assert len(pages) == 1
        assert pages[0].exists()
        assert pages[0].parent == settings.temp_dir
        assert cv2.imdecode(np.fromfile(pages[0], dtype=np.uint8), cv2.IMREAD_COLOR) is not None
    finally:
        preproc.cleanup(pages)
    assert not pages[0].exists(), "cleanup must remove the temp page"


def test_photo_mode_runs_the_heavy_path(preproc: Preprocessor, tmp_path: Path) -> None:
    source = write_png(tmp_path / "photo.png", make_image(700, 500))
    pages = preproc.preprocess(source, photo_mode=True)
    try:
        assert len(pages) == 1 and pages[0].exists()
    finally:
        preproc.cleanup(pages)


def test_missing_file_raises_a_domain_error(preproc: Preprocessor, tmp_path: Path) -> None:
    with pytest.raises(CorruptDocumentError, match="does not exist"):
        preproc.preprocess(tmp_path / "absent.png")


def test_undecodable_image_raises_a_domain_error(preproc: Preprocessor, tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\nnot actually an image")
    with pytest.raises(CorruptDocumentError):
        preproc.preprocess(broken)


def test_oversized_image_is_refused(preproc: Preprocessor, tmp_path: Path, settings: Settings) -> None:
    tiny_limit = Preprocessor(settings.model_copy(update={"max_image_megapixels": 0.1}))
    source = write_png(tmp_path / "big.png", make_image(2000, 1500))
    with pytest.raises(CorruptDocumentError, match="too large"):
        tiny_limit.preprocess(source)


def test_cleanup_tolerates_already_deleted_files(tmp_path: Path) -> None:
    Preprocessor.cleanup([tmp_path / "never-existed.png"])


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_order_points_sorts_corners_clockwise_from_top_left() -> None:
    scrambled = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    ordered = _order_points(scrambled)
    assert ordered.tolist() == [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_clamp_dpi_leaves_a_normal_page_alone() -> None:
    # A4 is 595 x 842 points; 220 DPI is about 4.6 megapixels.
    assert _clamp_dpi(595, 842, 220, max_pixels=80_000_000) == 220


def test_clamp_dpi_reduces_an_enormous_page() -> None:
    clamped = _clamp_dpi(5000, 5000, 300, max_pixels=10_000_000)
    assert 72 <= clamped < 300


def test_clamp_dpi_never_drops_below_72() -> None:
    assert _clamp_dpi(20000, 20000, 300, max_pixels=1000) == 72
