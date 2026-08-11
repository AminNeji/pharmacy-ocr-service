"""OCR engine abstraction.

The pipeline depends on the `OCREngine` protocol, not on PaddleOCR. That
indirection is what makes the parser, cache, pipeline, and API testable without
loading 500 MB of model weights — tests inject a fake engine that returns
recorded tokens.

Engine state (the loaded predictor and its lock) lives on the instance rather
than in module globals, so two engines with different settings can coexist and
tests never leak a predictor between cases.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pharmacy_ocr.domain.errors import OCREngineError
from pharmacy_ocr.ocr.types import Box, OCRPage, OCRToken, Polygon
from pharmacy_ocr.settings import OCREngineName, Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class OCREngine(Protocol):
    """Reads text boxes from a single rendered page image."""

    @property
    def name(self) -> str: ...

    def warmup(self) -> None:
        """Load models eagerly so the first real request is not slow."""
        ...

    def run_page(self, image_path: str | Path, page_index: int = 0) -> OCRPage: ...


# --------------------------------------------------------------------------- #
# Result normalization
#
# PaddleOCR's return shape has changed repeatedly across 2.x/3.x. These helpers
# accept every shape we have observed and fall back progressively rather than
# raising, because a partial read still beats a hard failure on one page.
# --------------------------------------------------------------------------- #
def _as_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _as_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain_data(item) for item in value]
    return value


_RESULT_KEYS = ("rec_texts", "rec_scores", "rec_boxes", "rec_polys")


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return _as_plain_data(result)

    for attr in ("json", "to_json", "to_dict", "_to_dict"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except TypeError:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            return _as_plain_data(value)

    if hasattr(result, "get"):
        data: dict[str, Any] = {}
        for key in _RESULT_KEYS:
            try:
                value = result.get(key)
            except Exception:  # noqa: BLE001 - third-party mapping, any error means "absent"
                value = None
            if value is not None:
                data[key] = _as_plain_data(value)
        if data:
            return data

    return {
        key: _as_plain_data(getattr(result, key))
        for key in _RESULT_KEYS
        if getattr(result, key, None) is not None
    }


def resolve_device(configured: str) -> str:
    """Turn an `ocr_device` setting into a device PaddleOCR will accept.

    "auto" picks the GPU only when PaddlePaddle was actually built with CUDA
    *and* a device is visible. The CPU-only wheel raises rather than falling
    back if asked for "gpu", and a machine can have a driver but no CUDA build,
    so both halves have to be checked.
    """
    requested = (configured or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            logger.info("auto-selected GPU for OCR")
            return "gpu"
    except Exception:  # noqa: BLE001 - any probe failure means "use the CPU"
        logger.debug("CUDA probe failed; using CPU", exc_info=True)
    return "cpu"


def _points(poly: Any) -> list[tuple[float, float]]:
    if not poly:
        return []
    points: list[tuple[float, float]] = []
    for point in poly:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return points


def _box_from_poly(poly: Any) -> Box | None:
    points = _points(poly)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_from_poly(poly: Any) -> Polygon | None:
    points = _points(poly)
    return tuple(points) if points else None


def _tokens_from_data(data: dict[str, Any], page_index: int, min_confidence: float) -> list[OCRToken]:
    texts = data.get("rec_texts") or data.get("texts") or []
    scores = data.get("rec_scores") or data.get("scores") or []
    boxes = data.get("rec_boxes") or data.get("boxes") or []
    polys = data.get("rec_polys") or data.get("dt_polys") or []

    tokens: list[OCRToken] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue

        score: float | None = None
        if index < len(scores):
            try:
                score = float(scores[index])
            except (TypeError, ValueError):
                score = None
        if score is not None and score < min_confidence:
            continue

        box: Box | None = None
        if index < len(boxes):
            raw_box = boxes[index]
            if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
                try:
                    values = [float(value) for value in raw_box[:4]]
                    box = (values[0], values[1], values[2], values[3])
                except (TypeError, ValueError):
                    box = None

        polygon: Polygon | None = None
        if index < len(polys):
            polygon = _polygon_from_poly(polys[index])
            box = box or _box_from_poly(polys[index])

        tokens.append(OCRToken(text=text, score=score, box=box, polygon=polygon, page_index=page_index))
    return tokens


def _tokens_from_legacy(result: Any, page_index: int, min_confidence: float) -> list[OCRToken]:
    """Parse PaddleOCR 2.x output: [[box, (text, score)], ...]."""
    rows = result
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list):
        return []

    tokens: list[OCRToken] = []
    for row in rows:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        poly, payload = row[0], row[1]
        if not (isinstance(payload, (list, tuple)) and payload):
            continue
        text = str(payload[0]).strip()
        if not text:
            continue
        score: float | None = None
        if len(payload) > 1:
            try:
                score = float(payload[1])
            except (TypeError, ValueError):
                score = None
        if score is not None and score < min_confidence:
            continue
        tokens.append(
            OCRToken(
                text=text,
                score=score,
                box=_box_from_poly(poly),
                polygon=_polygon_from_poly(poly),
                page_index=page_index,
            )
        )
    return tokens


def _extract_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, dict):
        return [text for value in obj.values() for text in _extract_strings(value)]
    if isinstance(obj, (list, tuple)):
        return [text for value in obj for text in _extract_strings(value)]
    return []


# --------------------------------------------------------------------------- #
# PaddleOCR engine
# --------------------------------------------------------------------------- #
class PaddleOCREngine:
    """PaddleOCR-backed engine.

    `run_page` is serialized with a lock because Paddle predictors are not
    thread-safe. Callers get parallelism from overlapping preprocessing and LLM
    mapping with OCR, not from concurrent OCR.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._predictor: Any | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._settings.ocr_engine.value

    # ------------------------------------------------------------------ load --
    def _apply_paddle_env(self) -> None:
        if not self._settings.paddle_disable_mkldnn:
            return
        # PaddlePaddle 3.3.x has a CPU oneDNN/PIR regression that breaks text
        # detection with ConvertPirAttribute2RuntimeAttribute errors. These must
        # be set before PaddleOCR imports Paddle internals, which is why the
        # import below is lazy.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")

    def _get_predictor(self) -> Any:
        # Double-checked locking: the fast path avoids the lock once loaded, and
        # the second read inside the lock catches a concurrent loader. Both reads
        # go through a local so the value cannot change between test and use.
        predictor = self._predictor
        if predictor is not None:
            return predictor

        with self._load_lock:
            predictor = self._predictor
            if predictor is not None:
                return predictor
            self._apply_paddle_env()
            start = time.perf_counter()
            try:
                self._predictor = (
                    self._build_vl_predictor()
                    if self._settings.ocr_engine is OCREngineName.vl
                    else self._build_general_predictor()
                )
            except OCREngineError:
                raise
            except Exception as exc:
                raise OCREngineError(f"could not load OCR engine '{self.name}': {exc}") from exc
            logger.info(
                "OCR engine loaded",
                extra={"engine": self.name, "load_seconds": round(time.perf_counter() - start, 2)},
            )
            return self._predictor

    def _build_general_predictor(self) -> Any:
        from paddleocr import PaddleOCR

        settings = self._settings
        kwargs: dict[str, Any] = {
            "lang": settings.ocr_lang,
            "ocr_version": settings.ocr_version,
            "device": resolve_device(settings.ocr_device),
            "use_doc_orientation_classify": settings.use_doc_orientation_classify,
            "use_doc_unwarping": settings.use_doc_unwarping,
            "use_textline_orientation": settings.use_textline_orientation,
            "text_recognition_batch_size": settings.ocr_text_recognition_batch_size,
        }
        if settings.ocr_text_detection_model_name:
            kwargs["text_detection_model_name"] = settings.ocr_text_detection_model_name
        if settings.ocr_text_recognition_model_name:
            kwargs["text_recognition_model_name"] = settings.ocr_text_recognition_model_name

        logger.info("loading PaddleOCR general engine", extra={"ocr_kwargs": kwargs})
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            # Older PaddleOCR builds use the 2.x constructor. Keep a compatible
            # path so a partially upgraded environment still starts.
            logger.warning("PaddleOCR 3.x constructor rejected; falling back to 2.x arguments")
            return PaddleOCR(use_angle_cls=settings.use_textline_orientation, lang=settings.ocr_lang)

    def _build_vl_predictor(self) -> Any:
        from paddleocr import PaddleOCRVL

        logger.warning(
            "loading PaddleOCR-VL legacy engine; this is slow on CPU and returns no box geometry",
            extra={"device": self._settings.ocr_device},
        )
        return PaddleOCRVL(
            device=resolve_device(self._settings.ocr_device),
            use_doc_orientation_classify=self._settings.use_doc_orientation_classify,
            use_doc_unwarping=self._settings.use_doc_unwarping,
        )

    def warmup(self) -> None:
        self._get_predictor()

    # ---------------------------------------------------------------- predict --
    def run_page(self, image_path: str | Path, page_index: int = 0) -> OCRPage:
        path = str(image_path)
        if self._settings.ocr_engine is OCREngineName.vl:
            return self._run_vl(path, page_index)
        return self._run_general(path, page_index)

    def _predict(self, path: str) -> Any:
        predictor = self._get_predictor()
        try:
            with self._predict_lock:
                if hasattr(predictor, "predict"):
                    return predictor.predict(path)
                return predictor.ocr(path, cls=self._settings.use_textline_orientation)
        except Exception as exc:
            raise OCREngineError(f"OCR failed on {Path(path).name}: {exc}") from exc

    def _run_general(self, path: str, page_index: int) -> OCRPage:
        start = time.perf_counter()
        results = self._predict(path)
        min_confidence = self._settings.ocr_min_confidence

        tokens: list[OCRToken] = []
        result_list = results if isinstance(results, list) else [results]
        for result in result_list:
            data = _result_to_dict(result)
            page_tokens = _tokens_from_data(data, page_index, min_confidence)
            if not page_tokens:
                page_tokens = _tokens_from_legacy(result, page_index, min_confidence)
            tokens.extend(page_tokens)

        text = "\n".join(token.text for token in tokens).strip()
        logger.info(
            "page OCR complete",
            extra={
                "page": page_index,
                "tokens": len(tokens),
                "chars": len(text),
                "seconds": round(time.perf_counter() - start, 2),
            },
        )
        return OCRPage(text=text, tokens=tokens, source_image=path, page_index=page_index)

    def _run_vl(self, path: str, page_index: int) -> OCRPage:
        start = time.perf_counter()
        results = self._predict(path)
        parts = [self._vl_text(result) for result in results]
        text = "\n\n".join(part for part in parts if part).strip()
        logger.info(
            "page VL OCR complete",
            extra={
                "page": page_index,
                "chars": len(text),
                "seconds": round(time.perf_counter() - start, 2),
            },
        )
        return OCRPage(text=text, tokens=[], source_image=path, page_index=page_index)

    @staticmethod
    def _vl_text(result: Any) -> str:
        markdown = getattr(result, "markdown", None)
        if markdown is None and isinstance(result, dict):
            markdown = result.get("markdown")
        texts = _extract_strings(markdown)
        if not texts:
            raw = getattr(result, "json", None)
            if raw is not None:
                raw = raw() if callable(raw) else raw
                texts = _extract_strings(raw)
        return "\n".join(texts).strip()


def create_engine(settings: Settings) -> OCREngine:
    """Build the engine named by `settings.ocr_engine`."""
    return PaddleOCREngine(settings)
