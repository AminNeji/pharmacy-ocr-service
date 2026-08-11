"""PaddleOCR result normalization, LLM response parsing, log formatting.

These are the defensive shims that absorb third-party output shapes. They are
exactly the code that breaks quietly on a dependency upgrade, so they are worth
pinning down without needing the dependency installed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from pharmacy_ocr.observability.logging import JsonFormatter, bind_request_id
from pharmacy_ocr.ocr.engine import (
    _box_from_poly,
    _result_to_dict,
    _tokens_from_data,
    _tokens_from_legacy,
)
from pharmacy_ocr.ocr.types import OCRPage
from pharmacy_ocr.parsing.llm import parse_items, rows_text_from_pages

MIN_CONF = 0.35


# --------------------------------------------------------------------------- #
# PaddleOCR 3.x dict output
# --------------------------------------------------------------------------- #
def test_tokens_from_modern_result() -> None:
    data = {
        "rec_texts": ["303616", "OMNICEF", "34.680"],
        "rec_scores": [0.99, 0.97, 0.95],
        "rec_boxes": [[10, 100, 70, 120], [90, 100, 200, 120], [400, 100, 460, 120]],
    }
    tokens = _tokens_from_data(data, page_index=0, min_confidence=MIN_CONF)

    assert [token.text for token in tokens] == ["303616", "OMNICEF", "34.680"]
    assert tokens[0].box == (10.0, 100.0, 70.0, 120.0)
    assert tokens[1].score == 0.97


def test_low_confidence_tokens_are_dropped() -> None:
    data = {"rec_texts": ["good", "garbled"], "rec_scores": [0.9, 0.1]}
    tokens = _tokens_from_data(data, 0, MIN_CONF)
    assert [token.text for token in tokens] == ["good"]


def test_blank_and_whitespace_tokens_are_dropped() -> None:
    data = {"rec_texts": ["real", "   ", ""], "rec_scores": [0.9, 0.9, 0.9]}
    assert len(_tokens_from_data(data, 0, MIN_CONF)) == 1


def test_box_is_derived_from_polygon_when_absent() -> None:
    data = {
        "rec_texts": ["OMNICEF"],
        "rec_scores": [0.9],
        "rec_polys": [[[10, 100], [70, 102], [70, 120], [10, 118]]],
    }
    (token,) = _tokens_from_data(data, 0, MIN_CONF)
    assert token.box == (10.0, 100.0, 70.0, 120.0)
    assert token.polygon is not None and len(token.polygon) == 4


def test_missing_scores_and_boxes_are_tolerated() -> None:
    (token,) = _tokens_from_data({"rec_texts": ["OMNICEF"]}, 0, MIN_CONF)
    assert token.score is None
    assert token.box is None


def test_malformed_score_does_not_crash() -> None:
    data = {"rec_texts": ["OMNICEF"], "rec_scores": ["not-a-number"]}
    (token,) = _tokens_from_data(data, 0, MIN_CONF)
    assert token.score is None


def test_alternate_key_names_are_accepted() -> None:
    data = {"texts": ["OMNICEF"], "scores": [0.9], "boxes": [[1, 2, 3, 4]]}
    (token,) = _tokens_from_data(data, 0, MIN_CONF)
    assert token.text == "OMNICEF"


# --------------------------------------------------------------------------- #
# PaddleOCR 2.x list output
# --------------------------------------------------------------------------- #
def test_tokens_from_legacy_result() -> None:
    result = [
        [[[10, 100], [70, 100], [70, 120], [10, 120]], ("303616", 0.99)],
        [[[90, 100], [200, 100], [200, 120], [90, 120]], ("OMNICEF", 0.97)],
    ]
    tokens = _tokens_from_legacy(result, 0, MIN_CONF)
    assert [token.text for token in tokens] == ["303616", "OMNICEF"]
    assert tokens[0].box == (10.0, 100.0, 70.0, 120.0)


def test_legacy_result_wrapped_in_an_outer_list() -> None:
    """PaddleOCR 2.x returns one entry per image, even for a single page."""
    inner = [[[[10, 100], [70, 100], [70, 120], [10, 120]], ("303616", 0.99)]]
    assert len(_tokens_from_legacy([inner[0]], 0, MIN_CONF)) == 1


@pytest.mark.parametrize("result", [None, [], "unexpected", [["malformed"]], [[None, None]]])
def test_legacy_parser_survives_garbage(result: Any) -> None:
    assert _tokens_from_legacy(result, 0, MIN_CONF) == []


# --------------------------------------------------------------------------- #
# Result unwrapping
# --------------------------------------------------------------------------- #
def test_result_to_dict_passes_through_a_mapping() -> None:
    assert _result_to_dict({"rec_texts": ["A"]}) == {"rec_texts": ["A"]}


def test_result_to_dict_calls_a_json_method() -> None:
    class Result:
        def json(self) -> str:
            return json.dumps({"rec_texts": ["A"]})

    assert _result_to_dict(Result()) == {"rec_texts": ["A"]}


def test_result_to_dict_reads_attributes_as_a_last_resort() -> None:
    class Result:
        rec_texts = ["A"]
        rec_scores = [0.9]

    assert _result_to_dict(Result()) == {"rec_texts": ["A"], "rec_scores": [0.9]}


def test_result_to_dict_converts_numpy_like_values() -> None:
    class Array:
        def tolist(self) -> list[int]:
            return [1, 2, 3]

    assert _result_to_dict({"rec_boxes": Array()}) == {"rec_boxes": [1, 2, 3]}


def test_box_from_empty_polygon_is_none() -> None:
    assert _box_from_poly([]) is None
    assert _box_from_poly(None) is None


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("configured", ["cpu", "gpu", "gpu:1"])
def test_explicit_device_is_passed_through(configured: str) -> None:
    from pharmacy_ocr.ocr.engine import resolve_device

    assert resolve_device(configured) == configured


def test_auto_falls_back_to_cpu_without_a_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU-only wheel raises if asked for 'gpu', so auto must not guess."""
    import paddle

    from pharmacy_ocr.ocr.engine import resolve_device

    monkeypatch.setattr(paddle.device, "is_compiled_with_cuda", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_auto_selects_gpu_when_cuda_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import paddle

    from pharmacy_ocr.ocr.engine import resolve_device

    monkeypatch.setattr(paddle.device, "is_compiled_with_cuda", lambda: True)
    monkeypatch.setattr(paddle.device.cuda, "device_count", lambda: 1)
    assert resolve_device("auto") == "gpu"


def test_auto_falls_back_when_the_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import paddle

    from pharmacy_ocr.ocr.engine import resolve_device

    def boom() -> bool:
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(paddle.device, "is_compiled_with_cuda", boom)
    assert resolve_device("auto") == "cpu"


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #
def test_parses_a_clean_json_object() -> None:
    items = parse_items('{"items":[{"MED":"OMNICEF","QT":1,"prix":5.32}]}')
    assert items == [{"MED": "OMNICEF", "QT": 1, "prix": 5.32}]


def test_parses_through_markdown_fences() -> None:
    content = '```json\n{"items":[{"MED":"OMNICEF"}]}\n```'
    assert parse_items(content) == [{"MED": "OMNICEF"}]


def test_parses_json_embedded_in_prose() -> None:
    content = 'Sure! Here you go:\n{"items":[{"MED":"OMNICEF"}]}\nHope that helps.'
    assert parse_items(content) == [{"MED": "OMNICEF"}]


def test_accepts_a_bare_array() -> None:
    assert parse_items('[{"MED":"OMNICEF"}]') == [{"MED": "OMNICEF"}]


def test_non_dict_entries_are_filtered_out() -> None:
    assert parse_items('{"items":[{"MED":"A"}, "junk", 42]}') == [{"MED": "A"}]


@pytest.mark.parametrize("content", ["", "no json here", "{ truncated", "null"])
def test_unparseable_content_returns_none(content: str) -> None:
    assert parse_items(content) is None


def test_rows_text_is_numbered_for_the_prompt() -> None:
    page = OCRPage(text="303616 OMNICEF 34.680\nTotal 34.680")
    assert rows_text_from_pages([page]) == "1. 303616 OMNICEF 34.680\n2. Total 34.680"


def test_rows_text_of_an_empty_page() -> None:
    assert rows_text_from_pages([OCRPage(text="")]) == ""


# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #
def _record(**extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="pharmacy_ocr.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="invoice %s complete",
        args=("a.pdf",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_object_per_line() -> None:
    output = JsonFormatter().format(_record())
    payload = json.loads(output)

    assert "\n" not in output
    assert payload["message"] == "invoice a.pdf complete"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "pharmacy_ocr.test"
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_inlines_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record(items=3, checksum_ok=True)))
    assert payload["items"] == 3
    assert payload["checksum_ok"] is True


def test_json_formatter_stringifies_unserializable_values() -> None:
    payload = json.loads(JsonFormatter().format(_record(engine=object())))
    assert isinstance(payload["engine"], str)


def test_json_formatter_includes_the_exception_text() -> None:
    try:
        raise ValueError("model weights missing")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))
    assert "model weights missing" in payload["exception"]


def test_request_id_binding_is_scoped() -> None:
    from pharmacy_ocr.observability.logging import current_request_id

    assert current_request_id() is None
    with bind_request_id("abc123"):
        assert current_request_id() == "abc123"
        with bind_request_id("nested"):
            assert current_request_id() == "nested"
        assert current_request_id() == "abc123"
    assert current_request_id() is None
