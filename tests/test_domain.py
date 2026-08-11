"""Settings validation and invoice model normalization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pharmacy_ocr.domain.models import InvoiceItem, InvoiceResult, ReviewStatus, clean_refcode
from pharmacy_ocr.settings import ExtractionMode, Settings


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def test_production_refuses_to_start_without_api_keys() -> None:
    with pytest.raises(ValidationError, match="API_KEYS"):
        Settings(environment="production", api_keys_raw="")


def test_production_allows_explicit_gateway_optout() -> None:
    settings = Settings(environment="production", api_keys_raw="", allow_unauthenticated=True)
    assert settings.auth_enabled is False


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="CORS"):
        Settings(environment="production", api_keys_raw="k", cors_origins_raw="*")


def test_production_rejects_debug() -> None:
    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(environment="production", api_keys_raw="k", debug=True)


def test_api_keys_and_cors_parse_from_comma_lists() -> None:
    settings = Settings(
        environment="development",
        api_keys_raw=" k1 , k2 ,, ",
        cors_origins_raw="https://a.tn, https://b.tn",
    )
    assert settings.api_keys == frozenset({"k1", "k2"})
    assert settings.cors_origins == ("https://a.tn", "https://b.tn")


def test_adaptive_block_size_must_be_odd() -> None:
    with pytest.raises(ValidationError, match="odd"):
        Settings(environment="development", adaptive_block_size=30)


def test_long_side_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="min_long_side"):
        Settings(environment="development", min_long_side=4000, max_long_side=1000)


def test_directories_derive_from_data_dir(tmp_path) -> None:
    settings = Settings(environment="development", data_dir=tmp_path / "d")
    assert settings.invoices_dir == tmp_path / "d" / "invoices"
    assert settings.ocr_cache_dir == tmp_path / "d" / "ocr_cache"


def test_public_snapshot_excludes_secrets() -> None:
    settings = Settings(environment="development", api_keys_raw="supersecret")
    snapshot = settings.public_snapshot()
    assert "supersecret" not in str(snapshot)
    assert snapshot["auth_enabled"] is True


def test_settings_are_immutable(settings: Settings) -> None:
    with pytest.raises(ValidationError):
        settings.extraction_mode = ExtractionMode.llm  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# InvoiceItem
# --------------------------------------------------------------------------- #
def test_accepts_alternate_key_names() -> None:
    item = InvoiceItem.model_validate(
        {"code": "0303154", "designation": " COPRED  ODT ", "qte": "1", "montant": "5,320"}
    )
    assert item.REFCODE == "0303154"
    assert item.MED == "COPRED ODT"
    assert item.QT == 1
    assert item.prix == 5.320


def test_decimal_quantity_is_swapped_into_price() -> None:
    """A decimal is never a quantity; it is a misplaced price."""
    item = InvoiceItem.model_validate({"MED": "X", "QT": 2.5, "prix": 3})
    assert item.QT == 3
    assert item.prix == 2.5


def test_decimal_quantity_without_price_becomes_the_price() -> None:
    item = InvoiceItem.model_validate({"MED": "X", "QT": 5.320})
    assert item.QT is None
    assert item.prix == 5.320


def test_line_total_derived_from_unit_price() -> None:
    item = InvoiceItem.model_validate({"MED": "OMNICEF", "QT": 2, "pu": "17.340"})
    assert item.prix == 34.680
    assert item.unit_price == 17.340
    assert item.line_total_matches_unit is True


def test_short_refcode_is_dropped() -> None:
    assert clean_refcode("12") is None
    assert clean_refcode("30-31 54") == "303154"
    assert clean_refcode("null") is None


def test_empty_medicine_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InvoiceItem.model_validate({"MED": "   "})


def test_unknown_keys_are_ignored() -> None:
    item = InvoiceItem.model_validate({"MED": "X", "prix": 1.0, "surprise": "value"})
    assert item.MED == "X"


# --------------------------------------------------------------------------- #
# InvoiceResult
# --------------------------------------------------------------------------- #
def test_status_is_ok_only_when_everything_agrees() -> None:
    item = InvoiceItem.model_validate({"MED": "X", "prix": 1.0})
    assert (
        InvoiceResult.derive_status(error=None, items=[item], checksum_ok=True, warnings=[])
        is ReviewStatus.ok
    )


@pytest.mark.parametrize(
    ("items", "checksum", "warnings"),
    [
        ([], True, []),
        ([1], False, []),
        ([1], None, []),
        ([1], True, ["something looked odd"]),
    ],
)
def test_status_falls_back_to_needs_review(items, checksum, warnings) -> None:
    assert (
        InvoiceResult.derive_status(error=None, items=items, checksum_ok=checksum, warnings=warnings)
        is ReviewStatus.needs_review
    )


def test_status_is_failed_when_an_error_is_present() -> None:
    assert (
        InvoiceResult.derive_status(error="boom", items=[], checksum_ok=True, warnings=[])
        is ReviewStatus.failed
    )
