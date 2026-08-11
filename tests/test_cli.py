"""CLI: argument wiring, batch behaviour, audit reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmacy_ocr.cli import commands
from pharmacy_ocr.cli.main import build_parser
from pharmacy_ocr.settings import ExtractionMode, Settings
from tests.conftest import PNG_MAGIC, FakeOCREngine, FakePreprocessor


@pytest.fixture
def invoices_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "invoices"
    directory.mkdir()
    for index in range(3):
        (directory / f"invoice{index}.png").write_bytes(PNG_MAGIC + f"payload{index}".encode())
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI build pipelines backed by the fake engine."""
    real_init = commands.InvoicePipeline.__init__

    def patched(self, settings, **kwargs):
        kwargs.setdefault("engine", FakeOCREngine())
        kwargs.setdefault("preprocessor", FakePreprocessor())
        real_init(self, settings, **kwargs)

    monkeypatch.setattr(commands.InvoicePipeline, "__init__", patched)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def test_batch_defaults() -> None:
    args = build_parser().parse_args(["batch"])
    assert args.command == "batch"
    assert args.photo_mode is None
    assert args.force is False


def test_photo_and_scan_modes_are_mutually_exclusive() -> None:
    assert build_parser().parse_args(["batch", "--photo-mode"]).photo_mode is True
    assert build_parser().parse_args(["batch", "--scan-mode"]).photo_mode is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["batch", "--photo-mode", "--scan-mode"])


def test_mode_is_validated() -> None:
    args = build_parser().parse_args(["batch", "--mode", "deterministic"])
    assert args.mode is ExtractionMode.deterministic
    with pytest.raises(SystemExit):
        build_parser().parse_args(["batch", "--mode", "nonsense"])


def test_cache_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["cache"])
    assert build_parser().parse_args(["cache", "prune"]).cache_command == "prune"


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
async def test_batch_writes_one_json_per_invoice(
    settings: Settings, invoices_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    code = await commands.run_batch(
        settings,
        input_dir=invoices_dir,
        output_dir=output_dir,
        concurrency=2,
        photo_mode=None,
        extraction_mode=ExtractionMode.deterministic,
        limit=None,
        force=False,
    )

    assert code == commands.EXIT_OK
    outputs = sorted(path.name for path in output_dir.glob("*.json"))
    assert outputs == ["_batch_summary.json", "invoice0.json", "invoice1.json", "invoice2.json"]

    body = json.loads((output_dir / "invoice0.json").read_text(encoding="utf-8"))
    assert body["status"] == "ok"
    assert len(body["items"]) == 3

    summary = json.loads((output_dir / "_batch_summary.json").read_text(encoding="utf-8"))
    assert summary["invoices"] == 3
    assert summary["ok"] == 3


async def test_batch_skips_existing_output_unless_forced(
    settings: Settings, invoices_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    kwargs = {
        "input_dir": invoices_dir,
        "output_dir": output_dir,
        "concurrency": 1,
        "photo_mode": None,
        "extraction_mode": ExtractionMode.deterministic,
        "limit": None,
    }
    await commands.run_batch(settings, force=False, **kwargs)
    marker = output_dir / "invoice0.json"
    marker.write_text('{"marker": true}', encoding="utf-8")

    await commands.run_batch(settings, force=False, **kwargs)
    assert json.loads(marker.read_text(encoding="utf-8")) == {"marker": True}

    await commands.run_batch(settings, force=True, **kwargs)
    assert "items" in json.loads(marker.read_text(encoding="utf-8"))


async def test_batch_honours_limit(settings: Settings, invoices_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    await commands.run_batch(
        settings,
        input_dir=invoices_dir,
        output_dir=output_dir,
        concurrency=1,
        photo_mode=None,
        extraction_mode=None,
        limit=1,
        force=False,
    )
    assert len(list(output_dir.glob("invoice*.json"))) == 1


async def test_batch_reports_missing_input(settings: Settings, tmp_path: Path) -> None:
    code = await commands.run_batch(
        settings,
        input_dir=tmp_path / "absent",
        output_dir=tmp_path / "out",
        concurrency=1,
        photo_mode=None,
        extraction_mode=None,
        limit=None,
        force=False,
    )
    assert code == commands.EXIT_NO_INPUT


def test_collect_files_ignores_unsupported_types(invoices_dir: Path) -> None:
    names = [path.name for path in commands.collect_files(invoices_dir)]
    assert names == ["invoice0.png", "invoice1.png", "invoice2.png"]


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_audit_summarizes_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "good.json").write_text(
        json.dumps({"status": "ok", "items": [{"prix": 5.0}], "checksum_ok": True}), encoding="utf-8"
    )
    (output_dir / "bad.json").write_text(
        json.dumps({"status": "needs_review", "items": [], "checksum_ok": False, "warnings": ["x"]}),
        encoding="utf-8",
    )

    assert commands.run_audit(output_dir) == commands.EXIT_OK
    output = capsys.readouterr().out
    assert "files=2" in output
    assert "bad.json" in output
    assert "good.json" not in output, "clean invoices are not listed unless --all"


def test_audit_handles_corrupt_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "broken.json").write_text("{ not json", encoding="utf-8")

    assert commands.run_audit(output_dir) == commands.EXIT_OK
    assert "not valid JSON" in capsys.readouterr().out


def test_audit_reports_empty_folder(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert commands.run_audit(empty) == commands.EXIT_NO_INPUT


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_redacts_api_keys(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    secured = settings.model_copy(update={"api_keys_raw": "supersecret"})
    assert commands.run_show_config(secured) == commands.EXIT_OK

    output = capsys.readouterr().out
    assert "supersecret" not in output
    assert "1 key(s) configured" in output
    assert json.loads(output)["resolved_paths"]["ocr_cache_dir"]
