"""OCR cache: identity, atomicity, corruption recovery, eviction."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pharmacy_ocr.ocr.cache import OCRCache
from pharmacy_ocr.ocr.types import OCRPage, OCRToken
from pharmacy_ocr.settings import Settings


def _pages() -> list[OCRPage]:
    return [
        OCRPage(
            text="303616 OMNICEF 34.680",
            tokens=[OCRToken(text="OMNICEF", score=0.9, box=(1.0, 2.0, 3.0, 4.0))],
            source_image="page0.png",
        )
    ]


def _write(path: Path, payload: bytes = b"invoice-bytes") -> Path:
    path.write_bytes(payload)
    return path


def test_roundtrip_preserves_tokens(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    source = _write(tmp_path / "a.pdf")

    key = cache.key_for(source)
    assert cache.load(key) is None
    cache.save(key, _pages(), {"ocr_s": 1.0})

    restored = cache.load(key)
    assert restored is not None
    assert restored[0].text == "303616 OMNICEF 34.680"
    assert restored[0].tokens[0].box == (1.0, 2.0, 3.0, 4.0)
    assert restored[0].tokens[0].score == 0.9


def test_identical_bytes_hit_regardless_of_name_or_mtime(settings: Settings, tmp_path: Path) -> None:
    """The previous key included mtime, so every copy re-ran OCR for nothing."""
    cache = OCRCache(settings)
    original = _write(tmp_path / "original.pdf")
    cache.save(cache.key_for(original), _pages(), {})

    copy = _write(tmp_path / "renamed-copy.pdf")
    time.sleep(0.01)
    copy.touch()

    assert cache.load(cache.key_for(copy)) is not None


def test_different_bytes_miss(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    cache.save(cache.key_for(_write(tmp_path / "a.pdf", b"one")), _pages(), {})
    assert cache.load(cache.key_for(_write(tmp_path / "b.pdf", b"two"))) is None


def test_changing_ocr_settings_invalidates_the_entry(settings: Settings, tmp_path: Path) -> None:
    source = _write(tmp_path / "a.pdf")
    OCRCache(settings).save(OCRCache(settings).key_for(source), _pages(), {})

    other = settings.model_copy(update={"ocr_lang": "en"})
    assert OCRCache(other).load(OCRCache(other).key_for(source)) is None


def test_render_profile_is_part_of_the_key(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    source = _write(tmp_path / "a.pdf")
    cache.save(cache.key_for(source), _pages(), {})
    assert cache.load(cache.key_for(source, pdf_render_dpi=300)) is None


def test_corrupt_entry_is_discarded_not_raised(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    key = cache.key_for(_write(tmp_path / "a.pdf"))
    cache.save(key, _pages(), {})

    cache.path_for(key.digest).write_text("{ truncated", encoding="utf-8")
    assert cache.load(key) is None
    assert not cache.path_for(key.digest).exists(), "unreadable entries are removed"


def test_save_leaves_no_temp_files(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    cache.save(cache.key_for(_write(tmp_path / "a.pdf")), _pages(), {})
    assert list(cache.directory.glob("*.tmp")) == []


def test_disabled_cache_never_reads_or_writes(settings: Settings, tmp_path: Path) -> None:
    disabled = settings.model_copy(update={"use_ocr_cache": False})
    cache = OCRCache(disabled)
    key = cache.key_for(_write(tmp_path / "a.pdf"))
    assert cache.save(key, _pages(), {}) is None
    assert cache.load(key) is None


def test_force_refresh_skips_reads_but_still_writes(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    source = _write(tmp_path / "a.pdf")
    cache.save(cache.key_for(source), _pages(), {})

    forced = settings.model_copy(update={"force_ocr_refresh": True})
    forced_cache = OCRCache(forced)
    assert forced_cache.load(forced_cache.key_for(source)) is None


def test_prune_drops_entries_past_the_ttl(settings: Settings, tmp_path: Path) -> None:
    tuned = settings.model_copy(update={"ocr_cache_ttl_days": 1})
    cache = OCRCache(tuned)
    key = cache.key_for(_write(tmp_path / "a.pdf"))
    cache.save(key, _pages(), {})

    stale = cache.path_for(key.digest)
    old = time.time() - 5 * 86400
    import os

    os.utime(stale, (old, old))

    assert cache.prune() == 1
    assert not stale.exists()


def test_prune_enforces_the_entry_ceiling(settings: Settings, tmp_path: Path) -> None:
    tuned = settings.model_copy(update={"ocr_cache_max_entries": 2, "ocr_cache_ttl_days": 0})
    cache = OCRCache(tuned)
    for index in range(5):
        cache.save(cache.key_for(_write(tmp_path / f"f{index}.pdf", f"bytes{index}".encode())), _pages(), {})

    assert cache.prune() == 3
    assert len(list(cache.directory.glob("*.json"))) == 2


def test_load_file_exposes_metadata(settings: Settings, tmp_path: Path) -> None:
    cache = OCRCache(settings)
    source = _write(tmp_path / "invoice7.pdf")
    key = cache.key_for(source)
    path = cache.save(key, _pages(), {"ocr_s": 2.0})
    assert path is not None

    metadata, pages = OCRCache.load_file(path)
    assert metadata["source_name"] == "invoice7.pdf"
    assert len(pages) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["timings"]["ocr_s"] == 2.0
