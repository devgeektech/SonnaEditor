from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.preset.pipeline import process_shoot

FIXTURE_PRESET = Path(__file__).parent / "fixtures" / "preset_sonna_v1.xmp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_files(tmp_path: Path, n: int = 3, ext: str = ".cr3") -> list[Path]:
    raws = []
    for i in range(n):
        p = tmp_path / f"photo_{i:03d}{ext}"
        p.touch()
        raws.append(p)
    return raws


def _fake_extract(raw_path, xmp_path=None):
    img = Image.new("RGB", (64, 64), color=(118, 118, 118))  # mid-grey
    hist = np.zeros((3, 32), dtype=np.float32)
    sliders = {f: 0.0 for f in SLIDER_FIELDS}
    return {
        "raw_path": str(raw_path),
        "xmp_path": None,
        "preview": img,
        "histogram": hist,
        "iso": 400,
        "shutter_speed": 1 / 125,
        "aperture": 2.8,
        "focal_length": 50.0,
        "lens_model": "RF 50mm",
        "camera_body": "Canon EOS R6",
        "capture_datetime": None,
        "exposure_compensation": 0.0,
        "white_balance_preset": "Auto",
        "camera_profile": "Adobe Standard",
        "width": 64,
        "height": 64,
        "sliders": sliders,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_shoot_returns_summary(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=3)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)
    assert "processed" in summary
    assert "failed" in summary
    assert "output_paths" in summary
    assert "failures" in summary


def test_process_shoot_processes_all_photos(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=4)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)
    assert summary["processed"] == 4
    assert summary["failed"] == 0


def test_process_shoot_writes_xmps_next_to_raws(tmp_path: Path) -> None:
    raws = _make_raw_files(tmp_path, n=2)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)
    for raw in raws:
        xmp = raw.with_suffix(".xmp")
        assert xmp.exists(), f"Expected XMP at {xmp}"


def test_process_shoot_writes_xmps_to_output_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "xmps"
    _make_raw_files(input_dir, n=2)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(input_dir, output_dir, FIXTURE_PRESET, {}, max_workers=1)
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.xmp"))) == 2


def test_process_shoot_dry_run_no_xmps_written(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=3)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1, dry_run=True)
    assert summary["processed"] == 3
    # No XMP files should exist
    assert list(tmp_path.glob("*.xmp")) == []


def test_process_shoot_handles_extraction_failure(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=3)

    call_count = 0
    def flaky_extract(raw_path, xmp_path=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("extraction failed")
        return _fake_extract(raw_path, xmp_path)

    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=flaky_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)

    assert summary["processed"] == 2
    assert summary["failed"] == 1
    assert len(summary["failures"]) == 1


def test_process_shoot_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No RAW files"):
        process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)


def test_process_shoot_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "2024" / "march"
    sub.mkdir(parents=True)
    _make_raw_files(sub, n=2)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)
    assert summary["processed"] == 2


def test_process_shoot_output_paths_count(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=5)
    with patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract):
        summary = process_shoot(tmp_path, None, FIXTURE_PRESET, {}, max_workers=1)
    assert len(summary["output_paths"]) == 5


def test_process_shoot_options_forwarded(tmp_path: Path) -> None:
    _make_raw_files(tmp_path, n=1)
    captured_options = {}

    def mock_compute(image, metadata, base_preset, options):
        captured_options.update(options)
        return {}

    with (
        patch("sonna_editor.preset.pipeline.extract_all", side_effect=_fake_extract),
        patch("sonna_editor.preset.pipeline.compute_adjustment", side_effect=mock_compute),
    ):
        process_shoot(
            tmp_path, None, FIXTURE_PRESET,
            {"auto_exposure": False, "auto_white_balance": True},
            max_workers=1,
        )

    assert captured_options.get("auto_exposure") is False
    assert captured_options.get("auto_white_balance") is True


@pytest.mark.integration
def test_process_shoot_real_photo(tmp_path: Path) -> None:
    """End-to-end with a real CR3 — requires sample.cr3 fixture."""
    fixture_cr3 = Path(__file__).parent / "fixtures" / "sample.cr3"
    if not fixture_cr3.exists():
        pytest.skip("Real CR3 fixture not present")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.cr3").symlink_to(fixture_cr3)

    summary = process_shoot(
        input_dir=input_dir,
        output_dir=tmp_path / "xmps",
        preset_path=FIXTURE_PRESET,
        options={},
        max_workers=1,
    )

    assert summary["processed"] == 1
    assert summary["failed"] == 0
    xmp_files = list((tmp_path / "xmps").glob("*.xmp"))
    assert len(xmp_files) == 1
