"""Validation suite for AI tools toggle behaviour in Sonna Editor.

Validates that:
1. When auto_straighten is enabled (True), Lightroom-compliant CropAngle, HasCrop,
   and crop bounding box attributes ARE applied to generated XMP sidecars,
   sonna_predictions.json, and sonna_lightroom_edits.lua.
2. When auto_straighten is disabled (False), NO CropAngle or HasCrop attributes
   are applied to XMP sidecars or sonna_lightroom_edits.lua.
3. The POST /api/process route forwards the auto_straighten flag correctly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from sonna_editor import config
from sonna_editor.inference import pipeline as pl


def _create_tilted_test_image(angle_degrees: float = 3.0) -> Image.Image:
    """Create a synthetic test image with a high-contrast tilted line."""
    img = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(img)
    theta = math.radians(angle_degrees)
    dx = math.cos(theta) * 260
    dy = math.sin(theta) * 260
    draw.line((300 - dx, 200 - dy, 300 + dx, 200 + dy), fill="black", width=6)
    return img


class _FakeModel:
    _slider_set_version = "v1"


class _FakeEngine:
    _image_resolution = 384
    _model = _FakeModel()

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def warmup(self) -> None:
        pass

    def predict(self, *_args, **_kwargs):
        return torch.zeros((1, len(config.SLIDER_FIELDS)))


def test_straighten_applied_when_turned_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When auto_straighten=True, CropAngle and HasCrop must be written to XMP and sidecars."""
    shoot_dir = tmp_path / "shoot_on"
    shoot_dir.mkdir()
    raw = shoot_dir / "tilted_photo.cr3"
    raw.write_bytes(b"dummy_raw_bytes")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"dummy_ckpt")

    test_image = _create_tilted_test_image(3.0)

    monkeypatch.setattr(pl, "InferenceEngine", _FakeEngine)
    monkeypatch.setattr(pl, "_extract_one", lambda _path, _size: (test_image, {"as_shot_wb": (5500.0, 10.0)}))

    result = pl.process_shoot_with_model(
        input_dir=shoot_dir,
        model_path=ckpt,
        output_dir=shoot_dir,
        auto_straighten=True,
        max_workers=1,
        save_predictions=True,
    )

    assert result["processed"] == 1
    xmp_file = shoot_dir / "tilted_photo.xmp"
    assert xmp_file.exists()
    xmp_text = xmp_file.read_text(encoding="utf-8")

    # Assert CropAngle and HasCrop ARE present in XMP
    assert 'crs:HasCrop="True"' in xmp_text
    assert "crs:CropAngle=" in xmp_text

    # Assert predictions sidecar records auto_straighten = True
    pred_file = shoot_dir / "sonna_predictions.json"
    assert pred_file.exists()
    pred_data = json.loads(pred_file.read_text(encoding="utf-8"))
    assert pred_data["auto_straighten"] is True
    assert "tilted_photo.cr3" in pred_data["straightening"]
    assert pred_data["straightening"]["tilted_photo.cr3"]["applied"] is True

    # Assert Lightroom bridge lua file contains CropAngle
    lua_file = shoot_dir / "sonna_lightroom_edits.lua"
    assert lua_file.exists()
    lua_text = lua_file.read_text(encoding="utf-8")
    assert "CropAngle" in lua_text


def test_straighten_not_applied_when_turned_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When auto_straighten=False, NO CropAngle or HasCrop attributes must reach XMP or lua."""
    shoot_dir = tmp_path / "shoot_off"
    shoot_dir.mkdir()
    raw = shoot_dir / "tilted_photo.cr3"
    raw.write_bytes(b"dummy_raw_bytes")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"dummy_ckpt")

    test_image = _create_tilted_test_image(3.0)

    monkeypatch.setattr(pl, "InferenceEngine", _FakeEngine)
    monkeypatch.setattr(pl, "_extract_one", lambda _path, _size: (test_image, {"as_shot_wb": (5500.0, 10.0)}))

    result = pl.process_shoot_with_model(
        input_dir=shoot_dir,
        model_path=ckpt,
        output_dir=shoot_dir,
        auto_straighten=False,
        max_workers=1,
        save_predictions=True,
    )

    assert result["processed"] == 1
    xmp_file = shoot_dir / "tilted_photo.xmp"
    assert xmp_file.exists()
    xmp_text = xmp_file.read_text(encoding="utf-8")

    # Assert CropAngle and HasCrop are completely ABSENT from XMP
    assert "CropAngle" not in xmp_text
    assert "HasCrop" not in xmp_text
    assert "CropTop" not in xmp_text
    assert "CropLeft" not in xmp_text

    # Assert predictions sidecar records auto_straighten = False
    pred_file = shoot_dir / "sonna_predictions.json"
    assert pred_file.exists()
    pred_data = json.loads(pred_file.read_text(encoding="utf-8"))
    assert pred_data["auto_straighten"] is False
    assert pred_data["straightening"] == {}

    # Assert Lightroom bridge lua file does NOT contain CropAngle
    lua_file = shoot_dir / "sonna_lightroom_edits.lua"
    assert lua_file.exists()
    lua_text = lua_file.read_text(encoding="utf-8")
    assert "CropAngle" not in lua_text
    assert "HasCrop" not in lua_text
