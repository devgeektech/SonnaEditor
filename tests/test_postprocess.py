"""Tests for sonna_editor.model.postprocess.

Includes regression coverage for the v1/v2 broadcast mismatch — see commit
that introduced this test for the breakage details. The full end-to-end
``InferenceEngine.predict`` path on a v1 checkpoint is exercised here to
close the gap that let the regression land undetected.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from sonna_editor import config
from sonna_editor.inference.engine import InferenceEngine
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor
from sonna_editor.model.postprocess import postprocess_predictions, predictions_to_dict


_TEMPERATURE_IDX: int = config.SLIDER_FIELDS.index("Temperature")


def _make_v1_ckpt(tmp_path: Path) -> Path:
    """Save a v1 SonnaEditor (135-output) ckpt for engine round-trip tests."""
    reg = EmbeddingRegistry()
    reg.camera_makes    = {"unknown": 0}
    reg.camera_models   = {"unknown": 0}
    reg.lenses          = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets      = {"unknown": 0}
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={
            "num_makes": 4, "num_models": 4, "num_lenses": 4,
            "num_profiles": 4, "num_wb_presets": 4,
        },
        _pretrained_backbone=False,
        arch_version=1,
        slider_set_version="v1",
    )
    path = tmp_path / "v1.ckpt"
    model.save_checkpoint(path)
    return path


def _make_v2_ckpt(tmp_path: Path) -> Path:
    """Save a v2 SonnaEditor (147-output) ckpt for engine round-trip tests."""
    reg = EmbeddingRegistry()
    reg.camera_makes    = {"unknown": 0}
    reg.camera_models   = {"unknown": 0}
    reg.lenses          = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets      = {"unknown": 0}
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={
            "num_makes": 4, "num_models": 4, "num_lenses": 4,
            "num_profiles": 4, "num_wb_presets": 4,
        },
        _pretrained_backbone=False,
        arch_version=1,
        slider_set_version="v2",
    )
    path = tmp_path / "v2.ckpt"
    model.save_checkpoint(path)
    return path


def _dummy_metadata() -> dict:
    return {
        "camera_make": "unknown", "camera_model": "unknown",
        "lens_model": "unknown", "camera_profile": "unknown",
        "white_balance_preset": "unknown",
        "iso": 100.0, "shutter_speed": 0.008, "aperture": 5.6,
        "focal_length": 50.0,
        "histogram": [[1.0 / 32] * 32] * 3,
        "as_shot_temperature": 5500.0, "as_shot_tint": 0.0,
    }


# ---------------------------------------------------------------------------
# Direct postprocess_predictions shape tests
# ---------------------------------------------------------------------------

def test_postprocess_predictions_accepts_v1_shape() -> None:
    """[B, 135] (v1 model output) must pass through without broadcast error."""
    raw = torch.zeros(2, 135)
    raw[:, _TEMPERATURE_IDX] = math.log(5500.0)
    out = postprocess_predictions(raw)
    assert out.shape == (2, 135)
    # Temperature converted back to Kelvin (exp), then clamp is a no-op for 5500.
    assert out[0, _TEMPERATURE_IDX].item() == pytest.approx(5500.0, rel=1e-4)
    assert out[1, _TEMPERATURE_IDX].item() == pytest.approx(5500.0, rel=1e-4)


def test_postprocess_predictions_accepts_v2_shape() -> None:
    """[B, 147] (v2 model output) preserves existing behaviour."""
    raw = torch.zeros(2, 147)
    raw[:, _TEMPERATURE_IDX] = math.log(5500.0)
    out = postprocess_predictions(raw)
    assert out.shape == (2, 147)
    assert out[0, _TEMPERATURE_IDX].item() == pytest.approx(5500.0, rel=1e-4)


def test_postprocess_predictions_clamps_to_valid_range_v1() -> None:
    """Out-of-range values clamp to slider bounds for v1 output too."""
    raw = torch.zeros(1, 135)
    raw[0, _TEMPERATURE_IDX] = math.log(5500.0)
    # Exposure2012 is idx 0, range (-5, 5). Push out of range.
    raw[0, 0] = 100.0
    out = postprocess_predictions(raw)
    assert out[0, 0].item() == 5.0


def test_postprocess_predictions_rejects_oversize_input() -> None:
    """Prediction tensor wider than SLIDER_FIELDS must raise, not silently truncate."""
    raw = torch.zeros(1, len(config.SLIDER_FIELDS) + 1)
    raw[0, _TEMPERATURE_IDX] = math.log(5500.0)
    with pytest.raises(ValueError, match="only defines"):
        postprocess_predictions(raw)


# ---------------------------------------------------------------------------
# End-to-end regression test — closes the gap that let the v2-expansion
# regression land. Loads a real v1 SonnaEditor checkpoint and runs the
# full InferenceEngine.predict() path on a synthetic image.
# ---------------------------------------------------------------------------

def test_engine_predict_end_to_end_v1_checkpoint(tmp_path: Path) -> None:
    """v1 ckpt → InferenceEngine.predict() must not raise.

    Regression coverage: between v2 slider expansion (commit 3d0d90c) and
    the postprocess fix that introduced this test, every production
    inference call on a v1 ckpt crashed at postprocess_predictions because
    its range tensors were length 147 while the model output was length 135.
    Full pytest passed because no test exercised this end-to-end path.
    """
    ckpt = _make_v1_ckpt(tmp_path)
    engine = InferenceEngine(ckpt, device="cpu")
    assert engine._model._slider_set_version == "v1"

    img = Image.new("RGB", (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION),
                    (128, 128, 128))
    preds = engine.predict([img], [_dummy_metadata()], batch_size=1)
    assert preds.shape == (1, 135)
    # Sanity: Temperature column is in Kelvin (positive, finite, within LR range).
    t = float(preds[0, _TEMPERATURE_IDX].item())
    assert 2000.0 <= t <= 50000.0, f"Temperature out of range: {t}"


def test_engine_predict_end_to_end_v2_checkpoint(tmp_path: Path) -> None:
    """v2 ckpt → InferenceEngine.predict() preserves the working path."""
    ckpt = _make_v2_ckpt(tmp_path)
    engine = InferenceEngine(ckpt, device="cpu")
    assert engine._model._slider_set_version == "v2"

    img = Image.new("RGB", (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION),
                    (128, 128, 128))
    preds = engine.predict([img], [_dummy_metadata()], batch_size=1)
    assert preds.shape == (1, 147)
    t = float(preds[0, _TEMPERATURE_IDX].item())
    assert 2000.0 <= t <= 50000.0, f"Temperature out of range: {t}"


# ---------------------------------------------------------------------------
# predictions_to_dict — v1/v2 shape coverage
# ---------------------------------------------------------------------------
# This is the second shape-mismatch site exposed by the v2 SLIDER_FIELDS
# expansion (commit 3d0d90c). Before migration to fields_matching_tensor,
# predictions_to_dict iterated config.SLIDER_FIELDS (length 147) and indexed
# row[i] for i=0..146, crashing with IndexError at i=135 when the input was a
# v1 prediction tensor (length 135). Tests below pin both v1 and v2 paths.

def test_predictions_to_dict_v1_shape_returns_135_keys() -> None:
    preds = torch.zeros(2, 135)
    preds[:, _TEMPERATURE_IDX] = math.log(5500.0)
    d = predictions_to_dict(preds, batch_idx=0)
    assert len(d) == 135
    assert "Exposure2012" in d
    assert "ToneCurveBlue_Pt6_Y" in d   # last v1 field
    assert "CurveRefineSaturation" not in d   # v2 extension


def test_predictions_to_dict_v2_shape_returns_147_keys() -> None:
    preds = torch.zeros(2, 147)
    preds[:, _TEMPERATURE_IDX] = math.log(5500.0)
    d = predictions_to_dict(preds, batch_idx=0)
    assert len(d) == 147
    assert "CurveRefineSaturation" in d   # v2 extension present


def test_predictions_to_dict_values_match_tensor() -> None:
    preds = torch.zeros(1, 135)
    preds[0, 0] = 0.42   # Exposure2012
    preds[0, 11] = math.log(5500.0)   # Temperature (log-space)
    d = predictions_to_dict(preds, batch_idx=0)
    assert d["Exposure2012"] == pytest.approx(0.42)
    assert d["Temperature"] == pytest.approx(math.log(5500.0))


def test_predictions_to_dict_batch_idx_selects_row() -> None:
    preds = torch.zeros(3, 135)
    preds[0, 0] = 0.10
    preds[1, 0] = 0.20
    preds[2, 0] = 0.30
    assert predictions_to_dict(preds, batch_idx=0)["Exposure2012"] == pytest.approx(0.10)
    assert predictions_to_dict(preds, batch_idx=1)["Exposure2012"] == pytest.approx(0.20)
    assert predictions_to_dict(preds, batch_idx=2)["Exposure2012"] == pytest.approx(0.30)


def test_predictions_to_dict_rejects_unsupported_length() -> None:
    """Tensor with non-v1/non-v2 last dim must fail loudly (caller bug)."""
    preds = torch.zeros(2, 130)
    with pytest.raises(ValueError, match="not a supported"):
        predictions_to_dict(preds, batch_idx=0)
