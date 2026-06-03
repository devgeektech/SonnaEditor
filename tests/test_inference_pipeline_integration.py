"""End-to-end integration test for the v1 production inference pipeline.

This test closes the structural test-coverage gap that allowed today's two
shape-mismatch bugs (postprocess_predictions broadcast crash in commit
4133e8a; predictions_to_dict IndexError in commit 8350903) to ship
undetected after the v2 SLIDER_FIELDS expansion (commit 3d0d90c).

Before this file existed, no test exercised a v1 checkpoint end-to-end
through the production pipeline:

    InferenceEngine.predict()
      -> model.forward()                   (raw log-space output)
      -> postprocess_predictions()         (range clamp + exp Temperature)
    predictions_to_dict()                  (tensor -> {field: value})
    WeightedSliderLoss.{forward, direction_stats, per_field_mae}
                                            (loss-side analysis)
    write_xmp()                            (final XMP on disk)
    read_xmp()                             (roundtrip verification)

A v1 ckpt running through every step is the verification gate for the
slider_set helper refactor. If this test passes, production v1.2.3
inference is unblocked end-to-end.

The fixture is a synthetic minimal v1 SonnaEditor (untrained,
``_pretrained_backbone=False``) so the test runs in seconds on CPU rather
than requiring the 250 MB production checkpoint. Untrained weights are
fine because we're testing PIPELINE shape compatibility, not prediction
quality.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image

from sonna_editor import config
from sonna_editor.data.xmp import read_xmp, write_xmp
from sonna_editor.inference.engine import InferenceEngine
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor
from sonna_editor.model.losses import WeightedSliderLoss
from sonna_editor.model.postprocess import predictions_to_dict


_TEMPERATURE_IDX: int = config.SLIDER_FIELDS.index("Temperature")


def _make_v1_ckpt(tmp_path: Path) -> Path:
    """Save a synthetic minimal v1 SonnaEditor ckpt."""
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


def _v1_targets_for_loss(B: int) -> torch.Tensor:
    """Synthetic targets sized for v1 (135), Temperature in raw Kelvin."""
    t = torch.full((B, 135), 0.0)
    t[:, _TEMPERATURE_IDX] = 5500.0
    return t


def _v1_loss_metadata(B: int) -> dict[str, torch.Tensor]:
    return {
        "as_shot_temperature": torch.full((B,), 5500.0),
        "as_shot_tint":        torch.full((B,), 0.0),
    }


def test_v1_production_pipeline_end_to_end(tmp_path: Path) -> None:
    """The integration gate. Every site patched today runs end-to-end on v1."""
    # ---- Setup ----
    ckpt = _make_v1_ckpt(tmp_path)
    engine = InferenceEngine(ckpt, device="cpu")
    assert engine._model._slider_set_version == "v1"

    img = Image.new(
        "RGB",
        (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION),
        (128, 128, 128),
    )
    metadata = _dummy_metadata()

    # ---- Step 1: engine.predict (engine._model + postprocess_predictions) ----
    # Before commit 4133e8a, this crashed at the range-tensor broadcast.
    preds = engine.predict([img], [metadata], batch_size=1)
    assert preds.shape == (1, 135), (
        f"engine.predict must return [B, 135] for v1; got {tuple(preds.shape)}"
    )
    assert torch.isfinite(preds).all(), "engine.predict produced non-finite values"
    temp_k = float(preds[0, _TEMPERATURE_IDX].item())
    assert 2000.0 <= temp_k <= 50000.0, f"Temperature out of LR range: {temp_k}"

    # ---- Step 2: predictions_to_dict ----
    # Before commit 8350903, this crashed at i=135 IndexError.
    slider_dict = predictions_to_dict(preds, batch_idx=0)
    assert len(slider_dict) == 135, (
        f"predictions_to_dict must return 135 keys for v1; got {len(slider_dict)}"
    )
    assert "Exposure2012" in slider_dict
    assert "ToneCurveBlue_Pt6_Y" in slider_dict   # last v1 field
    assert "CurveRefineSaturation" not in slider_dict  # v2 extension must NOT appear
    # Values are finite floats
    for field, value in slider_dict.items():
        assert isinstance(value, float), f"{field} value is {type(value).__name__}, not float"
        assert math.isfinite(value), f"{field} value is non-finite: {value}"

    # ---- Step 3: raw forward output for loss-side analysis ----
    # WeightedSliderLoss expects raw (log-space Temperature) predictions, not
    # the postprocessed (exp+clamp) output. Call the model directly to get
    # the raw tensor.
    img_batch, meta_batch = engine._build_batch([img], [metadata], 0, 1)
    engine._model.eval()
    with torch.no_grad():
        raw_pred = engine._model(img_batch, meta_batch)   # [1, 135], log-K
    assert raw_pred.shape == (1, 135)

    # ---- Step 4: WeightedSliderLoss(v1) forward ----
    # Before commit 31e82ca, building the loss with v2-sized buffers and then
    # passing a v1 prediction would broadcast-crash.
    loss_fn = WeightedSliderLoss(slider_set_version="v1")
    targets = _v1_targets_for_loss(1)
    md = _v1_loss_metadata(1)
    total_loss = loss_fn(raw_pred, targets, md)
    assert total_loss.shape == (), "loss must be scalar"
    assert torch.isfinite(total_loss).item(), "loss must be finite"

    # Loss components return dict — exercise that path too
    components = loss_fn(raw_pred, targets, md, return_components=True)
    for key in ("total", "mse", "spread", "temp_bucket", "tint_bucket", "sign_wrong"):
        assert key in components, f"loss components missing key: {key}"
        assert torch.isfinite(components[key]).item(), f"{key} non-finite"

    # ---- Step 5: direction_stats on the same raw output ----
    # Before commit 31e82ca, this crashed at IndexError i=135.
    dir_stats = loss_fn.direction_stats(raw_pred, targets, md)
    assert len(dir_stats) == 135, (
        f"direction_stats must return 135 entries for v1; got {len(dir_stats)}"
    )
    assert "Temperature" in dir_stats
    assert "CurveRefineSaturation" not in dir_stats

    # ---- Step 6: per_field_mae on the same raw output ----
    # Before commit 31e82ca, this crashed at IndexError i=135.
    mae = loss_fn.per_field_mae(raw_pred, targets)
    assert len(mae) == 135, (
        f"per_field_mae must return 135 entries for v1; got {len(mae)}"
    )
    assert "Exposure2012" in mae
    assert "CurveRefineSaturation" not in mae

    # ---- Step 7: write_xmp using the predictions dict ----
    # The downstream pipeline.py:write_xmp call. Must produce a valid XMP
    # without choking on the 135-key dict (write_xmp is dict-keyed, but
    # confirming end-to-end is the point of this test).
    xmp_path = tmp_path / "out.xmp"
    write_xmp(xmp_path, slider_dict)
    assert xmp_path.exists(), "write_xmp did not produce a file"
    assert xmp_path.stat().st_size > 0, "write_xmp produced an empty file"

    # ---- Step 8: read_xmp roundtrip ----
    # Read the just-written XMP back and confirm a sample of slider values
    # match what we wrote (within fp32 quantisation tolerance — XMP stores
    # ints/short floats).
    parsed = read_xmp(xmp_path)
    # Exposure2012 spot-check: should be close to slider_dict["Exposure2012"]
    parsed_exposure = parsed.get("Exposure2012")
    assert parsed_exposure is not None, "Exposure2012 absent from written XMP"
    assert abs(float(parsed_exposure) - slider_dict["Exposure2012"]) < 0.5, (
        f"Exposure2012 roundtrip mismatch: wrote {slider_dict['Exposure2012']}, "
        f"read {parsed_exposure}"
    )
    # Temperature roundtrip: must be in Kelvin, within tolerance of what we wrote.
    parsed_temp = parsed.get("Temperature")
    assert parsed_temp is not None, "Temperature absent from written XMP"
    assert abs(float(parsed_temp) - slider_dict["Temperature"]) < 5.0, (
        f"Temperature roundtrip mismatch: wrote {slider_dict['Temperature']}, "
        f"read {parsed_temp}"
    )


    def test_build_batch_maps_metadata_strings_to_registry_ids(tmp_path: Path) -> None:
        reg = EmbeddingRegistry()
        reg.camera_makes = {"unknown": 0, "Canon": 1}
        reg.camera_models = {"unknown": 0, "EOS R5": 1}
        reg.lenses = {"unknown": 0, "RF24-70mm": 1}
        reg.camera_profiles = {"unknown": 0, "Standard": 1}
        reg.wb_presets = {"unknown": 0, "Daylight": 1}
        model = SonnaEditor(
            registry=reg,
            _embedding_sizes={
                "num_makes": 2, "num_models": 2, "num_lenses": 2,
                "num_profiles": 2, "num_wb_presets": 2,
            },
            _pretrained_backbone=False,
            arch_version=1,
            slider_set_version="v1",
        )
        ckpt = tmp_path / "registry_ids.ckpt"
        model.save_checkpoint(ckpt)

        engine = InferenceEngine(ckpt, device="cpu")
        img = Image.new("RGB", (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION), (128, 128, 128))
        metadata = {
            "camera_make": "Canon",
            "camera_model": "EOS R5",
            "lens_model": "RF24-70mm",
            "camera_profile": "Standard",
            "white_balance_preset": "Daylight",
            "iso": 100.0,
            "shutter_speed": 0.008,
            "aperture": 5.6,
            "focal_length": 50.0,
            "as_shot_wb": (5500.0, 0.0),
        }
        _, meta_batch = engine._build_batch([img], [metadata], 0, 1)
        assert meta_batch["camera_make_id"].item() == 1
        assert meta_batch["camera_model_id"].item() == 1
        assert meta_batch["lens_id"].item() == 1
        assert meta_batch["camera_profile_id"].item() == 1
        assert meta_batch["wb_preset_id"].item() == 1


def test_v2_production_pipeline_end_to_end(tmp_path: Path) -> None:
    """Same gate for v2. Confirms today's slider_set refactor doesn't
    accidentally break v2 paths while fixing v1."""
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
    ckpt = tmp_path / "v2.ckpt"
    model.save_checkpoint(ckpt)

    engine = InferenceEngine(ckpt, device="cpu")
    assert engine._model._slider_set_version == "v2"

    img = Image.new(
        "RGB",
        (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION),
        (128, 128, 128),
    )
    metadata = _dummy_metadata()
    preds = engine.predict([img], [metadata], batch_size=1)
    assert preds.shape == (1, 147)

    slider_dict = predictions_to_dict(preds, batch_idx=0)
    assert len(slider_dict) == 147
    assert "CurveRefineSaturation" in slider_dict
    assert "ShadowTint" in slider_dict

    img_batch, meta_batch = engine._build_batch([img], [metadata], 0, 1)
    engine._model.eval()
    with torch.no_grad():
        raw_pred = engine._model(img_batch, meta_batch)
    assert raw_pred.shape == (1, 147)

    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    targets = torch.full((1, 147), 0.0)
    targets[:, _TEMPERATURE_IDX] = 5500.0
    md = _v1_loss_metadata(1)
    total_loss = loss_fn(raw_pred, targets, md)
    assert torch.isfinite(total_loss).item()

    dir_stats = loss_fn.direction_stats(raw_pred, targets, md)
    assert len(dir_stats) == 147
    assert "CurveRefineSaturation" in dir_stats

    mae = loss_fn.per_field_mae(raw_pred, targets)
    assert len(mae) == 147
    assert "ShadowTint" in mae

    xmp_path = tmp_path / "v2_out.xmp"
    write_xmp(xmp_path, slider_dict)
    assert xmp_path.exists() and xmp_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Mode B end-to-end — sidecar metadata propagation through the production
# pipeline (covers Commits X + Y from 2026-05-14 Mode B Step 3)
# ---------------------------------------------------------------------------
# This test exercises the full Mode B inference path:
#   1. Synthetic v1 base ckpt at non-default resolution (256)
#   2. Mode B ckpt built from it via the Step 2 converter
#   3. Mode B sidecar JSON correctly inherits the base ckpt's resolution
#      (Commit Y: fix(mode-b): Mode B ckpt sidecar inherits base ckpt's
#      resolution)
#   4. process_shoot_with_model loads the Mode B ckpt, runs inference,
#      writes sonna_predictions.json
#   5. sonna_predictions.json carries profile_type, profile_id,
#      base_checkpoint, slider_set_version propagated from the Mode B
#      sidecar (Commit X: feat(inference): record profile_type and
#      slider_set_version in predictions sidecar)


def test_mode_b_end_to_end_sidecar_propagation(tmp_path: Path) -> None:
    from unittest.mock import patch
    import json
    from PIL import Image as _Image

    from sonna_editor.inference import pipeline as pl
    from sonna_editor.mode_b.checkpoint_builder import (
        build_mode_b_checkpoint,
    )
    from sonna_editor.mode_b import survey as survey_mod

    # ---- 1. Synthetic v1 base ckpt at resolution=256 (non-default) ----
    reg = EmbeddingRegistry()
    reg.camera_makes    = {"unknown": 0}
    reg.camera_models   = {"unknown": 0}
    reg.lenses          = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets      = {"unknown": 0}
    base_model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_makes": 4, "num_models": 4, "num_lenses": 4,
                          "num_profiles": 4, "num_wb_presets": 4},
        _pretrained_backbone=False,
        arch_version=1,
        slider_set_version="v1",
    )
    base_ckpt = tmp_path / "base-256px.ckpt"
    base_model.save_checkpoint(base_ckpt)
    # Override arch_config so the base ckpt resolution is explicitly 256
    blob = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    blob["arch_config"]["image_resolution"] = 256
    torch.save(blob, base_ckpt)

    # ---- 2. Build Mode B ckpt from the 256px base ----
    survey_path = tmp_path / "survey.json"
    survey_mod.write_survey(
        survey_mod.build_survey_payload(
            {key: 0 for key in survey_mod.QUESTION_ORDER}
        ),
        survey_path,
    )
    preset_path = Path(__file__).parent / "fixtures" / "preset_sonna_v1.xmp"
    mode_b_ckpt = tmp_path / "mb.ckpt"
    build_mode_b_checkpoint(
        preset_path=preset_path,
        survey_path=survey_path,
        base_ckpt_path=base_ckpt,
        output_ckpt_path=mode_b_ckpt,
        profile_name="Mode B - E2E Test",
    )

    # ---- 3. Mode B sidecar inherits base resolution (Commit Y assertion) ----
    mb_sidecar = json.loads(mode_b_ckpt.with_suffix(".json").read_text())
    assert mb_sidecar["resolution"] == 256, (
        f"Mode B sidecar should inherit base ckpt resolution 256; "
        f"got {mb_sidecar['resolution']}"
    )
    assert mb_sidecar["profile_type"] == "mode_b_initial"
    mode_b_profile_id = mb_sidecar["profile_id"]

    # ---- 4. Run process_shoot_with_model end-to-end on the Mode B ckpt ----
    shoot_dir = tmp_path / "shoot"
    shoot_dir.mkdir()
    (shoot_dir / "img_0.cr3").write_bytes(b"x")  # placeholder; _extract_one patched
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    fake_preview = _Image.new("RGB", (256, 256), (128, 128, 128))
    fake_metadata = {
        "iso": 100.0, "shutter_speed": 0.008, "aperture": 5.6,
        "focal_length": 50.0,
        "camera_make": "unknown", "camera_model": "unknown",
        "lens_model": "unknown", "camera_profile": "unknown",
        "white_balance_preset": "unknown",
        "histogram": [[1.0 / 32] * 32] * 3,
        "as_shot_temperature": 5500.0,
        "as_shot_tint": 0.0,
        "as_shot_wb": (5500.0, 0.0),
    }

    with patch.object(pl, "_extract_one",
                      lambda p, target_size: (fake_preview, fake_metadata)), \
         patch.object(pl, "write_xmp", lambda *a, **k: None):
        pl.process_shoot_with_model(
            input_dir=shoot_dir,
            model_path=mode_b_ckpt,
            output_dir=output_dir,
            save_predictions=True,
            device="cpu",
        )

    # ---- 5. sonna_predictions.json carries Mode B identity (Commit X) ----
    pred_sidecar_path = output_dir / "sonna_predictions.json"
    assert pred_sidecar_path.exists(), "sonna_predictions.json not written"
    pred_sidecar = json.loads(pred_sidecar_path.read_text())
    assert pred_sidecar["profile_type"] == "mode_b_initial", (
        "profile_type must propagate from Mode B ckpt sidecar to predictions sidecar"
    )
    assert pred_sidecar["profile_id"] == mode_b_profile_id
    assert pred_sidecar["base_checkpoint"] == str(base_ckpt)
    assert pred_sidecar["slider_set_version"] == "v1"
