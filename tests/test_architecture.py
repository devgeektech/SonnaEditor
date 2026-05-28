"""
Tests for SonnaEditor model architecture.

Covers: forward pass shape, resolution flexibility, single-sample batch,
parameter count, save/load roundtrip, embedding registry growth (mean init),
backbone freeze/unfreeze, Temperature log-space encoding, postprocessing.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from sonna_editor import config
from sonna_editor.model.architecture import SonnaEditor, _grow_embedding
from sonna_editor.model.postprocess import (
    _TEMPERATURE_IDX,
    postprocess_predictions,
    predictions_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_metadata(batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "iso": torch.full((batch_size,), 400.0, device=device),
        "shutter_speed": torch.full((batch_size,), 1 / 200.0, device=device),
        "aperture": torch.full((batch_size,), 2.8, device=device),
        "focal_length": torch.full((batch_size,), 85.0, device=device),
        # v1.0.x reads camera_body_id; v1.1.0 reads camera_make_id + camera_model_id.
        # Provide all three so the helper covers both archs.
        "camera_body_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "camera_make_id":  torch.zeros(batch_size, dtype=torch.long, device=device),
        "camera_model_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "lens_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "camera_profile_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "wb_preset_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "histogram": torch.rand(batch_size, 96, device=device),
        # v1.1.0 AsShot inputs. arch_version=0 tests ignore these by design;
        # arch_version=1 (default) reads them.
        "as_shot_temperature": torch.full((batch_size,), 5500.0, device=device),
        "as_shot_tint":        torch.zeros(batch_size, device=device),
    }


def _dummy_image(batch_size: int, resolution: int = 384, device: str = "cpu") -> torch.Tensor:
    return torch.rand(batch_size, 3, resolution, resolution, device=device)


@pytest.fixture(scope="module")
def model() -> SonnaEditor:
    """CPU model shared across tests that don't need device-specific behaviour."""
    m = SonnaEditor()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

def test_output_shape_default(model: SonnaEditor) -> None:
    B = 4
    N = len(config.SLIDER_FIELDS)
    out = model(_dummy_image(B), _dummy_metadata(B))
    assert out.shape == (B, N), f"Expected ({B}, {N}), got {out.shape}"


def test_output_shape_batch_1(model: SonnaEditor) -> None:
    """Single-sample batch must not crash or produce wrong shape."""
    out = model(_dummy_image(1), _dummy_metadata(1))
    assert out.shape == (1, len(config.SLIDER_FIELDS))


@pytest.mark.parametrize("resolution", [384, 512, 768])
def test_resolution_flexibility(resolution: int) -> None:
    """Architecture must accept all three supported input resolutions without code changes."""
    m = SonnaEditor()
    m.eval()
    N = len(config.SLIDER_FIELDS)
    with torch.no_grad():
        out = m(_dummy_image(2, resolution=resolution), _dummy_metadata(2))
    assert out.shape == (2, N), f"Failed at resolution {resolution}"


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def test_parameter_count(model: SonnaEditor) -> None:
    total = sum(p.numel() for p in model.parameters())
    total_M = total / 1e6
    # ConvNeXt-Tiny ~28.6M + metadata encoder ~0.05M + 13 heads ~1.1M ≈ 29-30M
    assert 25 <= total_M <= 35, f"Parameter count {total_M:.1f}M outside expected 25-35M range"


# ---------------------------------------------------------------------------
# Output ordering matches config.SLIDER_FIELDS
# ---------------------------------------------------------------------------

def test_output_length_matches_slider_fields(model: SonnaEditor) -> None:
    out = model(_dummy_image(1), _dummy_metadata(1))
    assert out.shape[-1] == len(config.SLIDER_FIELDS)


def test_temperature_at_correct_index() -> None:
    """Temperature is at index 11 (after Tone×8 + Presence×3)."""
    assert config.SLIDER_FIELDS[11] == "Temperature"
    assert _TEMPERATURE_IDX == 11


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze
# ---------------------------------------------------------------------------

def test_freeze_backbone_stages() -> None:
    m = SonnaEditor(freeze_backbone=True)
    frozen_params = [
        p for p in m.backbone_features[0].parameters()
    ] + [
        p for p in m.backbone_features[1].parameters()
    ]
    assert all(not p.requires_grad for p in frozen_params), \
        "Stages 0-1 should be frozen after freeze_backbone=True"

    # Later stages should still be trainable
    later_params = list(m.backbone_features[2].parameters())
    assert all(p.requires_grad for p in later_params), \
        "Stages 2+ should remain trainable"


def test_unfreeze_backbone_restores_grads() -> None:
    m = SonnaEditor(freeze_backbone=True)
    m.unfreeze_backbone()
    for p in m.backbone_features.parameters():
        assert p.requires_grad, "All backbone params should require grad after unfreeze"


# ---------------------------------------------------------------------------
# Embedding registry growth (mean init)
# ---------------------------------------------------------------------------

def test_grow_embedding_mean_init() -> None:
    emb = torch.nn.Embedding(4, 8)
    torch.nn.init.normal_(emb.weight)
    expected_mean = emb.weight.data.mean(dim=0)

    new_emb = _grow_embedding(emb)

    assert new_emb.num_embeddings == 5, "Should have grown by 1"
    assert torch.allclose(new_emb.weight.data[-1], expected_mean, atol=1e-6), \
        "New row should be mean of old rows"
    # Old rows unchanged
    assert torch.allclose(new_emb.weight.data[:4], emb.weight.data, atol=1e-6)


def test_add_camera_body_grows_embedding() -> None:
    """
    The embedding table starts at _MIN_BODIES (8) capacity. Adding bodies within
    capacity reuses existing rows (no growth). Only the body that exceeds capacity
    triggers a table extension.
    """
    # arch_version=0 explicitly: body_emb only exists on the legacy v1.0.x arch.
    # v1.1.0 replaced it with separate make_emb + model_emb (covered in the
    # v1.1.0 tests in test_training.py).
    m = SonnaEditor(arch_version=0)
    min_cap = m._MIN_BODIES  # 8
    assert m.metadata_encoder.body_emb.num_embeddings == min_cap

    # Fill all pre-allocated slots — no growth should occur
    for i in range(min_cap):
        bid = m.add_camera_body(f"Body {i}")
        assert bid == i
    assert m.metadata_encoder.body_emb.num_embeddings == min_cap

    # Next add exceeds capacity and must trigger growth
    overflow_id = m.add_camera_body("Body overflow")
    assert overflow_id == min_cap
    assert m.metadata_encoder.body_emb.num_embeddings == min_cap + 1
    assert "Body overflow" in m.registry.camera_bodies

    # Duplicate returns same id, no further growth
    same_id = m.add_camera_body("Body overflow")
    assert same_id == overflow_id
    assert m.metadata_encoder.body_emb.num_embeddings == min_cap + 1


def test_add_camera_body_new_row_is_mean() -> None:
    """The row added when capacity is exceeded must be the mean of existing rows."""
    # arch_version=0 explicitly: body_emb only exists on the legacy v1.0.x arch.
    # v1.1.0 replaced it with separate make_emb + model_emb (covered in the
    # v1.1.0 tests in test_training.py).
    m = SonnaEditor(arch_version=0)
    min_cap = m._MIN_BODIES  # 8

    # Fill all pre-allocated slots first
    for i in range(min_cap):
        m.add_camera_body(f"Body {i}")

    old_weight = m.metadata_encoder.body_emb.weight.data.clone()
    expected_new_row = old_weight.mean(dim=0)

    m.add_camera_body("NewBody triggers growth")
    new_row = m.metadata_encoder.body_emb.weight.data[-1]
    assert torch.allclose(new_row, expected_new_row, atol=1e-6)


def test_add_lens() -> None:
    """Lens table grows only when we exceed _MIN_LENSES (16) capacity."""
    m = SonnaEditor()
    cap = m._MIN_LENSES  # 16
    for i in range(cap):
        m.add_lens(f"Lens {i}")
    assert m.metadata_encoder.lens_emb.num_embeddings == cap
    m.add_lens("Lens overflow")
    assert m.metadata_encoder.lens_emb.num_embeddings == cap + 1


def test_add_camera_profile() -> None:
    """Profile table grows only when we exceed _MIN_PROFILES (8) capacity."""
    m = SonnaEditor()
    cap = m._MIN_PROFILES  # 8
    for i in range(cap):
        m.add_camera_profile(f"Profile {i}")
    assert m.metadata_encoder.profile_emb.num_embeddings == cap
    m.add_camera_profile("Profile overflow")
    assert m.metadata_encoder.profile_emb.num_embeddings == cap + 1


def test_add_wb_preset() -> None:
    """WB table grows only when we exceed _MIN_WB (8) capacity."""
    m = SonnaEditor()
    cap = m._MIN_WB  # 8
    for i in range(cap):
        m.add_wb_preset(f"WB {i}")
    assert m.metadata_encoder.wb_emb.num_embeddings == cap
    m.add_wb_preset("WB overflow")
    assert m.metadata_encoder.wb_emb.num_embeddings == cap + 1


# ---------------------------------------------------------------------------
# New camera body can be used in a forward pass immediately
# ---------------------------------------------------------------------------

def test_new_body_usable_in_forward() -> None:
    m = SonnaEditor()
    m.eval()
    new_id = m.add_camera_body("Nikon Z8")
    meta = _dummy_metadata(2)
    meta["camera_body_id"] = torch.tensor([new_id, new_id])
    with torch.no_grad():
        out = m(_dummy_image(2), meta)
    assert out.shape == (2, len(config.SLIDER_FIELDS))


# ---------------------------------------------------------------------------
# Save / load checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip() -> None:
    m = SonnaEditor()
    m.eval()
    m.add_camera_body("Fujifilm X-T5")

    image = _dummy_image(2)
    meta = _dummy_metadata(2)
    meta["camera_body_id"] = torch.tensor([0, 0])

    with torch.no_grad():
        original_out = m(image, meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.ckpt"
        m.save_checkpoint(ckpt_path)

        loaded = SonnaEditor.from_checkpoint(ckpt_path, device="cpu")
        loaded.eval()

        with torch.no_grad():
            loaded_out = loaded(image, meta)

    assert torch.allclose(original_out, loaded_out, atol=1e-5), \
        "Loaded model output differs from original"
    assert "Fujifilm X-T5" in loaded.registry.camera_bodies


# ---------------------------------------------------------------------------
# Temperature log-scale encoding
# ---------------------------------------------------------------------------

def test_temperature_prediction_in_log_space() -> None:
    """
    Model predicts log(Kelvin) for Temperature slot.
    After postprocessing (exp), value should fall in a plausible Kelvin range.
    """
    m = SonnaEditor()
    m.eval()
    with torch.no_grad():
        raw_out = m(_dummy_image(4), _dummy_metadata(4))

    # Raw temperature slot is in log space — check it's a plausible log(Kelvin)
    raw_temp = raw_out[:, _TEMPERATURE_IDX]
    kelvin = torch.exp(raw_temp)
    # After exp, should be positive; clamping to [2000, 50000] happens in postprocess
    assert (kelvin > 0).all(), "exp(log_temp) must be positive"


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def test_postprocess_temperature_conversion() -> None:
    N = len(config.SLIDER_FIELDS)
    pred = torch.zeros(2, N)
    log_temp = math.log(5500.0)
    pred[:, _TEMPERATURE_IDX] = log_temp

    out = postprocess_predictions(pred)

    assert torch.allclose(out[:, _TEMPERATURE_IDX], torch.tensor(5500.0), atol=1.0)


def test_postprocess_clamps_to_valid_ranges() -> None:
    N = len(config.SLIDER_FIELDS)
    pred = torch.full((1, N), 9999.0)
    pred[:, _TEMPERATURE_IDX] = math.log(9999.0)  # log so exp gives 9999

    out = postprocess_predictions(pred)

    for i, field in enumerate(config.SLIDER_FIELDS):
        lo, hi = config.SLIDER_RANGES[field]
        assert float(out[0, i]) <= hi + 1e-4, f"{field} exceeds max {hi}"
        assert float(out[0, i]) >= lo - 1e-4, f"{field} below min {lo}"


def test_postprocess_does_not_modify_non_temperature() -> None:
    """Tint and other sliders at 0 should remain 0 after postprocessing."""
    N = len(config.SLIDER_FIELDS)
    pred = torch.zeros(1, N)
    pred[:, _TEMPERATURE_IDX] = math.log(6000.0)

    out = postprocess_predictions(pred)

    tint_idx = config.SLIDER_FIELDS.index("Tint")
    assert float(out[0, tint_idx]) == pytest.approx(0.0, abs=1e-5)


def test_predictions_to_dict() -> None:
    N = len(config.SLIDER_FIELDS)
    pred = torch.zeros(1, N)
    pred[0, _TEMPERATURE_IDX] = math.log(5000.0)
    out = postprocess_predictions(pred)
    d = predictions_to_dict(out)

    assert set(d.keys()) == set(config.SLIDER_FIELDS)
    assert d["Temperature"] == pytest.approx(5000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# MPS forward pass (skipped if MPS unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_forward_on_mps() -> None:
    m = SonnaEditor()
    m.eval()
    m.to("mps")

    image = _dummy_image(4, device="mps")
    meta = _dummy_metadata(4, device="mps")

    with torch.no_grad():
        out = m(image, meta)

    N = len(config.SLIDER_FIELDS)
    assert out.shape == (4, N)
    assert out.device.type == "mps"
    assert not torch.isnan(out).any(), "NaN values in MPS forward pass"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
@pytest.mark.parametrize("resolution", [384, 512, 768])
def test_mps_resolution_flexibility(resolution: int) -> None:
    m = SonnaEditor()
    m.eval()
    m.to("mps")
    N = len(config.SLIDER_FIELDS)
    with torch.no_grad():
        out = m(_dummy_image(2, resolution=resolution, device="mps"), _dummy_metadata(2, device="mps"))
    assert out.shape == (2, N)


# ---------------------------------------------------------------------------
# v2 extension heads + slider_set_version (commit 78511ce)
# ---------------------------------------------------------------------------

_V2_EXT_HEADS = (
    "noise_ext_head", "defringe_head", "lens_profile_head",
    "calibration_ext_head", "curve_ext_head",
)


class TestSliderSetVersion:
    """v2 extension heads + slider_set_version flag."""

    def test_default_instantiation_is_v2(self) -> None:
        m = SonnaEditor()
        assert m._slider_set_version == "v2"
        assert m._use_wb_metadata_skip is True
        assert hasattr(m, "wb_metadata_skip")
        for h in _V2_EXT_HEADS:
            assert hasattr(m, h), f"v2 model missing {h}"

    def test_v1_instantiation_has_no_extension_heads(self) -> None:
        m = SonnaEditor(slider_set_version="v1")
        assert m._slider_set_version == "v1"
        for h in _V2_EXT_HEADS:
            assert not hasattr(m, h), f"v1 model unexpectedly has {h}"

    def test_v2_instantiation_has_5_extension_heads(self) -> None:
        m = SonnaEditor(slider_set_version="v2")
        for h in _V2_EXT_HEADS:
            assert hasattr(m, h)

    def test_invalid_slider_set_version_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown slider_set_version"):
            SonnaEditor(slider_set_version="v3")

    def test_v1_forward_outputs_135(self) -> None:
        m = SonnaEditor(slider_set_version="v1")
        m.eval()
        with torch.no_grad():
            out = m(_dummy_image(2), _dummy_metadata(2))
        assert out.shape == (2, 135)

    def test_v2_forward_outputs_147(self) -> None:
        m = SonnaEditor(slider_set_version="v2")
        m.eval()
        with torch.no_grad():
            out = m(_dummy_image(2), _dummy_metadata(2))
        assert out.shape == (2, 147)

    def test_wb_metadata_skip_identity_initialisation(self) -> None:
        """New models start WB predictions near AsShot, then learn residuals."""
        m = SonnaEditor(slider_set_version="v2")
        m.eval()
        for p in m.wb_head.parameters():
            p.data.zero_()
        meta = _dummy_metadata(2)
        meta["as_shot_temperature"] = torch.tensor([3200.0, 6400.0])
        meta["as_shot_tint"] = torch.tensor([-8.0, 12.0])
        with torch.no_grad():
            out = m(_dummy_image(2), meta)
        assert out[0, 11].item() == pytest.approx(math.log(3200.0), rel=1e-5)
        assert out[1, 11].item() == pytest.approx(math.log(6400.0), rel=1e-5)
        assert out[0, 12].item() == pytest.approx(-8.0, abs=1e-5)
        assert out[1, 12].item() == pytest.approx(12.0, abs=1e-5)

    def test_wb_metadata_skip_can_be_disabled_for_legacy_compat(self) -> None:
        m = SonnaEditor(use_wb_metadata_skip=False)
        assert m._use_wb_metadata_skip is False
        assert not hasattr(m, "wb_metadata_skip")


_PROD_CKPT = Path("v1_learning/model-v1.2.3-prod256.ckpt")
_HAS_PROD_CKPT = _PROD_CKPT.exists()


class TestCheckpointCrossVersion:
    """save/load round-trip + cross-version load gating (commit 78511ce, decision B 2026-05-13)."""

    @pytest.mark.skipif(not _HAS_PROD_CKPT, reason="v1.2.3 ckpt not present")
    def test_v123_ckpt_loads_native_as_v1(self) -> None:
        m = SonnaEditor.from_checkpoint(_PROD_CKPT, device="cpu")
        assert m._slider_set_version == "v1"
        m.eval()
        with torch.no_grad():
            out = m(_dummy_image(2, resolution=256), _dummy_metadata(2))
        assert out.shape == (2, 135)

    @pytest.mark.skipif(not _HAS_PROD_CKPT, reason="v1.2.3 ckpt not present")
    def test_v123_ckpt_warm_starts_to_v2(self) -> None:
        m = SonnaEditor.from_checkpoint(
            _PROD_CKPT, device="cpu", target_slider_set_version="v2",
        )
        assert m._slider_set_version == "v2"
        assert hasattr(m, "defringe_head")
        m.eval()
        with torch.no_grad():
            out = m(_dummy_image(2, resolution=256), _dummy_metadata(2))
        assert out.shape == (2, 147)

    def test_v2_ckpt_to_v1_load_raises(self, tmp_path: Path) -> None:
        """v2→v1 load rejected to avoid silent information loss from
        dropping the 5 extension heads (per decision B 2026-05-13)."""
        v2_ckpt = tmp_path / "v2_test.ckpt"
        SonnaEditor().save_checkpoint(v2_ckpt)
        with pytest.raises(ValueError, match="v2 checkpoint as v1"):
            SonnaEditor.from_checkpoint(v2_ckpt, target_slider_set_version="v1")

    def test_v2_ckpt_save_load_roundtrip(self, tmp_path: Path) -> None:
        v2_ckpt = tmp_path / "v2_roundtrip.ckpt"
        SonnaEditor().save_checkpoint(v2_ckpt)
        loaded = SonnaEditor.from_checkpoint(v2_ckpt)
        assert loaded._slider_set_version == "v2"
        assert loaded._use_wb_metadata_skip is True
        assert hasattr(loaded, "defringe_head")

    def test_legacy_native_ckpt_loads_without_wb_skip(self, tmp_path: Path) -> None:
        legacy_ckpt = tmp_path / "legacy_no_wb_skip.ckpt"
        model = SonnaEditor(use_wb_metadata_skip=False)
        model.save_checkpoint(legacy_ckpt)
        ckpt = torch.load(legacy_ckpt, map_location="cpu", weights_only=False)
        ckpt["arch_config"].pop("use_wb_metadata_skip", None)
        torch.save(ckpt, legacy_ckpt)

        loaded = SonnaEditor.from_checkpoint(legacy_ckpt)
        assert loaded._use_wb_metadata_skip is False
        with torch.no_grad():
            out = loaded(_dummy_image(1), _dummy_metadata(1))
        assert out.shape == (1, 147)
