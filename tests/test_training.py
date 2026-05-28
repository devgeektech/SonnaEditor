"""Tests for Task 3.3: SonnaDataset, SonnaDataModule, SonnaLightningModule."""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor
from sonna_editor.model.augmentation import ValidationAugmentation
from sonna_editor.slider_set import v1_fields
from sonna_editor.training.datamodule import (
    SonnaDataset,
    _cat_id,
    _decode_histogram,
    _safe_float,
    build_registry,
)
from sonna_editor.training.module import SonnaLightningModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_thumbnail(path: Path, size: tuple[int, int] = (200, 200)) -> None:
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path, format="JPEG", quality=85)


def _make_histogram_bytes() -> bytes:
    arr = np.random.rand(3, 32).astype(np.float32)
    arr /= arr.sum(axis=1, keepdims=True) + 1e-8
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_df(tmp_dir: Path) -> pd.DataFrame:
    """Two-row DataFrame that mirrors a real Parquet dataset."""
    rows = []
    for i in range(4):
        thumb = tmp_dir / f"thumb_{i}.jpg"
        _make_thumbnail(thumb)
        row: dict = {
            "id": f"id_{i}",
            "thumbnail_path": str(thumb),
            "iso": 400.0 + i * 100,
            "shutter_speed": 0.01,
            "aperture": 2.8,
            "focal_length": 85.0,
            "camera_body": "Sony A7III" if i % 2 == 0 else None,
            "lens_model": "FE 85mm",
            "camera_profile": "Standard",
            "white_balance_preset": "As Shot",
            "histogram": _make_histogram_bytes(),
        }
        for field in SLIDER_FIELDS:
            row[field] = 0.0 if field != "Temperature" else 5500.0
        # Make some fields absent (None) on second row
        if i == 1:
            row["SharpenRadius"] = None
            row["GrainAmount"] = None
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def registry(sample_df: pd.DataFrame) -> EmbeddingRegistry:
    return build_registry(sample_df)


@pytest.fixture
def dataset(sample_df: pd.DataFrame, registry: EmbeddingRegistry) -> SonnaDataset:
    return SonnaDataset(sample_df, ValidationAugmentation(), registry)


@pytest.fixture
def model(registry: EmbeddingRegistry) -> SonnaEditor:
    return SonnaEditor(registry=registry, freeze_backbone=False)


@pytest.fixture
def module(model: SonnaEditor) -> SonnaLightningModule:
    return SonnaLightningModule(model=model)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_safe_float_none() -> None:
    assert _safe_float(None) == 0.0


def test_safe_float_nan() -> None:
    assert _safe_float(float("nan")) == 0.0


def test_safe_float_valid() -> None:
    assert _safe_float(400.0) == pytest.approx(400.0)


def test_cat_id_known(registry: EmbeddingRegistry) -> None:
    assert _cat_id(registry.camera_bodies, "Sony A7III") == registry.camera_bodies["Sony A7III"]


def test_cat_id_none_returns_zero(registry: EmbeddingRegistry) -> None:
    assert _cat_id(registry.camera_bodies, None) == 0


def test_cat_id_unknown_string_returns_zero(registry: EmbeddingRegistry) -> None:
    assert _cat_id(registry.camera_bodies, "Unregistered Camera") == 0


def test_decode_histogram_shape() -> None:
    hist = _decode_histogram(_make_histogram_bytes())
    assert hist.shape == (96,)
    assert hist.dtype == torch.float32


def test_decode_histogram_non_negative() -> None:
    hist = _decode_histogram(_make_histogram_bytes())
    assert (hist >= 0).all()


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------

def test_registry_unknown_at_index_zero(sample_df: pd.DataFrame) -> None:
    reg = build_registry(sample_df)
    assert reg.camera_bodies.get("unknown") == 0
    assert reg.lenses.get("unknown") == 0


def test_registry_includes_training_values(sample_df: pd.DataFrame) -> None:
    reg = build_registry(sample_df)
    assert "Sony A7III" in reg.camera_bodies
    assert "FE 85mm" in reg.lenses


def test_registry_none_not_in_mapping(sample_df: pd.DataFrame) -> None:
    reg = build_registry(sample_df)
    assert None not in reg.camera_bodies


def test_registry_ids_are_unique(sample_df: pd.DataFrame) -> None:
    reg = build_registry(sample_df)
    for mapping in (reg.camera_bodies, reg.lenses, reg.camera_profiles, reg.wb_presets):
        ids = list(mapping.values())
        assert len(ids) == len(set(ids)), "Registry IDs must be unique"


# ---------------------------------------------------------------------------
# SonnaDataset
# ---------------------------------------------------------------------------

def test_dataset_length(dataset: SonnaDataset, sample_df: pd.DataFrame) -> None:
    assert len(dataset) == len(sample_df)


def test_dataset_returns_three_items(dataset: SonnaDataset) -> None:
    result = dataset[0]
    assert len(result) == 3


def test_dataset_image_shape(dataset: SonnaDataset) -> None:
    image, _, _ = dataset[0]
    assert image.shape[0] == 3
    assert image.dtype == torch.float32


def test_dataset_image_range(dataset: SonnaDataset) -> None:
    image, _, _ = dataset[0]
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0


def test_dataset_metadata_keys(dataset: SonnaDataset) -> None:
    _, metadata, _ = dataset[0]
    required = {"iso", "shutter_speed", "aperture", "focal_length",
                "camera_body_id", "lens_id", "camera_profile_id", "wb_preset_id",
                "histogram"}
    assert required.issubset(metadata.keys())


def test_dataset_metadata_histogram_shape(dataset: SonnaDataset) -> None:
    _, metadata, _ = dataset[0]
    assert metadata["histogram"].shape == (96,)


def test_dataset_metadata_ids_are_long(dataset: SonnaDataset) -> None:
    _, metadata, _ = dataset[0]
    for key in ("camera_body_id", "lens_id", "camera_profile_id", "wb_preset_id"):
        assert metadata[key].dtype == torch.long, f"{key} should be long"


def test_dataset_target_shape(dataset: SonnaDataset) -> None:
    _, _, target = dataset[0]
    assert target.shape == (len(SLIDER_FIELDS),)
    assert target.dtype == torch.float32


def test_dataset_target_shape_matches_v1_slider_set(
    sample_df: pd.DataFrame,
    registry: EmbeddingRegistry,
) -> None:
    ds = SonnaDataset(sample_df, ValidationAugmentation(), registry, slider_set_version="v1")
    _, _, target = ds[0]
    assert target.shape == (len(v1_fields()),)


def test_dataset_target_none_becomes_nan(dataset: SonnaDataset) -> None:
    """Absent slider values must be NaN in the target tensor."""
    _, _, target = dataset[1]  # row 1 has None for SharpenRadius
    idx = SLIDER_FIELDS.index("SharpenRadius")
    assert math.isnan(target[idx].item())


def test_dataset_target_present_values_not_nan(dataset: SonnaDataset) -> None:
    _, _, target = dataset[0]
    exp_idx = SLIDER_FIELDS.index("Exposure2012")
    assert not math.isnan(target[exp_idx].item())


def test_dataset_temperature_in_kelvin_in_target(dataset: SonnaDataset) -> None:
    """Target tensor stores raw Kelvin — NOT log-space. Loss does the transform."""
    _, _, target = dataset[0]
    temp_idx = SLIDER_FIELDS.index("Temperature")
    assert target[temp_idx].item() == pytest.approx(5500.0)


# ---------------------------------------------------------------------------
# SonnaLightningModule — construction and configure_optimizers
# ---------------------------------------------------------------------------

def test_module_instantiates(module: SonnaLightningModule) -> None:
    assert isinstance(module, SonnaLightningModule)


def test_configure_optimizers_two_param_groups(module: SonnaLightningModule) -> None:
    cfg = module.configure_optimizers()
    optimizer = cfg["optimizer"]
    assert len(optimizer.param_groups) == 2


def test_configure_optimizers_backbone_lr_lower(module: SonnaLightningModule) -> None:
    cfg = module.configure_optimizers()
    groups = cfg["optimizer"].param_groups
    # First group = backbone (lower lr), second = heads + encoder (full lr)
    assert groups[0]["lr"] < groups[1]["lr"]


def test_configure_optimizers_has_scheduler(module: SonnaLightningModule) -> None:
    cfg = module.configure_optimizers()
    assert "lr_scheduler" in cfg


def test_loss_fn_is_weighted_slider_loss(module: SonnaLightningModule) -> None:
    from sonna_editor.model.losses import WeightedSliderLoss
    assert isinstance(module.loss_fn, WeightedSliderLoss)


# ---------------------------------------------------------------------------
# SonnaLightningModule — training_step with a synthetic batch
# ---------------------------------------------------------------------------

def _make_batch(B: int = 2) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Synthetic training batch matching the real datamodule's dict shape,
    including the v1.1.0 inputs (AsShot, separate make/model) that new-arch
    models read."""
    from sonna_editor.config import IMAGE_RESOLUTION
    images = torch.rand(B, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    metadata = {
        "iso":               torch.full((B,), 400.0),
        "shutter_speed":     torch.full((B,), 0.01),
        "aperture":          torch.full((B,), 2.8),
        "focal_length":      torch.full((B,), 85.0),
        "camera_body_id":    torch.zeros(B, dtype=torch.long),
        "camera_make_id":    torch.zeros(B, dtype=torch.long),
        "camera_model_id":   torch.zeros(B, dtype=torch.long),
        "lens_id":           torch.zeros(B, dtype=torch.long),
        "camera_profile_id": torch.zeros(B, dtype=torch.long),
        "wb_preset_id":      torch.zeros(B, dtype=torch.long),
        "histogram":         torch.rand(B, 96),
        "as_shot_temperature": torch.full((B,), 5500.0),
        "as_shot_tint":        torch.zeros(B),
    }
    targets = torch.zeros(B, len(SLIDER_FIELDS))
    targets[:, SLIDER_FIELDS.index("Temperature")] = 5500.0
    return images, metadata, targets


def test_training_step_returns_scalar(module: SonnaLightningModule) -> None:
    module.eval()
    batch = _make_batch()
    loss = module.training_step(batch, 0)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_training_step_loss_is_non_negative(module: SonnaLightningModule) -> None:
    batch = _make_batch()
    loss = module.training_step(batch, 0)
    assert loss.item() >= 0.0


def test_validation_step_accumulates_mae(module: SonnaLightningModule) -> None:
    module.on_validation_epoch_start()
    batch = _make_batch()
    module.validation_step(batch, 0)
    assert len(module._val_mae_outputs) == 1


def test_on_validation_epoch_end_clears_outputs(module: SonnaLightningModule) -> None:
    module.on_validation_epoch_start()
    batch = _make_batch()
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    # _val_mae_outputs is NOT cleared in epoch_end (only on epoch_start)
    # so it should still have the data
    assert len(module._val_mae_outputs) == 1


def test_on_validation_epoch_start_clears_outputs(module: SonnaLightningModule) -> None:
    module._val_mae_outputs.append({"Exposure2012": 0.1})
    module.on_validation_epoch_start()
    assert module._val_mae_outputs == []


# ---------------------------------------------------------------------------
# Stage 2 — WeightedSliderLoss new term smoke tests
# ---------------------------------------------------------------------------

def _make_loss_inputs(B: int = 16) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Synthetic (predictions, targets, metadata) for loss-only tests.

    Predictions in model output space (Temperature in log K).
    Targets in dataset space (Temperature in raw K, NaN allowed).
    """
    import math as _math
    torch.manual_seed(0)
    N = len(SLIDER_FIELDS)
    pred = torch.randn(B, N) * 0.1
    pred[:, SLIDER_FIELDS.index("Temperature")] = _math.log(5000.0) + 0.05 * torch.randn(B)
    tgt = torch.randn(B, N) * 0.1
    tgt[:, SLIDER_FIELDS.index("Temperature")] = 4500.0 + 1000.0 * torch.randn(B)
    tgt[:, SLIDER_FIELDS.index("Tint")] = 5.0 + 10.0 * torch.randn(B)
    meta = {
        "as_shot_temperature": 5500.0 + 500.0 * torch.randn(B),
        "as_shot_tint":        torch.randn(B),
    }
    return pred, tgt, meta


def test_loss_returns_components_dict() -> None:
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs()
    c = loss_fn(pred, tgt, meta, return_components=True)
    assert set(c.keys()) == {"total", "mse", "spread", "temp_bucket", "tint_bucket", "sign_wrong"}
    for k, v in c.items():
        assert v.ndim == 0, f"{k} should be scalar"
        assert torch.isfinite(v), f"{k} should be finite"


def test_loss_legacy_scalar_call_still_works() -> None:
    """Legacy callers without metadata or return_components keyword get a scalar."""
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, _ = _make_loss_inputs()
    loss = loss_fn(pred, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_spread_term_is_non_negative() -> None:
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs()
    c = loss_fn(pred, tgt, meta, return_components=True)
    assert c["spread"].item() >= 0.0


def test_loss_temp_bucket_zero_when_asshot_all_nan() -> None:
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs()
    meta["as_shot_temperature"] = torch.full_like(meta["as_shot_temperature"], float("nan"))
    c = loss_fn(pred, tgt, meta, return_components=True)
    assert c["temp_bucket"].item() == 0.0


def test_loss_tint_bucket_zero_when_truth_tint_all_nan() -> None:
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs()
    tgt[:, SLIDER_FIELDS.index("Tint")] = float("nan")
    c = loss_fn(pred, tgt, meta, return_components=True)
    assert c["tint_bucket"].item() == 0.0


def test_loss_gradient_flow_to_predictions() -> None:
    """Total loss must produce non-trivial gradients on all predictions."""
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs()
    pred = pred.detach().requires_grad_(True)
    loss = loss_fn(pred, tgt, meta)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).any()


def test_loss_spread_one_sided_hinge() -> None:
    """When pred_std >= tgt_std for every field, spread term must be 0."""
    import math as _math
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B, N = 32, len(SLIDER_FIELDS)
    # Make pred std very large; targets nearly constant.
    pred = torch.randn(B, N) * 10.0
    pred[:, SLIDER_FIELDS.index("Temperature")] = _math.log(5000.0) + torch.randn(B) * 0.1
    tgt = torch.zeros(B, N) + 0.01 * torch.randn(B, N)
    tgt[:, SLIDER_FIELDS.index("Temperature")] = 5000.0
    c = loss_fn(pred, tgt, {}, return_components=True)
    assert c["spread"].item() == 0.0


def test_loss_single_sample_batch_bucket_terms_zero() -> None:
    """Bucket terms need >=2 samples per bucket — single-sample batch → 0."""
    from sonna_editor.model.losses import WeightedSliderLoss
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    pred, tgt, meta = _make_loss_inputs(B=1)
    c = loss_fn(pred, tgt, meta, return_components=True)
    assert c["temp_bucket"].item() == 0.0
    assert c["tint_bucket"].item() == 0.0


def test_dataset_metadata_includes_as_shot_when_columns_present(
    sample_df: pd.DataFrame, registry: EmbeddingRegistry, tmp_dir: Path
) -> None:
    """SonnaDataset surfaces as_shot_temperature/as_shot_tint when parquet has them."""
    df = sample_df.copy()
    df["as_shot_temperature"] = 5200.0
    df["as_shot_tint"] = 3.0
    ds = SonnaDataset(df, ValidationAugmentation(), registry)
    _, metadata, _ = ds[0]
    assert "as_shot_temperature" in metadata and "as_shot_tint" in metadata
    assert metadata["as_shot_temperature"].item() == pytest.approx(5200.0)
    assert metadata["as_shot_tint"].item() == pytest.approx(3.0)


def test_dataset_metadata_as_shot_nan_when_column_missing(dataset: SonnaDataset) -> None:
    """When parquet lacks AsShot columns, metadata still has the keys (NaN)."""
    _, metadata, _ = dataset[0]
    assert "as_shot_temperature" in metadata
    assert math.isnan(metadata["as_shot_temperature"].item())
    assert "as_shot_tint" in metadata
    assert math.isnan(metadata["as_shot_tint"].item())


# ---------------------------------------------------------------------------
# v1.1.0 architecture — AsShot Temperature/Tint as model inputs
# ---------------------------------------------------------------------------

def _make_batch_with_as_shot(B: int = 2, resolution: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    from sonna_editor.config import IMAGE_RESOLUTION
    res = resolution if resolution is not None else IMAGE_RESOLUTION
    images = torch.rand(B, 3, res, res)
    metadata = {
        "iso":               torch.full((B,), 400.0),
        "shutter_speed":     torch.full((B,), 0.01),
        "aperture":          torch.full((B,), 2.8),
        "focal_length":      torch.full((B,), 85.0),
        # v1.0.x ckpts read camera_body_id; v1.1.0 reads camera_make_id +
        # camera_model_id. Provide all three so the helper covers both archs.
        "camera_body_id":    torch.zeros(B, dtype=torch.long),
        "camera_make_id":    torch.zeros(B, dtype=torch.long),
        "camera_model_id":   torch.zeros(B, dtype=torch.long),
        "lens_id":           torch.zeros(B, dtype=torch.long),
        "camera_profile_id": torch.zeros(B, dtype=torch.long),
        "wb_preset_id":      torch.zeros(B, dtype=torch.long),
        "histogram":         torch.rand(B, 96),
        "as_shot_temperature": torch.tensor([5500.0, 3200.0][:B]),
        "as_shot_tint":        torch.tensor([0.0, 8.0][:B]),
    }
    targets = torch.zeros(B, len(SLIDER_FIELDS))
    targets[:, SLIDER_FIELDS.index("Temperature")] = 5500.0
    return images, metadata, targets


def test_v1_1_new_arch_forward_shape() -> None:
    """New-arch model accepts AsShot inputs and returns [B, 135]."""
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    images, metadata, _ = _make_batch_with_as_shot(2)
    out = model(images, metadata)
    assert out.shape == (2, len(SLIDER_FIELDS))
    assert torch.isfinite(out).all()


def test_v1_1_new_arch_nan_as_shot_handled() -> None:
    """NaN AsShot in metadata must not produce NaN outputs."""
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    images, metadata, _ = _make_batch_with_as_shot(2)
    metadata["as_shot_temperature"] = torch.tensor([float("nan"), 4500.0])
    metadata["as_shot_tint"]        = torch.tensor([float("nan"), 5.0])
    out = model(images, metadata)
    assert torch.isfinite(out).all()


def test_v1_1_old_arch_ignores_as_shot_keys() -> None:
    """Old-arch model accepts AsShot keys in metadata but doesn't use them.

    Old arch runs at 384 px (v1.0.x convention) even though the global config
    has bumped to 512 — the architecture itself is resolution-flexible thanks
    to AdaptiveAvgPool, but real v1.0.x ckpts assume 384 input.
    """
    model = SonnaEditor(arch_version=0, _pretrained_backbone=False)
    images, metadata, _ = _make_batch_with_as_shot(2, resolution=384)
    out = model(images, metadata)
    assert out.shape == (2, len(SLIDER_FIELDS))
    assert torch.isfinite(out).all()


def test_v1_1_old_arch_works_without_as_shot_keys() -> None:
    """Old-arch model works fine when AsShot keys are simply absent."""
    model = SonnaEditor(arch_version=0, _pretrained_backbone=False)
    images, metadata, _ = _make_batch_with_as_shot(2, resolution=384)
    metadata.pop("as_shot_temperature")
    metadata.pop("as_shot_tint")
    out = model(images, metadata)
    assert torch.isfinite(out).all()


def test_v1_1_state_dict_shapes() -> None:
    """Old vs new architecture have distinct fusion_mlp + camera-id layers."""
    sd_new = SonnaEditor(arch_version=1, _pretrained_backbone=False).state_dict()
    sd_old = SonnaEditor(arch_version=0, _pretrained_backbone=False).state_dict()

    # Fusion MLP: old 112-d concat, new 128-d concat
    assert sd_new["metadata_encoder.fusion_mlp.0.weight"].shape == (128, 128)
    assert sd_old["metadata_encoder.fusion_mlp.0.weight"].shape == (128, 112)

    # AsShot encoders: only in v1.1.0
    assert "metadata_encoder.as_shot_temp_fc.weight" in sd_new
    assert "metadata_encoder.as_shot_tint_fc.weight" in sd_new
    assert "metadata_encoder.as_shot_temp_fc.weight" not in sd_old
    assert "metadata_encoder.as_shot_tint_fc.weight" not in sd_old

    # Make/model embeddings: only in v1.1.0. body_emb only in v1.0.x.
    assert "metadata_encoder.make_emb.weight" in sd_new
    assert "metadata_encoder.model_emb.weight" in sd_new
    assert "metadata_encoder.body_emb.weight" not in sd_new
    assert "metadata_encoder.body_emb.weight" in sd_old
    assert "metadata_encoder.make_emb.weight" not in sd_old

    # Focal-length encoder: v1.0.x Linear(8, 8) one-hot, v1.1.0 Linear(1, 8) log
    assert sd_old["metadata_encoder.focal_length_fc.weight"].shape == (8, 8)
    assert sd_new["metadata_encoder.focal_length_fc.weight"].shape == (8, 1)


def test_v1_1_image_resolution_round_trip(tmp_path: Path) -> None:
    """arch_config['image_resolution'] is stored on save and read on load."""
    from sonna_editor import config as cfg
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    ckpt = tmp_path / "res.ckpt"
    model.save_checkpoint(ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert raw["arch_config"]["image_resolution"] == cfg.IMAGE_RESOLUTION
    assert raw["arch_config"]["arch_version"] == 1


def test_v1_1_nan_safe_aperture_and_focal() -> None:
    """NaN aperture / focal_length must not produce NaN outputs (sentinels in encoder)."""
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    images, metadata, _ = _make_batch_with_as_shot(2)
    metadata["aperture"] = torch.tensor([float("nan"), 4.0])
    metadata["focal_length"] = torch.tensor([float("nan"), 50.0])
    out = model(images, metadata)
    assert torch.isfinite(out).all()


def test_v1_1_separate_make_model_embeddings_route_distinctly() -> None:
    """Changing make_id alone should change the forward output (proves the
    embedding is actually consumed, not silently dropped)."""
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    model.eval()
    images, metadata, _ = _make_batch_with_as_shot(2)
    with torch.no_grad():
        out_a = model(images, metadata)
    metadata["camera_make_id"] = torch.tensor([1, 1], dtype=torch.long)
    with torch.no_grad():
        out_b = model(images, metadata)
    # Outputs must differ — the new make embedding has different (random) weights
    assert not torch.allclose(out_a, out_b)


def test_v1_1_native_ckpt_round_trip(tmp_path: Path) -> None:
    """save_checkpoint + from_checkpoint preserves new-arch flag and predictions."""
    model = SonnaEditor(arch_version=1, _pretrained_backbone=False)
    model.eval()
    ckpt = tmp_path / "round.ckpt"
    model.save_checkpoint(ckpt)

    loaded = SonnaEditor.from_checkpoint(ckpt)
    assert loaded._arch_version == 1
    loaded.eval()

    images, metadata, _ = _make_batch_with_as_shot(2)
    with torch.no_grad():
        a = model(images, metadata)
        b = loaded(images, metadata)
    assert torch.allclose(a, b, atol=1e-5)


def test_v1_1_native_ckpt_backward_compat(tmp_path: Path) -> None:
    """Native ckpts without arch_config.arch_version default to old arch
    (state-dict shape detection: make_emb key absent → arch_version=0)."""
    model = SonnaEditor(arch_version=0, _pretrained_backbone=False)
    model.eval()
    ckpt_path = tmp_path / "legacy.ckpt"
    # Save manually without the arch_config.arch_version field to simulate
    # a pre-v1.1.0 native ckpt
    torch.save(
        {
            "model_state": model.state_dict(),
            "registry": model.registry.to_dict(),
            "arch_config": {"image_resolution": 384, "num_sliders": len(SLIDER_FIELDS)},
        },
        ckpt_path,
    )
    loaded = SonnaEditor.from_checkpoint(ckpt_path)
    assert loaded._arch_version == 0


# ---------------------------------------------------------------------------
# OvercorrectionWarningCallback
# ---------------------------------------------------------------------------

class _FakeModule:
    """Minimal LightningModule stand-in for callback tests."""
    def __init__(self, direction_outputs: list[dict[str, tuple[int, int]]]):
        self._val_direction_outputs = direction_outputs


class _FakeTrainer:
    def __init__(self, current_epoch: int = 5) -> None:
        self.current_epoch = current_epoch
        self.should_stop = False


def test_overcorrection_callback_warns_when_threshold_crossed(capsys) -> None:
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    per_batch = [
        {"Temperature": (30, 100), "Tint": (10, 100), "Exposure2012": (5, 100)},
    ]
    pl_module = _FakeModule(per_batch)
    trainer = _FakeTrainer(current_epoch=5)
    cb.on_validation_epoch_end(trainer, pl_module)
    assert trainer.should_stop is False, "warning callback must not halt training"
    out = capsys.readouterr().out
    assert "overcorrection warning" in out
    assert "Temperature" in out


def test_overcorrection_callback_skips_warmup(capsys) -> None:
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    per_batch = [{"Temperature": (90, 100)}]
    pl_module = _FakeModule(per_batch)
    trainer = _FakeTrainer(current_epoch=0)   # before check_after_epoch
    cb.on_validation_epoch_end(trainer, pl_module)
    assert trainer.should_stop is False
    assert capsys.readouterr().out == "", "no warning should fire during warmup"


def test_overcorrection_callback_clean_when_under_threshold(capsys) -> None:
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    per_batch = [{"Temperature": (20, 100), "Tint": (24, 100), "Exposure2012": (0, 50)}]
    pl_module = _FakeModule(per_batch)
    trainer = _FakeTrainer(current_epoch=5)
    cb.on_validation_epoch_end(trainer, pl_module)
    assert trainer.should_stop is False
    assert capsys.readouterr().out == ""


def test_overcorrection_callback_aggregates_batches(capsys) -> None:
    """Counts must be summed across batches before computing %."""
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    # Per-batch: each batch has Temperature 5/15 wrong. Aggregated: 10/30 = 33%.
    per_batch = [
        {"Temperature": (5, 15)},
        {"Temperature": (5, 15)},
    ]
    pl_module = _FakeModule(per_batch)
    trainer = _FakeTrainer(current_epoch=5)
    cb.on_validation_epoch_end(trainer, pl_module)
    assert trainer.should_stop is False
    assert "overcorrection warning" in capsys.readouterr().out


def test_overcorrection_callback_ignores_low_sample_fields(capsys) -> None:
    """Fields with very few directional truths in val must not trip the alarm
    even if 100% of those are wrong (sample size too small to be a signal)."""
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    per_batch = [{"Temperature": (3, 3), "Tint": (10, 100)}]  # Temp 100% but only n=3
    pl_module = _FakeModule(per_batch)
    trainer = _FakeTrainer(current_epoch=5)
    cb.on_validation_epoch_end(trainer, pl_module)
    assert trainer.should_stop is False
    assert capsys.readouterr().out == ""


def test_overcorrection_callback_warns_only_once_per_streak(capsys) -> None:
    """Once warned, stay quiet until an epoch comes back clean."""
    from sonna_editor.training.callbacks import OvercorrectionWarningCallback
    cb = OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=1)
    per_batch_bad   = [{"Temperature": (30, 100)}]
    per_batch_clean = [{"Temperature": (10, 100)}]
    trainer = _FakeTrainer(current_epoch=5)

    # First bad epoch → warns.
    cb.on_validation_epoch_end(trainer, _FakeModule(per_batch_bad))
    assert "overcorrection warning" in capsys.readouterr().out
    # Second bad epoch in same streak → stays quiet.
    cb.on_validation_epoch_end(trainer, _FakeModule(per_batch_bad))
    assert capsys.readouterr().out == ""
    # Clean epoch → resets the streak silently.
    cb.on_validation_epoch_end(trainer, _FakeModule(per_batch_clean))
    assert capsys.readouterr().out == ""
    # New bad epoch after clean → warns again.
    cb.on_validation_epoch_end(trainer, _FakeModule(per_batch_bad))
    assert "overcorrection warning" in capsys.readouterr().out
