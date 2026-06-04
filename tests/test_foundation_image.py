from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from sonna_editor import config
from sonna_editor.foundation import (
    is_image_foundation_checkpoint,
    load_sonna_model_from_foundation_checkpoint,
)
from sonna_editor.mode_b import survey as survey_mod
from sonna_editor.mode_b.checkpoint_builder import build_mode_b_checkpoint
from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.training.image_foundation import (
    FOUNDATION_IMAGE_TYPE,
    FoundationEnhancementModel,
    PairedImageDataset,
    find_image_pairs,
    split_image_pairs,
)
from sonna_editor.training.profile_runner import _warm_start_model_from_checkpoint
from sonna_editor.training.profile_runner import _initialise_image_foundation_output_priors


FIXTURE_PRESET = Path(__file__).parent / "fixtures" / "preset_sonna_v1.xmp"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path)


def _write_image_foundation_checkpoint(tmp_path: Path) -> Path:
    model = FoundationEnhancementModel(pretrained_backbone=False)
    path = tmp_path / "foundation-image.ckpt"
    model.save_checkpoint(
        path,
        image_resolution=64,
        train_rows=2,
        val_rows=1,
        test_rows=1,
        metrics={"best_val_loss": 0.1},
    )
    return path


def test_find_image_pairs_matches_by_stem(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_dng"
    tiff_dir = tmp_path / "expert_tiff"
    _write_image(raw_dir / "a.jpg", (10, 20, 30))
    _write_image(raw_dir / "b.jpg", (40, 50, 60))
    _write_image(tiff_dir / "a.tiff", (70, 80, 90))

    pairs = find_image_pairs(raw_dir, tiff_dir)

    assert len(pairs) == 1
    assert pairs[0].source_path.name == "a.jpg"
    assert pairs[0].target_path.name == "a.tiff"


def test_paired_image_dataset_returns_aligned_tensors(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    tiff_dir = tmp_path / "tiff"
    _write_image(raw_dir / "sample.jpg", (10, 20, 30))
    _write_image(tiff_dir / "sample.tif", (70, 80, 90))
    pair = find_image_pairs(raw_dir, tiff_dir)[0]

    item = PairedImageDataset([pair], resolution=32)[0]

    assert item["source"].shape == torch.Size([3, 32, 32])
    assert item["target"].shape == torch.Size([3, 32, 32])
    assert item["source"].dtype == torch.float32
    assert item["target"].min() >= 0
    assert item["target"].max() <= 1


def test_split_image_pairs_keeps_train_non_empty(tmp_path: Path) -> None:
    pairs = []
    for i in range(5):
        source = tmp_path / "raw" / f"{i}.jpg"
        target = tmp_path / "tiff" / f"{i}.tif"
        _write_image(source, (i, i, i))
        _write_image(target, (i + 1, i + 1, i + 1))
        pairs.append(find_image_pairs(source.parent, target.parent)[-1])

    train, val, test = split_image_pairs(pairs, val_ratio=0.2, test_ratio=0.2)

    assert len(train) >= 1
    assert len(val) == 1
    assert len(test) == 1


def test_image_foundation_checkpoint_loads_as_sonna_backbone(tmp_path: Path) -> None:
    ckpt = _write_image_foundation_checkpoint(tmp_path)

    assert is_image_foundation_checkpoint(ckpt)
    model = load_sonna_model_from_foundation_checkpoint(
        ckpt,
        slider_set_version=config.CURRENT_SLIDER_SET_VERSION,
    )

    assert isinstance(model, SonnaEditor)
    assert model._slider_set_version == config.CURRENT_SLIDER_SET_VERSION


def test_profile_training_warm_start_accepts_image_foundation(tmp_path: Path) -> None:
    ckpt = _write_image_foundation_checkpoint(tmp_path)

    model = _warm_start_model_from_checkpoint(
        model_cls=SonnaEditor,
        checkpoint_path=ckpt,
        registry=None,
        slider_set_version=config.CURRENT_SLIDER_SET_VERSION,
    )

    assert isinstance(model, SonnaEditor)
    assert model._slider_set_version == config.CURRENT_SLIDER_SET_VERSION


def test_image_foundation_warm_start_initialises_output_priors(tmp_path: Path) -> None:
    ckpt = _write_image_foundation_checkpoint(tmp_path)
    train_parquet = tmp_path / "train.parquet"
    row = {field: 0.0 for field in config.SLIDER_FIELDS}
    row["Exposure2012"] = 0.75
    row["Temperature"] = 4800.0
    row["Tint"] = 4.0
    pd.DataFrame([row]).to_parquet(train_parquet)
    model = SonnaEditor(_pretrained_backbone=False, slider_set_version="v2")
    with torch.no_grad():
        model.tone_head[-1].bias.zero_()

    priors = _initialise_image_foundation_output_priors(
        model=model,
        base_model_checkpoint=ckpt,
        train_parquet=train_parquet,
        slider_set_version="v2",
        disabled=False,
    )

    assert priors is not None
    assert priors["Exposure2012"] == 0.75
    assert model.tone_head[-1].bias[0].item() == 0.75
    assert torch.allclose(model.wb_head[-1].bias, torch.zeros_like(model.wb_head[-1].bias))


def test_mode_b_can_build_from_image_foundation_checkpoint(tmp_path: Path) -> None:
    ckpt = _write_image_foundation_checkpoint(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey_mod.write_survey(
        survey_mod.build_survey_payload({key: 0 for key in survey_mod.QUESTION_ORDER}),
        survey_path,
    )
    output = tmp_path / "mode-b.ckpt"

    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey_path,
        base_ckpt_path=ckpt,
        output_ckpt_path=output,
        profile_name="Lite From TIFF Foundation",
    )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert output.exists()
    assert sidecar["base_foundation_type"] == FOUNDATION_IMAGE_TYPE
    assert sidecar["slider_set_version"] == config.CURRENT_SLIDER_SET_VERSION
    assert sidecar["resolution"] == 64
