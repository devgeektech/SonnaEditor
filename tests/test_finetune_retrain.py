"""Tests for finetune/retrain.py and the DataModule WeightedRandomSampler extension."""

from __future__ import annotations

import json
import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sonna_editor import config
from sonna_editor.finetune.retrain import (
    _aggregate_mae,
    _atomic_save,
    _bump_version,
    _write_version_sidecar,
)
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor
from sonna_editor.training.datamodule import SonnaDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_native_checkpoint(tmp_path: Path) -> Path:
    """Save a minimal native SonnaEditor checkpoint and return its path."""
    reg = EmbeddingRegistry()
    reg.camera_bodies = {"unknown": 0, "Canon EOS R5": 1}
    reg.lenses = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets = {"unknown": 0}
    # arch_version=0 here so the legacy add_camera_body test below exercises
    # the body_emb growth path. v1.1.0 (default) replaced body_emb with
    # separate make_emb + model_emb (covered in test_training.py).
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_bodies": 2, "num_lenses": 1, "num_profiles": 1, "num_wb_presets": 1},
        _pretrained_backbone=False,
        arch_version=0,
    )
    ckpt_path = tmp_path / "model-v1.0.1.ckpt"
    model.save_checkpoint(ckpt_path)
    return ckpt_path


def _make_histogram_bytes() -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, np.ones((3, 32), dtype=np.float32) / 32)
    return buf.getvalue()


def _make_minimal_parquet(tmp_path: Path, n_rows: int = 4, name: str = "data.parquet") -> Path:
    """Write a minimal Parquet compatible with SonnaDataModule."""
    import io, os
    from PIL import Image

    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir(exist_ok=True)
    thumb_paths = []
    for i in range(n_rows):
        img_path = thumb_dir / f"img_{i}.jpg"
        Image.new("RGB", (384, 384), color=(128, 128, 128)).save(img_path)
        thumb_paths.append(str(img_path))

    data: dict = {
        "thumbnail_path": thumb_paths,
        "iso": [400.0] * n_rows,
        "shutter_speed": [0.002] * n_rows,
        "aperture": [2.8] * n_rows,
        "focal_length": [50.0] * n_rows,
        "camera_body": ["Canon EOS R5"] * n_rows,
        "lens_model": ["RF 24-70mm"] * n_rows,
        "camera_profile": [None] * n_rows,
        "white_balance_preset": ["Auto"] * n_rows,
        "histogram": [_make_histogram_bytes()] * n_rows,
        "sample_weight": [1.0] * n_rows,
    }
    for field in config.SLIDER_FIELDS:
        data[field] = [0.5] * n_rows

    path = tmp_path / name
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Version bumping
# ---------------------------------------------------------------------------

class TestBumpVersion:
    def test_empty_dir_bumps_base_patch(self, tmp_path):
        base = tmp_path / "model-v1.0.1.ckpt"
        base.touch()
        result = _bump_version(tmp_path, base)
        assert result == "model-v1.0.2"

    def test_finds_highest_version(self, tmp_path):
        base = tmp_path / "model-v1.0.1.ckpt"
        higher = tmp_path / "model-v1.0.3.ckpt"
        base.touch()
        higher.touch()
        result = _bump_version(tmp_path, base)
        assert result == "model-v1.0.4"

    def test_output_format_matches_pattern(self, tmp_path):
        import re
        base = tmp_path / "model-v1.0.1.ckpt"
        base.touch()
        result = _bump_version(tmp_path, base)
        assert re.fullmatch(r"model-v\d+\.\d+\.\d+", result)

    def test_ignores_candidate_suffix_in_version_parse(self, tmp_path):
        base = tmp_path / "model-v1.0.1.ckpt"
        candidate = tmp_path / "model-v1.0.2-candidate.ckpt"
        base.touch()
        candidate.touch()
        result = _bump_version(tmp_path, base)
        assert result == "model-v1.0.3"


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------

class TestAtomicSave:
    def test_creates_file_at_path(self, tmp_path):
        ckpt = _make_minimal_native_checkpoint(tmp_path)
        model = SonnaEditor.from_checkpoint(ckpt)
        out = tmp_path / "out.ckpt"
        _atomic_save(model, out)
        assert out.exists()

    def test_no_tmp_file_remains(self, tmp_path):
        ckpt = _make_minimal_native_checkpoint(tmp_path)
        model = SonnaEditor.from_checkpoint(ckpt)
        out = tmp_path / "out.ckpt"
        _atomic_save(model, out)
        assert not out.with_suffix(".ckpt.tmp").exists()

    def test_base_checkpoint_unchanged(self, tmp_path):
        base = _make_minimal_native_checkpoint(tmp_path)
        original_mtime = base.stat().st_mtime
        original_size = base.stat().st_size

        model = SonnaEditor.from_checkpoint(base)
        out = tmp_path / "model-v1.0.2.ckpt"
        _atomic_save(model, out)

        assert base.stat().st_mtime == original_mtime
        assert base.stat().st_size == original_size


# ---------------------------------------------------------------------------
# Version sidecar
# ---------------------------------------------------------------------------

class TestVersionSidecar:
    def test_sidecar_written_alongside(self, tmp_path):
        ckpt_path = tmp_path / "model-v1.0.2.ckpt"
        ckpt_path.touch()
        _write_version_sidecar(ckpt_path, {"version": "model-v1.0.2"})
        assert (tmp_path / "model-v1.0.2.json").exists()

    def test_sidecar_has_required_keys(self, tmp_path):
        ckpt_path = tmp_path / "model-v1.0.2.ckpt"
        ckpt_path.touch()
        metadata = {
            "version": "model-v1.0.2",
            "date_iso": "2026-05-10T08:00:00+00:00",
            "base_version": "model-v1.0.1",
            "base_val_loss": 0.001,
            "ft_val_loss": 0.0009,
            "improvement_pct": 10.0,
            "n_capture_rows": 16,
            "checkpoint_status": "promoted",
        }
        _write_version_sidecar(ckpt_path, metadata)
        loaded = json.loads((tmp_path / "model-v1.0.2.json").read_text())
        for key in ("version", "date_iso", "base_version", "ft_val_loss", "checkpoint_status"):
            assert key in loaded, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# DataModule: WeightedRandomSampler
# ---------------------------------------------------------------------------

class TestDataModuleExtensions:
    def test_weighted_sampler_when_unequal(self, tmp_path):
        """Mixed weights (1.0 and 2.0) → WeightedRandomSampler used."""
        path = _make_minimal_parquet(tmp_path)
        df = pd.read_parquet(path)
        df.loc[2:, "sample_weight"] = 2.0
        path.unlink()
        df.to_parquet(path, index=False)

        dm = SonnaDataModule(
            train_parquet=path,
            val_parquet=path,
            test_parquet=path,
            batch_size=2,
            num_workers=0,
            sample_weight_col="sample_weight",
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert loader.sampler is not None
        from torch.utils.data import WeightedRandomSampler
        assert isinstance(loader.sampler, WeightedRandomSampler)

    def test_shuffle_when_all_equal(self, tmp_path):
        """All-equal weights → WeightedRandomSampler NOT used (shuffle=True path)."""
        path = _make_minimal_parquet(tmp_path)
        dm = SonnaDataModule(
            train_parquet=path,
            val_parquet=path,
            test_parquet=path,
            batch_size=2,
            num_workers=0,
            sample_weight_col="sample_weight",
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        from torch.utils.data import WeightedRandomSampler
        assert not isinstance(loader.sampler, WeightedRandomSampler)

    def test_injected_registry_skips_rebuild(self, tmp_path):
        """Pre-built registry → build_registry() not called during setup()."""
        path = _make_minimal_parquet(tmp_path)
        pre_built = EmbeddingRegistry()
        pre_built.camera_bodies = {"unknown": 0, "Canon EOS R5": 1}

        with mock.patch(
            "sonna_editor.training.datamodule.build_registry"
        ) as mock_build:
            dm = SonnaDataModule(
                train_parquet=path,
                val_parquet=path,
                test_parquet=path,
                batch_size=2,
                num_workers=0,
                registry=pre_built,
            )
            dm.setup("fit")
            mock_build.assert_not_called()

        assert dm.registry is pre_built

    def test_new_camera_body_grows_embedding(self, tmp_path):
        """model.add_camera_body() for a novel body → num_embeddings increases."""
        ckpt = _make_minimal_native_checkpoint(tmp_path)
        model = SonnaEditor.from_checkpoint(ckpt)

        n_before = model.metadata_encoder.body_emb.num_embeddings
        model.add_camera_body("Sony A7R V")
        n_after = model.metadata_encoder.body_emb.num_embeddings

        assert n_after == n_before + 1
        assert "Sony A7R V" in model.registry.camera_bodies


# ---------------------------------------------------------------------------
# finetune_model() — full pipeline (Trainer mocked)
# ---------------------------------------------------------------------------

class TestFinetunModel:
    """Mock pl.Trainer so no actual GPU training runs. Tests logic only."""

    def _make_setup(self, tmp_path: Path):
        """Return (base_ckpt_path, finetune_parquet, val_parquet)."""
        base_ckpt = _make_minimal_native_checkpoint(tmp_path)
        finetune_pq = _make_minimal_parquet(tmp_path, n_rows=4, name="ft.parquet")
        val_pq = _make_minimal_parquet(tmp_path, n_rows=4, name="val.parquet")
        return base_ckpt, finetune_pq, val_pq

    def _mock_trainer_fit(self, trainer_instance, best_ckpt_path: Path, val_loss: float):
        """Configure mock trainer to simulate a completed training run."""
        trainer_instance.current_epoch = 5
        trainer_instance.callback_metrics = {"val_loss": torch.tensor(val_loss)}

        ckpt_cb = mock.MagicMock()
        ckpt_cb.best_model_path = str(best_ckpt_path)
        trainer_instance.checkpoint_callback = ckpt_cb

    @pytest.fixture
    def base_setup(self, tmp_path):
        return self._make_setup(tmp_path)

    def test_return_dict_all_keys(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            # Simulate: ft_val_loss slightly better than base
            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),   # base
                (0.0009, {f: 0.9 for f in config.SLIDER_FIELDS}),  # ft
            ]
            trainer_inst = MockTrainer.return_value

            # Best ckpt must be a real file since _load_from_checkpoint is called on it
            real_ckpt = tmp_path / "best.ckpt"
            model_for_best = SonnaEditor.from_checkpoint(base_ckpt)
            model_for_best.save_checkpoint(real_ckpt)
            self._mock_trainer_fit(trainer_inst, real_ckpt, val_loss=0.0009)

            from sonna_editor.finetune.retrain import finetune_model
            result = finetune_model(
                base_ckpt, ft_pq, val_pq,
                output_dir=tmp_path / "out",
                n_capture_rows=16,
                n_original_rows=100,
            )

        required_keys = {
            "base_version", "ft_version", "checkpoint_path", "checkpoint_status",
            "base_val_loss", "ft_val_loss", "improvement_pct", "improved",
            "epochs_trained", "n_capture_rows", "n_original_rows",
            "base_per_field_mae", "ft_per_field_mae",
        }
        assert required_keys.issubset(result.keys())

    def test_improved_flag_and_status(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup
        real_ckpt = tmp_path / "best.ckpt"
        SonnaEditor.from_checkpoint(base_ckpt).save_checkpoint(real_ckpt)

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),
                (0.0008, {f: 0.8 for f in config.SLIDER_FIELDS}),
            ]
            self._mock_trainer_fit(MockTrainer.return_value, real_ckpt, 0.0008)

            from sonna_editor.finetune.retrain import finetune_model
            result = finetune_model(base_ckpt, ft_pq, val_pq, output_dir=tmp_path / "out")

        assert result["improved"] is True
        assert result["checkpoint_status"] == "promoted"
        assert result["improvement_pct"] > 0

    def test_regression_flag_and_candidate(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup
        real_ckpt = tmp_path / "best.ckpt"
        SonnaEditor.from_checkpoint(base_ckpt).save_checkpoint(real_ckpt)

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),
                (0.0015, {f: 1.5 for f in config.SLIDER_FIELDS}),  # worse
            ]
            self._mock_trainer_fit(MockTrainer.return_value, real_ckpt, 0.0015)

            from sonna_editor.finetune.retrain import finetune_model
            result = finetune_model(base_ckpt, ft_pq, val_pq, output_dir=tmp_path / "out")

        assert result["improved"] is False
        assert result["checkpoint_status"] == "candidate"
        assert result["improvement_pct"] < 0
        assert "candidate" in result["checkpoint_path"]

    def test_checkpoint_written_in_output_dir(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup
        real_ckpt = tmp_path / "best.ckpt"
        SonnaEditor.from_checkpoint(base_ckpt).save_checkpoint(real_ckpt)
        out_dir = tmp_path / "output"

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),
                (0.0009, {f: 0.9 for f in config.SLIDER_FIELDS}),
            ]
            self._mock_trainer_fit(MockTrainer.return_value, real_ckpt, 0.0009)

            from sonna_editor.finetune.retrain import finetune_model
            result = finetune_model(base_ckpt, ft_pq, val_pq, output_dir=out_dir)

        ckpt_path = Path(result["checkpoint_path"])
        assert ckpt_path.parent == out_dir.resolve()
        assert ckpt_path.exists()

    def test_base_checkpoint_file_preserved(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup
        real_ckpt = tmp_path / "best.ckpt"
        SonnaEditor.from_checkpoint(base_ckpt).save_checkpoint(real_ckpt)

        original_mtime = base_ckpt.stat().st_mtime
        original_size = base_ckpt.stat().st_size

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),
                (0.0009, {f: 0.9 for f in config.SLIDER_FIELDS}),
            ]
            self._mock_trainer_fit(MockTrainer.return_value, real_ckpt, 0.0009)

            from sonna_editor.finetune.retrain import finetune_model
            finetune_model(base_ckpt, ft_pq, val_pq, output_dir=tmp_path / "out")

        assert base_ckpt.stat().st_mtime == original_mtime
        assert base_ckpt.stat().st_size == original_size

    def test_version_sidecar_created(self, tmp_path, base_setup):
        base_ckpt, ft_pq, val_pq = base_setup
        real_ckpt = tmp_path / "best.ckpt"
        SonnaEditor.from_checkpoint(base_ckpt).save_checkpoint(real_ckpt)
        out_dir = tmp_path / "out"

        with mock.patch("sonna_editor.finetune.retrain.pl.Trainer") as MockTrainer, \
             mock.patch("sonna_editor.finetune.retrain._evaluate") as mock_eval:

            mock_eval.side_effect = [
                (0.001, {f: 1.0 for f in config.SLIDER_FIELDS}),
                (0.0009, {f: 0.9 for f in config.SLIDER_FIELDS}),
            ]
            self._mock_trainer_fit(MockTrainer.return_value, real_ckpt, 0.0009)

            from sonna_editor.finetune.retrain import finetune_model
            result = finetune_model(base_ckpt, ft_pq, val_pq, output_dir=out_dir)

        ckpt_path = Path(result["checkpoint_path"])
        sidecar = ckpt_path.with_suffix(".json")
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "ft_val_loss" in meta
        assert "checkpoint_status" in meta


# ---------------------------------------------------------------------------
# v1/v2 shape-mismatch regression coverage for _aggregate_mae
# ---------------------------------------------------------------------------
# Before today's slider_set helper migration, _aggregate_mae iterated
# config.SLIDER_FIELDS (147) and looked up d[field] from per_field_mae
# dicts. For v1 inputs (135 keys) it produced 12 always-NaN entries for
# v2-extension fields, polluting the validation report.

import math
from sonna_editor.slider_set import v1_fields, fields_for_version


def test_aggregate_mae_v1_inputs_returns_135_keys() -> None:
    """v1 per_field_mae dicts → aggregated result has exactly 135 keys."""
    v1 = v1_fields()
    val_outputs = [
        {f: float(i) * 0.01 for i, f in enumerate(v1)},
        {f: float(i) * 0.02 for i, f in enumerate(v1)},
    ]
    result = _aggregate_mae(val_outputs)
    assert len(result) == 135
    assert set(result.keys()) == set(v1)
    # v2-extension fields must NOT appear
    assert "CurveRefineSaturation" not in result
    assert "ShadowTint" not in result
    # Sanity: a known field's value is nanmean across the two inputs
    expected_exposure = (0.0 + 0.0) / 2  # index 0 in both
    assert result["Exposure2012"] == pytest.approx(expected_exposure)


def test_aggregate_mae_v2_inputs_returns_147_keys() -> None:
    """v2 per_field_mae dicts preserve v2-extension entries in aggregation."""
    v2 = fields_for_version("v2")
    val_outputs = [
        {f: float(i) * 0.01 for i, f in enumerate(v2)},
    ]
    result = _aggregate_mae(val_outputs)
    assert len(result) == 147
    assert "CurveRefineSaturation" in result


def test_aggregate_mae_empty_inputs_returns_empty_dict() -> None:
    """No batches → empty result, not 147 keys of NaN."""
    assert _aggregate_mae([]) == {}
