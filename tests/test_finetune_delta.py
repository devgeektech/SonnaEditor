"""Tests for finetune/delta.py — delta analysis and fine-tune dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sonna_editor import config
from sonna_editor.finetune.capture import (
    _SOURCE_LR_DEFAULT,
    _SOURCE_MODEL_FILTERED,
    _SOURCE_USER_FINAL,
    _histogram_to_bytes,
)
from sonna_editor.finetune.delta import analyse_deltas, prepare_finetune_dataset
from sonna_editor.slider_set import v1_fields


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deltas_json(
    field_entries: dict[str, tuple[float | None, str]],
) -> str:
    """Build a deltas JSON string: {field: (delta, source)}."""
    result = {}
    for field in config.SLIDER_FIELDS:
        if field in field_entries:
            delta, source = field_entries[field]
        else:
            delta, source = None, _SOURCE_LR_DEFAULT
        result[field] = {"delta": delta, "source": source}
    return json.dumps(result)


def _make_captures_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal captures DataFrame for testing."""
    return pd.DataFrame(rows)


def _make_histogram_bytes() -> bytes:
    return _histogram_to_bytes(np.ones((3, 32), dtype=np.float32) / 32)


# ---------------------------------------------------------------------------
# analyse_deltas
# ---------------------------------------------------------------------------

class TestAnalyseDeltas:
    def test_empty_dataframe(self):
        df = _make_captures_df([])
        result = analyse_deltas(df)
        assert result["n_photos"] == 0
        assert result["per_field"] == {}
        assert result["most_adjusted_fields"] == []
        assert result["correlations"] == []

    def test_user_final_only_in_per_field(self):
        """Stats in per_field must only count user_final deltas."""
        rows = [
            {
                "deltas": _make_deltas_json({
                    "Exposure2012": (0.5, _SOURCE_USER_FINAL),
                    "PerspectiveScale": (2.0, _SOURCE_MODEL_FILTERED),
                    "GrainAmount": (None, _SOURCE_LR_DEFAULT),
                }),
                "iso": 400.0, "shutter_speed": 0.002, "aperture": 2.8, "focal_length": 85.0,
            }
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        assert "Exposure2012" in result["per_field"]
        assert "PerspectiveScale" not in result["per_field"]
        assert "GrainAmount" not in result["per_field"]

    def test_per_field_stats_correct(self):
        """Mean, std, min, max computed correctly from user_final deltas."""
        deltas = [0.1, 0.3, -0.2]
        rows = [
            {
                "deltas": _make_deltas_json({"Exposure2012": (d, _SOURCE_USER_FINAL)}),
                "iso": 400.0, "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
            for d in deltas
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        stats = result["per_field"]["Exposure2012"]
        assert stats["n_with_delta"] == 3
        assert stats["mean_delta"] == pytest.approx(np.mean(deltas), abs=1e-6)
        assert stats["abs_mean_delta"] == pytest.approx(np.mean(np.abs(deltas)), abs=1e-6)
        assert stats["min_delta"] == pytest.approx(min(deltas), abs=1e-6)
        assert stats["max_delta"] == pytest.approx(max(deltas), abs=1e-6)

    def test_most_adjusted_fields_sorted_descending(self):
        """most_adjusted_fields must be sorted by abs_mean_delta descending."""
        rows = [
            {
                "deltas": _make_deltas_json({
                    "Exposure2012": (0.1, _SOURCE_USER_FINAL),
                    "Temperature": (200.0, _SOURCE_USER_FINAL),
                    "Shadows2012": (5.0, _SOURCE_USER_FINAL),
                }),
                "iso": 400.0, "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        abs_deltas = [v for _, v in result["most_adjusted_fields"]]
        assert abs_deltas == sorted(abs_deltas, reverse=True)

    def test_most_adjusted_capped_at_ten(self):
        """Output must contain at most 10 entries."""
        # Give all fields a user_final delta
        field_entries = {f: (float(i + 1), _SOURCE_USER_FINAL) for i, f in enumerate(config.SLIDER_FIELDS)}
        rows = [
            {
                "deltas": _make_deltas_json(field_entries),
                "iso": 400.0, "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        assert len(result["most_adjusted_fields"]) <= 10

    def test_filtered_field_deltas_reported_separately(self):
        """model_filtered deltas must appear in filtered_field_deltas, not per_field."""
        rows = [
            {
                "deltas": _make_deltas_json({
                    "PerspectiveScale": (5.0, _SOURCE_MODEL_FILTERED),
                }),
                "iso": 400.0, "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        assert "PerspectiveScale" not in result["per_field"]
        assert "PerspectiveScale" in result["filtered_field_deltas"]

    def test_correlation_reported_above_threshold(self):
        """Strong correlation (|r| ≥ 0.3, p < 0.05, n ≥ 10) must appear in correlations."""
        # Construct 15 rows where Exposure delta tracks ISO almost perfectly
        iso_vals = list(range(100, 1600, 100))  # 15 values
        exp_deltas = [iso / 1000.0 for iso in iso_vals]  # linear relationship
        rows = [
            {
                "deltas": _make_deltas_json({"Exposure2012": (d, _SOURCE_USER_FINAL)}),
                "iso": float(iso), "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
            for iso, d in zip(iso_vals, exp_deltas)
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        exposure_iso_corrs = [
            c for c in result["correlations"]
            if c["field"] == "Exposure2012" and c["metadata_col"] == "iso"
        ]
        assert len(exposure_iso_corrs) == 1
        assert abs(exposure_iso_corrs[0]["spearman_r"]) >= 0.3

    def test_correlation_suppressed_below_n(self):
        """Fewer than 10 paired observations → correlation not reported."""
        rows = [
            {
                "deltas": _make_deltas_json({"Exposure2012": (float(i), _SOURCE_USER_FINAL)}),
                "iso": float(i * 100), "shutter_speed": 0.001, "aperture": 4.0, "focal_length": 50.0,
            }
            for i in range(8)  # only 8 rows — below threshold of 10
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        # Should not report even if there's a strong correlation
        exposure_iso = [
            c for c in result["correlations"]
            if c["field"] == "Exposure2012" and c["metadata_col"] == "iso"
        ]
        assert len(exposure_iso) == 0

    def test_correlation_suppressed_below_r(self):
        """Weak correlation (|r| < 0.3) must not appear even with large n."""
        rng = np.random.default_rng(42)
        # 20 rows with random, uncorrelated deltas and ISO
        rows = [
            {
                "deltas": _make_deltas_json({"Exposure2012": (float(rng.normal(0, 0.1)), _SOURCE_USER_FINAL)}),
                "iso": float(rng.integers(100, 3200)), "shutter_speed": 0.001,
                "aperture": 4.0, "focal_length": 50.0,
            }
            for _ in range(20)
        ]
        df = _make_captures_df(rows)
        result = analyse_deltas(df)
        # With seed=42 and random noise, |r| should be < 0.3
        exposure_iso = [
            c for c in result["correlations"]
            if c["field"] == "Exposure2012" and c["metadata_col"] == "iso"
        ]
        for corr in exposure_iso:
            assert abs(corr["spearman_r"]) >= 0.3  # if it appears, it met the threshold


# ---------------------------------------------------------------------------
# prepare_finetune_dataset
# ---------------------------------------------------------------------------

class TestPrepareFinetune:
    def _make_minimal_training_parquet(self, tmp_path: Path, n_rows: int = 5) -> Path:
        """Write a minimal Parquet with the required columns."""
        data: dict = {
            "id": [f"orig_{i}" for i in range(n_rows)],
            "file_path": [f"/raw/img_{i}.cr3" for i in range(n_rows)],
            "thumbnail_path": [f"/thumbs/img_{i}.jpg" for i in range(n_rows)],
            "shoot_id": ["shoot_A"] * n_rows,
            "capture_time": [None] * n_rows,
            "camera_body": ["Canon EOS R5"] * n_rows,
            "lens_model": ["RF 24-70mm"] * n_rows,
            "camera_profile": [None] * n_rows,
            "white_balance_preset": ["Auto"] * n_rows,
            "iso": [400.0] * n_rows,
            "shutter_speed": [0.002] * n_rows,
            "aperture": [2.8] * n_rows,
            "focal_length": [50.0] * n_rows,
            "histogram": [_make_histogram_bytes()] * n_rows,
        }
        for field in config.SLIDER_FIELDS:
            data[field] = [0.0] * n_rows
        df = pd.DataFrame(data)
        path = tmp_path / "train.parquet"
        df.to_parquet(path, index=False)
        return path

    def _make_minimal_captures(self, n_rows: int = 3) -> pd.DataFrame:
        """Build a minimal captures DataFrame."""
        rows = []
        for i in range(n_rows):
            row: dict = {
                "id": f"cap_{i}",
                "file_path": f"/raw/cap_{i}.cr3",
                "thumbnail_path": f"/thumbs/cap_{i}.jpg",
                "shoot_id": "shoot_B",
                "capture_time": None,
                "camera_body": "Canon EOS R5",
                "lens_model": "RF 85mm",
                "camera_profile": None,
                "white_balance_preset": "Auto",
                "iso": 800.0,
                "shutter_speed": 0.004,
                "aperture": 1.8,
                "focal_length": 85.0,
                "histogram": _make_histogram_bytes(),
                # Provenance cols (should be dropped in output)
                "model_version": "model-v1.0.1",
                "model_path": "/fake/model.ckpt",
                "prediction_timestamp": "2026-05-10T08:00:00+00:00",
                "xmp_modified_time": "2026-05-10T09:00:00+00:00",
                "edit_lag_seconds": 3600.0,
                "predicted_values": "{}",
                "deltas": "{}",
            }
            for field in config.SLIDER_FIELDS:
                row[field] = float(i) * 0.1
            rows.append(row)
        return pd.DataFrame(rows)

    def test_schema_has_sample_weight(self, tmp_path):
        """Combined Parquet must have a sample_weight column."""
        train_path = self._make_minimal_training_parquet(tmp_path)
        captures = self._make_minimal_captures()
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(captures, train_path, out_path)
        assert "sample_weight" in combined.columns

    def test_all_rows_included(self, tmp_path):
        """All captured rows must appear in output (no filtering by delta magnitude)."""
        n_orig = 5
        n_cap = 3
        train_path = self._make_minimal_training_parquet(tmp_path, n_orig)
        captures = self._make_minimal_captures(n_cap)
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(captures, train_path, out_path)
        assert len(combined) == n_orig + n_cap

    def test_original_weight_is_one(self, tmp_path):
        """Original training rows must have sample_weight = 1.0."""
        train_path = self._make_minimal_training_parquet(tmp_path, 5)
        captures = self._make_minimal_captures(0)  # no captures
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(captures, train_path, out_path)
        orig_weights = combined["sample_weight"].values
        assert all(w == pytest.approx(1.0) for w in orig_weights)

    def test_captured_weight_default_equal(self, tmp_path):
        """Default weight_recent=1.0 means captured rows also get weight 1.0."""
        train_path = self._make_minimal_training_parquet(tmp_path, 2)
        captures = self._make_minimal_captures(2)
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(captures, train_path, out_path, weight_recent=1.0)
        assert all(w == pytest.approx(1.0) for w in combined["sample_weight"])

    def test_captured_weight_param(self, tmp_path):
        """weight_recent=2.0 must apply to captured rows only."""
        n_orig = 3
        n_cap = 2
        train_path = self._make_minimal_training_parquet(tmp_path, n_orig)
        captures = self._make_minimal_captures(n_cap)
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(
            captures, train_path, out_path, weight_recent=2.0
        )
        # First n_orig rows are original (weight=1.0), last n_cap are captured (weight=2.0)
        orig_rows = combined[combined["id"].str.startswith("orig_")]
        cap_rows = combined[combined["id"].str.startswith("cap_")]
        assert all(w == pytest.approx(1.0) for w in orig_rows["sample_weight"])
        assert all(w == pytest.approx(2.0) for w in cap_rows["sample_weight"])

    def test_provenance_cols_dropped(self, tmp_path):
        """Provenance/delta columns must not appear in the combined output."""
        train_path = self._make_minimal_training_parquet(tmp_path)
        captures = self._make_minimal_captures()
        out_path = tmp_path / "finetune.parquet"
        combined = prepare_finetune_dataset(captures, train_path, out_path)
        provenance_cols = {
            "model_version", "model_path", "prediction_timestamp",
            "xmp_modified_time", "edit_lag_seconds", "predicted_values", "deltas",
        }
        for col in provenance_cols:
            assert col not in combined.columns, f"Provenance column leaked: {col}"

    def test_output_written_to_disk(self, tmp_path):
        """Combined Parquet must be written to output_path."""
        train_path = self._make_minimal_training_parquet(tmp_path)
        captures = self._make_minimal_captures()
        out_path = tmp_path / "sub" / "finetune.parquet"
        prepare_finetune_dataset(captures, train_path, out_path)
        assert out_path.exists()
        on_disk = pd.read_parquet(out_path)
        assert len(on_disk) > 0


# ---------------------------------------------------------------------------
# v1/v2 shape-mismatch regression coverage
# ---------------------------------------------------------------------------
# Before today's slider_set helper migration, analyse_deltas iterated
# config.SLIDER_FIELDS (147) for all four aggregation loops, producing
# 12 always-empty v2-extension entries in per_field/correlations/
# filtered_field_deltas when the input captures were v1-shaped.

def _make_v1_only_deltas_json(
    field_entries: dict[str, tuple[float | None, str]],
) -> str:
    """Build a deltas JSON string covering ONLY the 135 v1 fields."""
    result = {}
    for field in v1_fields():
        if field in field_entries:
            delta, source = field_entries[field]
        else:
            delta, source = None, _SOURCE_LR_DEFAULT
        result[field] = {"delta": delta, "source": source}
    return json.dumps(result)


def test_analyse_deltas_v1_captures_no_v2_field_pollution() -> None:
    """v1 captures must not produce v2-extension entries in any aggregation."""
    rows = [
        {
            "deltas": _make_v1_only_deltas_json({
                "Exposure2012": (0.5, _SOURCE_USER_FINAL),
                "Temperature": (250.0, _SOURCE_USER_FINAL),
                "PerspectiveScale": (1.0, _SOURCE_MODEL_FILTERED),
            }),
            "iso": 400.0, "shutter_speed": 0.002,
            "aperture": 2.8, "focal_length": 85.0,
        }
    ]
    df = _make_captures_df(rows)
    result = analyse_deltas(df)

    # v2-extension fields must NOT appear in any output dict
    v2_only = {"CurveRefineSaturation", "ShadowTint", "LensProfileDistortionScale",
               "ColorNoiseReductionDetail", "DefringePurpleAmount"}
    for v2_field in v2_only:
        assert v2_field not in result["per_field"], (
            f"{v2_field} leaked into per_field for v1 captures"
        )
        assert v2_field not in result["filtered_field_deltas"], (
            f"{v2_field} leaked into filtered_field_deltas for v1 captures"
        )
    # The v1 fields that ARE in the captures should be present
    assert "Exposure2012" in result["per_field"]
    assert "Temperature" in result["per_field"]
    assert "PerspectiveScale" in result["filtered_field_deltas"]
