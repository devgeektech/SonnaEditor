"""Tests for finetune/capture.py — edit capture and delta tracking."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sonna_editor import config
from sonna_editor.finetune.capture import (
    _SOURCE_LR_DEFAULT,
    _SOURCE_MODEL_FILTERED,
    _SOURCE_USER_FINAL,
    _build_deltas_json,
    _compute_edit_lag,
    _histogram_to_bytes,
)
from sonna_editor.slider_set import fields_for_version, v1_fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_fields_predicted() -> dict[str, float]:
    """A predicted values dict with a distinct value for every SLIDER_FIELD."""
    return {f: float(i) * 0.5 for i, f in enumerate(config.SLIDER_FIELDS)}


@pytest.fixture
def v1_skip() -> frozenset[str]:
    return frozenset({"PerspectiveScale", "PerspectiveVertical", "LensManualDistortionAmount"})


# ---------------------------------------------------------------------------
# _build_deltas_json
# ---------------------------------------------------------------------------

class TestBuildDeltasJson:
    def test_user_final_source_and_delta(self, all_fields_predicted, v1_skip):
        """Field present in final XMP → source=user_final, correct delta."""
        final = {"Exposure2012": 1.0}
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        entry = result["Exposure2012"]
        assert entry["source"] == _SOURCE_USER_FINAL
        expected_delta = 1.0 - all_fields_predicted["Exposure2012"]
        assert abs(entry["delta"] - expected_delta) < 1e-6

    def test_model_filtered_source(self, all_fields_predicted, v1_skip):
        """Field in v1_skip_fields + absent from final XMP → source=model_filtered, delta=None."""
        final: dict = {}  # PerspectiveScale not present
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        entry = result["PerspectiveScale"]
        assert entry["source"] == _SOURCE_MODEL_FILTERED
        assert entry["delta"] is None

    def test_lr_default_source(self, all_fields_predicted, v1_skip):
        """Field absent from XMP and NOT in v1_skip_fields → source=lr_default, delta=None."""
        final: dict = {}
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        entry = result["Exposure2012"]
        assert entry["source"] == _SOURCE_LR_DEFAULT
        assert entry["delta"] is None

    def test_all_fields_present(self, all_fields_predicted, v1_skip):
        """Every SLIDER_FIELD must appear in the output dict."""
        final: dict = {}
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        assert set(result.keys()) == set(config.SLIDER_FIELDS)
        assert len(result) == len(config.SLIDER_FIELDS)

    def test_negative_delta(self, all_fields_predicted, v1_skip):
        """Delta is correctly negative when final < predicted."""
        # Exposure2012 predicted = 0.0 (index 0 × 0.5 = 0.0)
        final = {"Exposure2012": -0.5}
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        entry = result["Exposure2012"]
        assert entry["source"] == _SOURCE_USER_FINAL
        assert entry["delta"] == pytest.approx(-0.5 - 0.0, abs=1e-6)

    def test_skip_field_with_final_value_is_user_final(self, all_fields_predicted, v1_skip):
        """If a skip field IS present in the final XMP, it's still user_final."""
        # This can happen if the user manually adds a perspective value in LR.
        final = {"PerspectiveScale": 100.0}
        result = json.loads(_build_deltas_json(all_fields_predicted, final, v1_skip))
        entry = result["PerspectiveScale"]
        assert entry["source"] == _SOURCE_USER_FINAL
        assert entry["delta"] is not None


# ---------------------------------------------------------------------------
# _histogram_to_bytes
# ---------------------------------------------------------------------------

class TestHistogramToBytes:
    def test_roundtrip_shape(self):
        """Serialised histogram must deserialise back to (3, 32)."""
        hist = np.random.rand(3, 32).astype(np.float32)
        data = _histogram_to_bytes(hist)
        recovered = np.load(io.BytesIO(data))
        assert recovered.shape == (3, 32)

    def test_roundtrip_values(self):
        """Values must be bit-exact after serialise/deserialise."""
        hist = np.ones((3, 32), dtype=np.float32) * 0.031
        data = _histogram_to_bytes(hist)
        recovered = np.load(io.BytesIO(data))
        np.testing.assert_array_equal(hist, recovered)


# ---------------------------------------------------------------------------
# _compute_edit_lag
# ---------------------------------------------------------------------------

class TestComputeEditLag:
    def test_positive_lag(self):
        """XMP modified after inference → positive lag."""
        run_ts = "2026-05-10T08:00:00+00:00"
        xmp_mtime = datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)
        lag = _compute_edit_lag(run_ts, xmp_mtime)
        assert lag == pytest.approx(7200.0, abs=1.0)

    def test_negative_lag(self):
        """XMP modified before inference → negative lag (pre-existing XMP)."""
        run_ts = "2026-05-10T10:00:00+00:00"
        xmp_mtime = datetime(2026, 5, 10, 8, 0, 0, tzinfo=timezone.utc)
        lag = _compute_edit_lag(run_ts, xmp_mtime)
        assert lag == pytest.approx(-7200.0, abs=1.0)

    def test_none_when_mtime_missing(self):
        """Returns None when xmp_mtime is None."""
        assert _compute_edit_lag("2026-05-10T08:00:00", None) is None

    def test_none_when_timestamp_invalid(self):
        """Returns None when run_timestamp is malformed."""
        xmp_mtime = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        assert _compute_edit_lag("not-a-date", xmp_mtime) is None


# ---------------------------------------------------------------------------
# capture_user_edits integration (using tmp filesystem — no real RAWs)
# ---------------------------------------------------------------------------

class TestCaptureUserEditsIntegration:
    def _make_predictions_json(
        self,
        tmp_path: Path,
        filenames: list[str],
        v1_skip: list[str] | None = None,
    ) -> Path:
        """Write a minimal sonna_predictions.json for the given filenames."""
        photos = {
            fn: {f: float(i) * 0.1 for i, f in enumerate(config.SLIDER_FIELDS)}
            for fn in filenames
        }
        sidecar = {
            "model_path": "/fake/model.ckpt",
            "model_version": "model-v1.0.1",
            "run_timestamp": "2026-05-10T08:00:00+00:00",
            "v1_skip_fields": v1_skip or [],
            "slider_fields": list(config.SLIDER_FIELDS),
            "photos": photos,
        }
        pred_path = tmp_path / "sonna_predictions.json"
        pred_path.write_text(json.dumps(sidecar))
        return pred_path

    def test_missing_xmp_skipped(self, tmp_path):
        """RAW with no adjacent XMP should not appear in the output DataFrame."""
        from sonna_editor.finetune.capture import capture_user_edits

        shoot_dir = tmp_path / "shoot"
        shoot_dir.mkdir()
        # Create a fake RAW file (just needs to exist with right extension)
        (shoot_dir / "IMG_0001.CR3").touch()

        pred_path = self._make_predictions_json(tmp_path, ["IMG_0001.CR3"])
        output_dir = tmp_path / "output"

        # capture_user_edits will try extract_preview/extract_metadata on the fake file
        # but should skip it because there's no XMP (not because extraction fails).
        # We patch the extract functions to avoid needing real RAW data.
        import unittest.mock as mock
        with mock.patch("sonna_editor.finetune.capture.extract_preview") as mock_prev, \
             mock.patch("sonna_editor.finetune.capture.extract_metadata") as mock_meta, \
             mock.patch("sonna_editor.finetune.capture.read_xmp") as mock_xmp:

            mock_prev.return_value = _make_fake_pil_image()
            mock_meta.return_value = _make_fake_meta()
            mock_xmp.return_value = {}

            df = capture_user_edits(shoot_dir, pred_path, output_dir)

        # No XMP exists next to the RAW → row should be skipped
        assert len(df) == 0

    def test_row_has_all_slider_columns(self, tmp_path):
        """When a photo has a matching prediction + XMP, all 135 slider cols must be present."""
        from sonna_editor.finetune.capture import capture_user_edits

        shoot_dir = tmp_path / "shoot"
        shoot_dir.mkdir()
        raw_path = shoot_dir / "IMG_0001.CR3"
        raw_path.touch()
        xmp_path = shoot_dir / "IMG_0001.xmp"
        xmp_path.touch()

        pred_path = self._make_predictions_json(tmp_path, ["IMG_0001.CR3"])
        output_dir = tmp_path / "output"

        final_sliders = {f: float(i) * 0.2 for i, f in enumerate(config.SLIDER_FIELDS)}

        import unittest.mock as mock
        with mock.patch("sonna_editor.finetune.capture.extract_preview") as mock_prev, \
             mock.patch("sonna_editor.finetune.capture.extract_metadata") as mock_meta, \
             mock.patch("sonna_editor.finetune.capture.read_xmp") as mock_xmp:

            mock_prev.return_value = _make_fake_pil_image()
            mock_meta.return_value = _make_fake_meta()
            mock_xmp.return_value = final_sliders

            df = capture_user_edits(shoot_dir, pred_path, output_dir)

        assert len(df) == 1
        for field in config.SLIDER_FIELDS:
            assert field in df.columns, f"Missing slider column: {field}"

    def test_deltas_json_has_correct_sources(self, tmp_path):
        """Deltas JSON must tag fields correctly based on XMP presence and skip list."""
        from sonna_editor.finetune.capture import capture_user_edits

        shoot_dir = tmp_path / "shoot"
        shoot_dir.mkdir()
        raw_path = shoot_dir / "IMG_0001.CR3"
        raw_path.touch()
        xmp_path = shoot_dir / "IMG_0001.xmp"
        xmp_path.touch()

        skip_field = "PerspectiveScale"
        non_skip_absent = "GrainAmount"
        written_field = "Exposure2012"

        pred_path = self._make_predictions_json(
            tmp_path, ["IMG_0001.CR3"], v1_skip=[skip_field]
        )
        output_dir = tmp_path / "output"

        # Only Exposure2012 in final XMP; PerspectiveScale and GrainAmount absent
        final_sliders = {written_field: 0.5}

        import unittest.mock as mock
        with mock.patch("sonna_editor.finetune.capture.extract_preview") as mock_prev, \
             mock.patch("sonna_editor.finetune.capture.extract_metadata") as mock_meta, \
             mock.patch("sonna_editor.finetune.capture.read_xmp") as mock_xmp:

            mock_prev.return_value = _make_fake_pil_image()
            mock_meta.return_value = _make_fake_meta()
            mock_xmp.return_value = final_sliders

            df = capture_user_edits(shoot_dir, pred_path, output_dir)

        assert len(df) == 1
        deltas = json.loads(df.iloc[0]["deltas"])
        assert deltas[written_field]["source"] == _SOURCE_USER_FINAL
        assert deltas[skip_field]["source"] == _SOURCE_MODEL_FILTERED
        assert deltas[non_skip_absent]["source"] == _SOURCE_LR_DEFAULT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_pil_image():
    from PIL import Image
    return Image.new("RGB", (384, 384), color=(128, 100, 80))


def _make_fake_meta() -> dict:
    return {
        "iso": 400.0,
        "shutter_speed": 0.002,
        "aperture": 2.8,
        "focal_length": 85.0,
        "lens_model": "Test Lens 85mm",
        "camera_body": "Canon EOS R5",
        "capture_datetime": datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc),
        "exposure_compensation": 0.0,
        "white_balance_preset": "Auto",
        "camera_profile": None,
        "width": 8192,
        "height": 5464,
    }


# ---------------------------------------------------------------------------
# v1/v2 shape-mismatch regression coverage
# ---------------------------------------------------------------------------
# Before today's slider_set helper migration, _build_deltas_json iterated
# config.SLIDER_FIELDS (147) and looked up predicted[field]; for a v1
# predictions dict (135 keys) the 12 v2-extension fields fell through the
# KeyError catch and got marked source="lr_default", silently mis-attributing
# every captured edit for v1 profiles. These tests pin the fixed behaviour:
# the result dict's field set matches the predicted dict's field set exactly.

class TestBuildDeltasJsonVersionAware:
    def test_v1_predicted_returns_135_keys_no_v2_extensions(self) -> None:
        """v1 predictions dict (135 keys) → output has exactly 135 keys, no v2 fields."""
        v1 = v1_fields()
        predicted = {f: float(i) * 0.5 for i, f in enumerate(v1)}
        final: dict = {}  # all fields fall through to lr_default
        result = json.loads(_build_deltas_json(predicted, final, frozenset()))
        assert set(result.keys()) == set(v1)
        assert len(result) == 135
        # Explicit guard: no v2-extension fields should appear in v1 deltas
        assert "CurveRefineSaturation" not in result
        assert "ShadowTint" not in result
        assert "LensProfileDistortionScale" not in result

    def test_v2_predicted_returns_147_keys_includes_extensions(self) -> None:
        """v2 predictions dict (147 keys) → output has all 147 fields including v2 extensions."""
        v2 = fields_for_version("v2")
        predicted = {f: float(i) * 0.5 for i, f in enumerate(v2)}
        final: dict = {}
        result = json.loads(_build_deltas_json(predicted, final, frozenset()))
        assert set(result.keys()) == set(v2)
        assert len(result) == 147
        # v2-extension fields must be present
        assert "CurveRefineSaturation" in result
        assert "ShadowTint" in result
