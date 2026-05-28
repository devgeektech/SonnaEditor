from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sonna_editor.config import IMAGE_RESOLUTION, SLIDER_FIELDS
from sonna_editor.data.extract import (
    compute_histogram,
    extract_all,
    extract_metadata,
    extract_preview,
)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "sample.cr3"
FIXTURE_XMP = Path(__file__).parent / "fixtures" / "sample.xmp"


# -----------------------------------------------------------------------
# extract_preview
# -----------------------------------------------------------------------

class TestExtractPreview:
    def test_returns_pil_image(self):
        img = extract_preview(FIXTURE_RAW)
        assert isinstance(img, Image.Image)

    def test_mode_is_rgb(self):
        img = extract_preview(FIXTURE_RAW)
        assert img.mode == "RGB"

    def test_long_edge_is_target(self):
        img = extract_preview(FIXTURE_RAW, target_size=384)
        assert max(img.size) == 384

    def test_aspect_ratio_preserved(self):
        img_small = extract_preview(FIXTURE_RAW, target_size=384)
        img_large = extract_preview(FIXTURE_RAW, target_size=512)
        ratio_small = img_small.width / img_small.height
        ratio_large = img_large.width / img_large.height
        assert abs(ratio_small - ratio_large) < 0.02

    def test_under_500ms(self):
        # Cold read of a 21MB CR3 on M1 Pro — rawpy header parse + JPEG extract.
        # Target is <100ms on warm cache; 500ms covers cold I/O in CI.
        extract_preview(FIXTURE_RAW)  # warm up OS cache
        start = time.perf_counter()
        extract_preview(FIXTURE_RAW)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"extract_preview took {elapsed_ms:.0f}ms, expected <500ms"

    def test_default_target_is_config_resolution(self):
        img = extract_preview(FIXTURE_RAW)
        assert max(img.size) == IMAGE_RESOLUTION


# -----------------------------------------------------------------------
# extract_metadata
# -----------------------------------------------------------------------

class TestExtractMetadata:
    def setup_method(self):
        self.meta = extract_metadata(FIXTURE_RAW)

    def test_returns_dict(self):
        assert isinstance(self.meta, dict)

    def test_required_keys_present(self):
        required = {
            "iso", "shutter_speed", "aperture", "focal_length",
            "lens_model", "camera_body", "capture_datetime",
            "exposure_compensation", "white_balance_preset",
            "camera_profile", "width", "height",
        }
        assert required.issubset(self.meta.keys())

    def test_iso_is_int(self):
        assert isinstance(self.meta["iso"], int)
        assert self.meta["iso"] == 160

    def test_shutter_speed_is_float(self):
        assert isinstance(self.meta["shutter_speed"], float)
        assert abs(self.meta["shutter_speed"] - 1 / 125) < 0.0001

    def test_aperture_is_float(self):
        assert isinstance(self.meta["aperture"], float)
        assert self.meta["aperture"] == pytest.approx(4.0)

    def test_focal_length_is_float(self):
        assert isinstance(self.meta["focal_length"], float)
        assert 24 <= self.meta["focal_length"] <= 70

    def test_camera_body_contains_model(self):
        assert self.meta["camera_body"] is not None
        assert "R6" in self.meta["camera_body"]

    def test_dimensions_are_positive(self):
        assert isinstance(self.meta["width"], int)
        assert isinstance(self.meta["height"], int)
        assert self.meta["width"] > 0
        assert self.meta["height"] > 0

    def test_capture_datetime_not_none(self):
        assert self.meta["capture_datetime"] is not None

    def test_lens_model_from_xmp(self):
        # lens_model comes from the XMP sidecar (not in basic EXIF)
        if FIXTURE_XMP.exists():
            assert self.meta["lens_model"] is not None
            assert "24-70" in self.meta["lens_model"]


# -----------------------------------------------------------------------
# compute_histogram
# -----------------------------------------------------------------------

class TestComputeHistogram:
    def test_shape(self):
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        hist = compute_histogram(img, bins=32)
        assert hist.shape == (3, 32)

    def test_dtype_float32(self):
        img = Image.new("RGB", (50, 50), (200, 100, 50))
        hist = compute_histogram(img)
        assert hist.dtype == np.float32

    def test_normalised(self):
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        hist = compute_histogram(img, bins=32)
        # Each channel sums to 1.0 (normalised)
        for ch in range(3):
            assert abs(hist[ch].sum() - 1.0) < 1e-5

    def test_custom_bins(self):
        img = Image.new("RGB", (64, 64), (10, 20, 30))
        hist = compute_histogram(img, bins=16)
        assert hist.shape == (3, 16)

    def test_real_image(self):
        img = extract_preview(FIXTURE_RAW, target_size=256)
        hist = compute_histogram(img, bins=32)
        assert hist.shape == (3, 32)
        assert np.all(hist >= 0)
        for ch in range(3):
            assert abs(hist[ch].sum() - 1.0) < 1e-5


# -----------------------------------------------------------------------
# extract_all
# -----------------------------------------------------------------------

class TestExtractAll:
    def test_returns_dict_with_all_keys(self):
        result = extract_all(FIXTURE_RAW, xmp_path=FIXTURE_XMP)
        assert "raw_path" in result
        assert "preview" in result
        assert "histogram" in result
        assert "sliders" in result
        assert "iso" in result

    def test_preview_is_pil_image(self):
        result = extract_all(FIXTURE_RAW)
        assert isinstance(result["preview"], Image.Image)

    def test_histogram_shape(self):
        result = extract_all(FIXTURE_RAW)
        assert result["histogram"].shape == (3, 32)

    def test_sliders_has_all_fields(self):
        result = extract_all(FIXTURE_RAW, xmp_path=FIXTURE_XMP)
        assert set(SLIDER_FIELDS).issubset(result["sliders"].keys())

    def test_sliders_exposure_from_xmp(self):
        # sample.xmp is the real Lightroom export (Exposure2012=+0.60)
        result = extract_all(FIXTURE_RAW, xmp_path=FIXTURE_XMP)
        assert result["sliders"]["Exposure2012"] == pytest.approx(0.60, abs=0.01)

    def test_no_xmp_sliders_are_none(self, tmp_path):
        # Point at a RAW with no sidecar
        raw_copy = tmp_path / "sample.cr3"
        raw_copy.symlink_to(FIXTURE_RAW)
        result = extract_all(raw_copy)
        assert all(v is None for v in result["sliders"].values())

    def test_auto_discovers_sidecar(self, tmp_path):
        # Place raw + xmp in same dir — extract_all should find the xmp
        raw_copy = tmp_path / "sample.cr3"
        xmp_copy = tmp_path / "sample.xmp"
        raw_copy.symlink_to(FIXTURE_RAW)
        xmp_copy.symlink_to(FIXTURE_XMP)
        result = extract_all(raw_copy)
        assert result["sliders"]["Exposure2012"] is not None
