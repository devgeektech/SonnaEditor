"""Structural tests for config.py — verifies SLIDER_FIELDS/RANGES/WEIGHTS consistency."""
from __future__ import annotations

from pathlib import Path

import pytest

import sonna_editor.config as config
from sonna_editor.config import SLIDER_FIELDS, SLIDER_LOSS_WEIGHTS, SLIDER_RANGES


class TestPlatformPaths:
    def test_dng_converter_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SONNA_DNG_CONVERTER lets every OS point at a custom converter path."""
        custom_path = Path("/custom/dng-converter")
        monkeypatch.setenv(config.DNG_CONVERTER_ENV_VAR, str(custom_path))
        assert config._default_dng_converter_path() == custom_path

    def test_app_state_defaults_to_project_root(self) -> None:
        assert config.APP_STATE_DIR == config.PROJECT_ROOT / ".saha"

    def test_training_workspace_defaults_to_project_data(self) -> None:
        assert config.TRAINING_WORKSPACE_DIR == config.DATA_DIR / "training_workspace"

    def test_original_train_parquet_uses_training_workspace(self) -> None:
        assert config.ORIGINAL_TRAIN_PARQUET == (
            config.TRAINING_WORKSPACE_DIR
            / "sonna_personal_001_dataset"
            / "splits_v2_stratified"
            / "train.parquet"
        )

    def test_foundation_repo_defaults_to_project_child(self) -> None:
        assert config.FOUNDATION_REPO_DIR == config.PROJECT_ROOT / "SonnaEditorFoundation"

    def test_training_sources_default_to_project_data(self) -> None:
        assert config.TRAINING_SOURCES_DIR == config.DATA_DIR / "training_sources"

    def test_ensure_runtime_directories_creates_repo_layout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data_dir = tmp_path / "data"
        app_state_dir = tmp_path / ".saha"
        checkpoints_dir = tmp_path / "v1_learning"
        foundation_repo_dir = tmp_path / "SonnaEditorFoundation"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "RAW_DIR", data_dir / "raw")
        monkeypatch.setattr(config, "RAW_TRAINING_DIR", data_dir / "raw" / "sonna_training")
        monkeypatch.setattr(config, "DIGITAL_DATASETS_DIR", data_dir / "datasets")
        monkeypatch.setattr(config, "TRAINING_SOURCES_DIR", data_dir / "training_sources")
        monkeypatch.setattr(config, "DNG_DIR", data_dir / "dng")
        monkeypatch.setattr(config, "PARQUET_DIR", data_dir / "parquet")
        monkeypatch.setattr(config, "THUMBNAIL_DIR", data_dir / "thumbnails")
        monkeypatch.setattr(config, "CAPTURES_DIR", data_dir / "captures")
        monkeypatch.setattr(config, "AUDITS_DIR", data_dir / "audits")
        monkeypatch.setattr(config, "DEBUG_DIR", data_dir / "dbg")
        monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints_dir)
        monkeypatch.setattr(config, "FOUNDATION_REPO_DIR", foundation_repo_dir)
        monkeypatch.setattr(config, "APP_STATE_DIR", app_state_dir)
        monkeypatch.setattr(config, "JOBS_DIR", app_state_dir / "jobs")
        monkeypatch.setattr(config, "PROFILE_TRAINING_RUNS_DIR", app_state_dir / "profile_training_runs")
        monkeypatch.setattr(config, "FINETUNE_RUNS_DIR", app_state_dir / "finetune_runs")

        config.ensure_runtime_directories()

        expected_dirs = [
            data_dir,
            config.RAW_DIR,
            config.RAW_TRAINING_DIR,
            config.DIGITAL_DATASETS_DIR,
            config.TRAINING_SOURCES_DIR,
            config.DNG_DIR,
            config.PARQUET_DIR,
            config.THUMBNAIL_DIR,
            config.CAPTURES_DIR,
            config.AUDITS_DIR,
            config.DEBUG_DIR,
            config.TRAINING_WORKSPACE_DIR,
            foundation_repo_dir,
            checkpoints_dir,
            app_state_dir,
            config.JOBS_DIR,
            config.PROFILE_TRAINING_RUNS_DIR,
            config.FINETUNE_RUNS_DIR,
        ]
        for path in expected_dirs:
            assert path.is_dir(), f"Expected directory to exist: {path}"


class TestRawExtensions:
    def test_supported_raw_extensions_cover_target_formats(self) -> None:
        expected = {
            ".cr2",
            ".cr3",
            ".nef",
            ".arw",
            ".raf",
            ".orf",
            ".rw2",
            ".pef",
            ".dng",
            ".x3f",
            ".rwl",
            ".srw",
        }
        assert config.SUPPORTED_RAW_EXTENSIONS == expected

    def test_inference_pipeline_uses_config_raw_extensions(self) -> None:
        from sonna_editor.inference.pipeline import RAW_EXTENSIONS

        assert RAW_EXTENSIONS == frozenset(config.SUPPORTED_RAW_EXTENSIONS)


class TestSliderFields:
    def test_count(self) -> None:
        assert len(SLIDER_FIELDS) == 147

    def test_no_duplicates(self) -> None:
        assert len(SLIDER_FIELDS) == len(set(SLIDER_FIELDS)), "Duplicate field names found"

    def test_temperature_at_index_11(self) -> None:
        assert SLIDER_FIELDS[11] == "Temperature"

    def test_group_order(self) -> None:
        # First 37 fields are unchanged from v1
        assert SLIDER_FIELDS[0] == "Exposure2012"
        assert SLIDER_FIELDS[7] == "Dehaze"           # end of Tone
        assert SLIDER_FIELDS[8] == "Texture"          # start of Presence
        assert SLIDER_FIELDS[10] == "Saturation"      # end of Presence
        assert SLIDER_FIELDS[12] == "Tint"            # end of WB
        assert SLIDER_FIELDS[13] == "HueAdjustmentRed"
        assert SLIDER_FIELDS[36] == "LuminanceAdjustmentMagenta"  # end of HSL
        # New groups start at 37
        assert SLIDER_FIELDS[37] == "ParametricHighlights"
        assert SLIDER_FIELDS[43] == "ParametricShadowSplit"       # end of Parametric
        # Color Grading (14): Shadow/Highlight Hue+Sat use SplitToning XMP names
        assert SLIDER_FIELDS[44] == "SplitToningShadowHue"
        assert SLIDER_FIELDS[56] == "ColorGradeGlobalLum"
        assert SLIDER_FIELDS[57] == "SplitToningBalance"          # end of ColorGrading
        # Calibration (6): LR uses short names without CameraCalibration prefix
        assert SLIDER_FIELDS[58] == "RedHue"
        assert SLIDER_FIELDS[63] == "BlueSaturation"              # end of Calibration
        # Sharpening (4)
        assert SLIDER_FIELDS[64] == "Sharpness"
        assert SLIDER_FIELDS[67] == "SharpenEdgeMasking"          # end of Sharpening
        # Noise (4)
        assert SLIDER_FIELDS[68] == "LuminanceSmoothing"
        assert SLIDER_FIELDS[71] == "ColorNoiseReduction"         # end of Noise
        # Effects (8): full vignette controls + full grain controls
        assert SLIDER_FIELDS[72] == "PostCropVignetteAmount"
        assert SLIDER_FIELDS[75] == "PostCropVignetteFeather"
        assert SLIDER_FIELDS[76] == "PostCropVignetteHighlightContrast"
        assert SLIDER_FIELDS[77] == "GrainAmount"
        assert SLIDER_FIELDS[79] == "GrainFrequency"              # end of Effects
        # Lens (2)
        assert SLIDER_FIELDS[80] == "LensManualDistortionAmount"
        assert SLIDER_FIELDS[81] == "VignetteAmount"              # end of Lens
        # Transform (5)
        assert SLIDER_FIELDS[82] == "PerspectiveVertical"
        assert SLIDER_FIELDS[86] == "PerspectiveAspect"           # end of Transform
        # Tone Curves (48): composite first, then R/G/B — each channel = 12 fields (Pt1-6 × X/Y)
        assert SLIDER_FIELDS[87] == "ToneCurve_Pt1_X"            # start of composite
        assert SLIDER_FIELDS[98] == "ToneCurve_Pt6_Y"            # end of composite
        assert SLIDER_FIELDS[99] == "ToneCurveRed_Pt1_X"         # start of Red channel
        assert SLIDER_FIELDS[110] == "ToneCurveRed_Pt6_Y"        # end of Red channel
        assert SLIDER_FIELDS[111] == "ToneCurveGreen_Pt1_X"      # start of Green channel
        assert SLIDER_FIELDS[122] == "ToneCurveGreen_Pt6_Y"      # end of Green channel
        assert SLIDER_FIELDS[123] == "ToneCurveBlue_Pt1_X"       # start of Blue channel
        assert SLIDER_FIELDS[134] == "ToneCurveBlue_Pt6_Y"       # last field

    def test_tone_curve_fields_at_idx_87_134(self) -> None:
        curve_fields = [f for f in SLIDER_FIELDS if f.startswith("ToneCurve")]
        assert len(curve_fields) == 48
        assert SLIDER_FIELDS[87:135] == curve_fields, "Tone curve fields must occupy idx 87-134"

    def test_tone_curve_ranges_are_0_255(self) -> None:
        for field in SLIDER_FIELDS:
            if field.startswith("ToneCurve"):
                assert SLIDER_RANGES[field] == (0.0, 255.0), (
                    f"{field} range should be (0.0, 255.0)"
                )

    def test_colorgrade_uses_abbreviated_lum_not_luminance(self) -> None:
        """Real LR XMP uses abbreviated Lum suffix — Luminance would never round-trip."""
        for field in SLIDER_FIELDS:
            if "ColorGrade" in field:
                assert "Luminance" not in field, (
                    f"{field} uses full 'Luminance' — must use abbreviated 'Lum'"
                )


class TestSliderRanges:
    def test_covers_all_slider_fields(self) -> None:
        missing = [f for f in SLIDER_FIELDS if f not in SLIDER_RANGES]
        assert missing == [], f"SLIDER_RANGES missing entries for: {missing}"

    def test_non_standard_ranges(self) -> None:
        assert SLIDER_RANGES["Exposure2012"] == (-5.0, 5.0)
        assert SLIDER_RANGES["Temperature"] == (2000.0, 50000.0)
        assert SLIDER_RANGES["Tint"] == (-150.0, 150.0)
        assert SLIDER_RANGES["Sharpness"] == (0.0, 150.0)
        assert SLIDER_RANGES["SharpenRadius"] == (0.5, 3.0)
        assert SLIDER_RANGES["PerspectiveRotate"] == (-10.0, 10.0)
        assert SLIDER_RANGES["PerspectiveScale"] == (50.0, 150.0)

    def test_colorgrade_hue_range_is_360(self) -> None:
        for field in ["SplitToningShadowHue", "ColorGradeMidtoneHue",
                      "SplitToningHighlightHue", "ColorGradeGlobalHue"]:
            lo, hi = SLIDER_RANGES[field]
            assert lo == 0.0 and hi == 360.0, f"{field} should be (0, 360)"

    def test_colorgrade_sat_and_blending_non_negative(self) -> None:
        for field in ["SplitToningShadowSaturation", "ColorGradeMidtoneSat",
                      "SplitToningHighlightSaturation", "ColorGradeGlobalSat",
                      "ColorGradeBlending"]:
            lo, _ = SLIDER_RANGES[field]
            assert lo == 0.0, f"{field} lower bound should be 0"

    def test_all_ranges_lo_less_than_hi(self) -> None:
        for field, (lo, hi) in SLIDER_RANGES.items():
            assert lo < hi, f"{field}: lo={lo} >= hi={hi}"


class TestSliderLossWeights:
    def test_covers_all_slider_fields(self) -> None:
        missing = [f for f in SLIDER_FIELDS if f not in SLIDER_LOSS_WEIGHTS]
        assert missing == [], f"SLIDER_LOSS_WEIGHTS missing entries for: {missing}"

    def test_c3k_tuned_weights(self) -> None:
        """v1.1.0-c3k-tuned weight bumps. See config.py `_TUNED_TIMID_BUMPS`
        and the full-loss audit at /tmp/saha_full_loss_audit.md.

        Baseline retained:
          Temperature=4.0, Tint=4.0, Composite/R/G/B tone curve Pt2-5_Y=3.0
          (Blue reduced to 2.0), HSL Hue/Sat/Lum=1.5, Camera Calibration
          Red/Green/BlueHue=2.0 (Blue/Green Hue then bumped to 3.0).

        Tuned bumps (TIMID 1.5×):
          11 scalar fields → 1.5 (was 1.0)
          5 HueAdjustment + 3 LuminanceAdjustment → 2.25 (was 1.5)
          BlueHue, GreenHue → 3.0 (was 2.0)
          ToneCurveGreen_Pt5_Y, ToneCurveRed_Pt2_Y → 4.5 (was 3.0)
          ToneCurveBlue Pt2-5 Y → 2.0 (was 3.0, REDUCED)
        """
        # Baseline retained
        assert SLIDER_LOSS_WEIGHTS["Exposure2012"] == 5.0
        assert SLIDER_LOSS_WEIGHTS["Temperature"] == 4.0
        assert SLIDER_LOSS_WEIGHTS["Tint"] == 4.0
        assert SLIDER_LOSS_WEIGHTS["RedHue"] == 2.0
        # Composite + R/G tone curve Pt2-5_Y still at 3.0 (except R/G/B-specific bumps below)
        for prefix in ("ToneCurve", "ToneCurveRed", "ToneCurveGreen"):
            for pt in (2, 3, 4, 5):
                expected = 4.5 if (prefix, pt) in {("ToneCurveRed", 2), ("ToneCurveGreen", 5)} else 3.0
                assert SLIDER_LOSS_WEIGHTS[f"{prefix}_Pt{pt}_Y"] == expected
        # Blue Y → 2.0 (reduced from 3.0)
        for pt in (2, 3, 4, 5):
            assert SLIDER_LOSS_WEIGHTS[f"ToneCurveBlue_Pt{pt}_Y"] == 2.0

        # Tuned HueAdjustment bumps: Red/Yellow/Orange/Aqua/Blue → 2.25; Green/Purple/Magenta stay 1.5
        for c in ("Red", "Orange", "Yellow", "Aqua", "Blue"):
            assert SLIDER_LOSS_WEIGHTS[f"HueAdjustment{c}"] == 2.25
        for c in ("Green", "Purple", "Magenta"):
            assert SLIDER_LOSS_WEIGHTS[f"HueAdjustment{c}"] == 1.5

        # Saturation: not bumped (direction-at-chance Red/Yellow/Green deferred)
        for c in ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"):
            assert SLIDER_LOSS_WEIGHTS[f"SaturationAdjustment{c}"] == 1.5

        # Luminance: Orange/Green/Blue bumped → 2.25; others stay 1.5
        for c in ("Orange", "Green", "Blue"):
            assert SLIDER_LOSS_WEIGHTS[f"LuminanceAdjustment{c}"] == 2.25
        for c in ("Red", "Yellow", "Aqua", "Purple", "Magenta"):
            assert SLIDER_LOSS_WEIGHTS[f"LuminanceAdjustment{c}"] == 1.5

        # Camera Calibration hue: Red 2.0, Blue/Green bumped 3.0
        assert SLIDER_LOSS_WEIGHTS["BlueHue"] == 3.0
        assert SLIDER_LOSS_WEIGHTS["GreenHue"] == 3.0

        # Visual-priority tone-panel scalars are protected by minimum weights.
        assert SLIDER_LOSS_WEIGHTS["Blacks2012"] == 2.0
        assert SLIDER_LOSS_WEIGHTS["Clarity2012"] == 1.5
        assert SLIDER_LOSS_WEIGHTS["Contrast2012"] == 3.0
        assert SLIDER_LOSS_WEIGHTS["Highlights2012"] == 3.0
        assert SLIDER_LOSS_WEIGHTS["Shadows2012"] == 3.0
        assert SLIDER_LOSS_WEIGHTS["Whites2012"] == 2.0
        assert SLIDER_LOSS_WEIGHTS["Saturation"] == 2.0
        assert SLIDER_LOSS_WEIGHTS["Vibrance"] == 2.0

        # Color-grading bumps
        for f in ("ColorGradeHighlightLum", "ColorGradeMidtoneHue",
                  "ColorGradeMidtoneSat", "ColorGradeShadowLum",
                  "SplitToningHighlightHue", "SplitToningHighlightSaturation"):
            assert SLIDER_LOSS_WEIGHTS[f] == 1.5

        # Direction-at-chance fields explicitly NOT bumped beyond visual-priority floors.
        assert SLIDER_LOSS_WEIGHTS["Saturation"] == 2.0
        assert SLIDER_LOSS_WEIGHTS["GrainFrequency"] == 1.0
        assert SLIDER_LOSS_WEIGHTS["ToneCurve_Pt4_Y"] == 3.0  # curve default, not bumped further
        assert SLIDER_LOSS_WEIGHTS["ToneCurve_Pt4_X"] == 1.0  # X default, not bumped

        # Visual-priority exposure floor.
        assert SLIDER_LOSS_WEIGHTS["Exposure2012"] == 5.0

    def test_count_matches_slider_fields(self) -> None:
        assert len(SLIDER_LOSS_WEIGHTS) == len(SLIDER_FIELDS)


class TestV2Extension:
    """Locked-append-only v2 extension fields (idx 135-146, commit 3d0d90c)."""

    def test_v2_extension_indices_are_locked(self) -> None:
        """idx 135-146 must be exactly these 12 fields in this order — forever."""
        expected = [
            "ColorNoiseReductionDetail",
            "ColorNoiseReductionSmoothness",
            "DefringePurpleAmount",
            "DefringePurpleHueLo",
            "DefringePurpleHueHi",
            "DefringeGreenAmount",
            "DefringeGreenHueLo",
            "DefringeGreenHueHi",
            "LensProfileDistortionScale",
            "LensProfileVignettingScale",
            "ShadowTint",
            "CurveRefineSaturation",
        ]
        assert SLIDER_FIELDS[135:] == expected

    def test_slider_defaults_keys_match_v2_extension(self) -> None:
        from sonna_editor.config import SLIDER_DEFAULTS
        assert set(SLIDER_DEFAULTS.keys()) == set(SLIDER_FIELDS[135:])

    def test_lr_defaults_covers_all_slider_fields(self) -> None:
        """Regression guard for the bug fixed by commit a2d3c81 (LR_DEFAULTS
        must contain a default for every SLIDER_FIELDS entry — xmp.py asserts
        this at import time)."""
        from sonna_editor.data.xmp import LR_DEFAULTS
        missing = set(SLIDER_FIELDS) - set(LR_DEFAULTS)
        assert not missing, f"LR_DEFAULTS missing: {missing}"

    def test_slider_defaults_values_match_lr_defaults(self) -> None:
        """The two tables are duplicate sources of truth (flagged for future
        consolidation in commit a2d3c81); this test ensures they don't drift."""
        from sonna_editor.config import SLIDER_DEFAULTS
        from sonna_editor.data.xmp import LR_DEFAULTS
        for field, default in SLIDER_DEFAULTS.items():
            assert LR_DEFAULTS[field] == default, (
                f"drift detected: SLIDER_DEFAULTS[{field!r}]={default} "
                f"!= LR_DEFAULTS[{field!r}]={LR_DEFAULTS[field]}"
            )
