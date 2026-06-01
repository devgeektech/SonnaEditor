import os
import platform
import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DNG_DIR = DATA_DIR / "dng"
PARQUET_DIR = DATA_DIR / "parquet"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
CAPTURES_DIR = DATA_DIR / "captures"
AUDITS_DIR = DATA_DIR / "audits"

# Trained checkpoints directory — scanned by the API's /api/profiles endpoint.
# v1 checkpoints live under v1_learning/ at project root; future Phase 6 profile
# registry will replace this with a manifest-based discovery layer.
CHECKPOINTS_DIR = PROJECT_ROOT / "v1_learning"

# Adobe DNG Converter binary.
# `SONNA_DNG_CONVERTER` always wins so packaged installs and unusual local
# setups can point at the converter without editing source code. The fallback
# list covers the default Adobe installer locations on macOS and Windows, plus
# PATH-based discovery for Linux/Wine or custom installs.
DNG_CONVERTER_ENV_VAR = "SONNA_DNG_CONVERTER"


def _default_dng_converter_path() -> Path:
    """Return the best-known Adobe DNG Converter executable path for this OS."""
    env_path = os.environ.get(DNG_CONVERTER_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    system = platform.system()
    candidates: list[Path] = []
    if system == "Darwin":
        candidates.append(
            Path("/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter")
        )
    elif system == "Windows":
        candidates.extend(
            [
                Path(r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe"),
                Path(r"C:\Program Files (x86)\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for executable in ("Adobe DNG Converter", "Adobe DNG Converter.exe", "dngconverter"):
        found = shutil.which(executable)
        if found:
            return Path(found)

    return candidates[0] if candidates else Path("Adobe DNG Converter")


DNG_CONVERTER_PATH = _default_dng_converter_path()

# Model input resolution
#   v1.0.x (legacy):  384 — value preserved in each ckpt's arch_config so
#                     v1.0.1 / v1.0.2-candidate3k keep running at 384 even
#                     though this global default is now 512.
#   v1.1.0+:          512 — current default for fresh training and thumbnail
#                     extraction. InferenceEngine reads each loaded ckpt's
#                     stored resolution rather than this global so old ckpts
#                     don't regress.
IMAGE_RESOLUTION = 512

# Training-time photometric jitter is disabled by default because targets are
# Lightroom slider values for the original edit. Even mild brightness/colour
# jitter can make the same target describe visibly different exposure/WB states,
# which pushes the model toward mean predictions on Exposure/Temperature/Tint.
# Keep augmentation geometric by default; enable photometric jitter only for
# large, stable datasets where overfitting is the bigger risk than label noise.
TRAIN_AUG_BRIGHTNESS = 0.0
TRAIN_AUG_CONTRAST = 0.0
TRAIN_AUG_SATURATION = 0.0
TRAIN_AUG_HUE = 0.0

# Supported RAW file extensions
SUPPORTED_RAW_EXTENSIONS = {
    ".cr3", ".cr2", ".nef", ".arw", ".raf", ".rw2",
    ".orf", ".pef", ".srw", ".x3f", ".dng",
}

# Slider fields — 147 values in order (must match model output order)
SLIDER_FIELDS: list[str] = [
    # Tone (8)
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Clarity2012",
    "Dehaze",
    # Presence (3)
    "Texture",
    "Vibrance",
    "Saturation",
    # White balance (2) — Temperature at index 11, unchanged
    "Temperature",
    "Tint",
    # HSL Hue (8)
    "HueAdjustmentRed",
    "HueAdjustmentOrange",
    "HueAdjustmentYellow",
    "HueAdjustmentGreen",
    "HueAdjustmentAqua",
    "HueAdjustmentBlue",
    "HueAdjustmentPurple",
    "HueAdjustmentMagenta",
    # HSL Saturation (8)
    "SaturationAdjustmentRed",
    "SaturationAdjustmentOrange",
    "SaturationAdjustmentYellow",
    "SaturationAdjustmentGreen",
    "SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue",
    "SaturationAdjustmentPurple",
    "SaturationAdjustmentMagenta",
    # HSL Luminance (8)
    "LuminanceAdjustmentRed",
    "LuminanceAdjustmentOrange",
    "LuminanceAdjustmentYellow",
    "LuminanceAdjustmentGreen",
    "LuminanceAdjustmentAqua",
    "LuminanceAdjustmentBlue",
    "LuminanceAdjustmentPurple",
    "LuminanceAdjustmentMagenta",
    # Parametric Tone Curve (7)
    "ParametricHighlights",
    "ParametricLights",
    "ParametricDarks",
    "ParametricShadows",
    "ParametricHighlightSplit",
    "ParametricMidtoneSplit",
    "ParametricShadowSplit",
    # Color Grading (14)
    # Shadow/Highlight Hue+Sat use legacy SplitToning XMP names for LR backward compatibility.
    # Midtone wheel and Lum channels use modern ColorGrade names.
    "SplitToningShadowHue",
    "SplitToningShadowSaturation",
    "ColorGradeShadowLum",
    "ColorGradeMidtoneHue",
    "ColorGradeMidtoneSat",
    "ColorGradeMidtoneLum",
    "SplitToningHighlightHue",
    "SplitToningHighlightSaturation",
    "ColorGradeHighlightLum",
    "ColorGradeBlending",
    "ColorGradeGlobalHue",
    "ColorGradeGlobalSat",
    "ColorGradeGlobalLum",
    "SplitToningBalance",
    # Camera Calibration (6) — LR uses short names without "CameraCalibration" prefix
    "RedHue",
    "RedSaturation",
    "GreenHue",
    "GreenSaturation",
    "BlueHue",
    "BlueSaturation",
    # Detail — Sharpening (4)
    "Sharpness",
    "SharpenRadius",
    "SharpenDetail",
    "SharpenEdgeMasking",
    # Detail — Noise Reduction (4)
    "LuminanceSmoothing",
    "LuminanceNoiseReductionDetail",
    "LuminanceNoiseReductionContrast",
    "ColorNoiseReduction",
    # Effects (8) — vignette controls + grain controls
    "PostCropVignetteAmount",
    "PostCropVignetteMidpoint",
    "PostCropVignetteRoundness",
    "PostCropVignetteFeather",
    "PostCropVignetteHighlightContrast",
    "GrainAmount",
    "GrainSize",
    "GrainFrequency",
    # Lens Corrections (2) — LensProfileEnable deferred to v2 (binary flag)
    "LensManualDistortionAmount",
    "VignetteAmount",
    # Transform (5)
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "PerspectiveRotate",
    "PerspectiveScale",
    "PerspectiveAspect",
    # Tone Curves (48) — 4 channels × 6 control points × (X, Y)
    # Composite channel — maps to crs:ToneCurvePV2012
    "ToneCurve_Pt1_X", "ToneCurve_Pt1_Y",
    "ToneCurve_Pt2_X", "ToneCurve_Pt2_Y",
    "ToneCurve_Pt3_X", "ToneCurve_Pt3_Y",
    "ToneCurve_Pt4_X", "ToneCurve_Pt4_Y",
    "ToneCurve_Pt5_X", "ToneCurve_Pt5_Y",
    "ToneCurve_Pt6_X", "ToneCurve_Pt6_Y",
    # Red channel — maps to crs:ToneCurvePV2012Red
    "ToneCurveRed_Pt1_X", "ToneCurveRed_Pt1_Y",
    "ToneCurveRed_Pt2_X", "ToneCurveRed_Pt2_Y",
    "ToneCurveRed_Pt3_X", "ToneCurveRed_Pt3_Y",
    "ToneCurveRed_Pt4_X", "ToneCurveRed_Pt4_Y",
    "ToneCurveRed_Pt5_X", "ToneCurveRed_Pt5_Y",
    "ToneCurveRed_Pt6_X", "ToneCurveRed_Pt6_Y",
    # Green channel — maps to crs:ToneCurvePV2012Green
    "ToneCurveGreen_Pt1_X", "ToneCurveGreen_Pt1_Y",
    "ToneCurveGreen_Pt2_X", "ToneCurveGreen_Pt2_Y",
    "ToneCurveGreen_Pt3_X", "ToneCurveGreen_Pt3_Y",
    "ToneCurveGreen_Pt4_X", "ToneCurveGreen_Pt4_Y",
    "ToneCurveGreen_Pt5_X", "ToneCurveGreen_Pt5_Y",
    "ToneCurveGreen_Pt6_X", "ToneCurveGreen_Pt6_Y",
    # Blue channel — maps to crs:ToneCurvePV2012Blue
    "ToneCurveBlue_Pt1_X", "ToneCurveBlue_Pt1_Y",
    "ToneCurveBlue_Pt2_X", "ToneCurveBlue_Pt2_Y",
    "ToneCurveBlue_Pt3_X", "ToneCurveBlue_Pt3_Y",
    "ToneCurveBlue_Pt4_X", "ToneCurveBlue_Pt4_Y",
    "ToneCurveBlue_Pt5_X", "ToneCurveBlue_Pt5_Y",
    "ToneCurveBlue_Pt6_X", "ToneCurveBlue_Pt6_Y",
    # === v2 SLIDER LIST EXTENSION — locked-append-only, idx 135-146 ===
    # Verified against real LR Classic 15.3 XMP (Canon R6 + RF24-70mm,
    # ProcessVersion 15.4, 2026-05-13). See HANDOVER Decision 6 for the
    # locked-append-only rule that governs this section.
    #
    # Detail / Noise Reduction extension (2) — idx 135-136
    "ColorNoiseReductionDetail",      # idx 135
    "ColorNoiseReductionSmoothness",  # idx 136
    # Lens Corrections / Manual Defringe (6) — idx 137-142
    "DefringePurpleAmount",  # idx 137
    "DefringePurpleHueLo",   # idx 138
    "DefringePurpleHueHi",   # idx 139
    "DefringeGreenAmount",   # idx 140
    "DefringeGreenHueLo",    # idx 141
    "DefringeGreenHueHi",    # idx 142
    # Lens Corrections / Profile scales (2) — idx 143-144
    # NOTE: LensProfileChromaticAberrationScale excluded — absent from the
    # real LR Classic 15.3 XMP even with lens profile enabled (the two
    # other Scale fields wrote at non-default values in the same XMP).
    # AutoLateralCA toggle (always-on postprocess rule in
    # inference/pipeline.py) covers CA correction in the v2 pipeline.
    "LensProfileDistortionScale",   # idx 143
    "LensProfileVignettingScale",   # idx 144
    # Calibration extension (1) — idx 145
    "ShadowTint",            # idx 145
    # Tone Curve extension (1) — idx 146
    "CurveRefineSaturation", # idx 146  ⚠ MEDIUM — range/default pending LR UI verification
]

assert len(SLIDER_FIELDS) == 147, f"Expected 147 slider fields, got {len(SLIDER_FIELDS)}"

# Lightroom slider valid ranges for output clamping (full LR valid ranges, not "sensible" subsets)
SLIDER_RANGES: dict[str, tuple[float, float]] = {
    # Tone
    "Exposure2012": (-5.0, 5.0),
    "Contrast2012": (-100.0, 100.0),
    "Highlights2012": (-100.0, 100.0),
    "Shadows2012": (-100.0, 100.0),
    "Whites2012": (-100.0, 100.0),
    "Blacks2012": (-100.0, 100.0),
    "Clarity2012": (-100.0, 100.0),
    "Dehaze": (-100.0, 100.0),
    # Presence
    "Texture": (-100.0, 100.0),
    "Vibrance": (-100.0, 100.0),
    "Saturation": (-100.0, 100.0),
    # White balance
    "Temperature": (2000.0, 50000.0),
    "Tint": (-150.0, 150.0),
    # HSL
    **{f"HueAdjustment{c}": (-100.0, 100.0) for c in
       ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]},
    **{f"SaturationAdjustment{c}": (-100.0, 100.0) for c in
       ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]},
    **{f"LuminanceAdjustment{c}": (-100.0, 100.0) for c in
       ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]},
    # Parametric Tone Curve
    "ParametricHighlights": (-100.0, 100.0),
    "ParametricLights": (-100.0, 100.0),
    "ParametricDarks": (-100.0, 100.0),
    "ParametricShadows": (-100.0, 100.0),
    "ParametricHighlightSplit": (0.0, 100.0),
    "ParametricMidtoneSplit": (0.0, 100.0),
    "ParametricShadowSplit": (0.0, 100.0),
    # Color Grading — Shadow/Highlight Hue+Sat use SplitToning XMP names
    # Hue fields wrap 0-360, Sat/Blending 0-100, Lum -100/100, Balance -100/100
    "SplitToningShadowHue": (0.0, 360.0),
    "SplitToningShadowSaturation": (0.0, 100.0),
    "ColorGradeShadowLum": (-100.0, 100.0),
    "ColorGradeMidtoneHue": (0.0, 360.0),
    "ColorGradeMidtoneSat": (0.0, 100.0),
    "ColorGradeMidtoneLum": (-100.0, 100.0),
    "SplitToningHighlightHue": (0.0, 360.0),
    "SplitToningHighlightSaturation": (0.0, 100.0),
    "ColorGradeHighlightLum": (-100.0, 100.0),
    "ColorGradeBlending": (0.0, 100.0),
    "ColorGradeGlobalHue": (0.0, 360.0),
    "ColorGradeGlobalSat": (0.0, 100.0),
    "ColorGradeGlobalLum": (-100.0, 100.0),
    "SplitToningBalance": (-100.0, 100.0),
    # Camera Calibration — LR uses short names without "CameraCalibration" prefix
    "RedHue": (-100.0, 100.0),
    "RedSaturation": (-100.0, 100.0),
    "GreenHue": (-100.0, 100.0),
    "GreenSaturation": (-100.0, 100.0),
    "BlueHue": (-100.0, 100.0),
    "BlueSaturation": (-100.0, 100.0),
    # Detail — Sharpening (non-standard ranges)
    "Sharpness": (0.0, 150.0),
    "SharpenRadius": (0.5, 3.0),
    "SharpenDetail": (0.0, 100.0),
    "SharpenEdgeMasking": (0.0, 100.0),
    # Detail — Noise Reduction
    "LuminanceSmoothing": (0.0, 100.0),
    "LuminanceNoiseReductionDetail": (0.0, 100.0),
    "LuminanceNoiseReductionContrast": (0.0, 100.0),
    "ColorNoiseReduction": (0.0, 100.0),
    # Effects — vignette controls + grain controls
    "PostCropVignetteAmount": (-100.0, 100.0),
    "PostCropVignetteMidpoint": (0.0, 100.0),
    "PostCropVignetteRoundness": (-100.0, 100.0),
    "PostCropVignetteFeather": (0.0, 100.0),
    "PostCropVignetteHighlightContrast": (0.0, 100.0),
    "GrainAmount": (0.0, 100.0),
    "GrainSize": (0.0, 100.0),
    "GrainFrequency": (0.0, 100.0),
    # Lens Corrections
    "LensManualDistortionAmount": (-100.0, 100.0),
    "VignetteAmount": (-100.0, 100.0),
    # Transform (non-standard ranges)
    "PerspectiveVertical": (-100.0, 100.0),
    "PerspectiveHorizontal": (-100.0, 100.0),
    "PerspectiveRotate": (-10.0, 10.0),
    "PerspectiveScale": (50.0, 150.0),
    "PerspectiveAspect": (-100.0, 100.0),
    # === v2 extensions — idx 135-146 ===
    "ColorNoiseReductionDetail":     (0.0, 100.0),
    "ColorNoiseReductionSmoothness": (0.0, 100.0),
    "DefringePurpleAmount":          (0.0, 20.0),
    "DefringePurpleHueLo":           (0.0, 100.0),
    "DefringePurpleHueHi":           (0.0, 100.0),
    "DefringeGreenAmount":           (0.0, 20.0),
    "DefringeGreenHueLo":            (0.0, 100.0),
    "DefringeGreenHueHi":            (0.0, 100.0),
    "LensProfileDistortionScale":    (0.0, 200.0),
    "LensProfileVignettingScale":    (0.0, 200.0),
    "ShadowTint":                    (-100.0, 100.0),
    "CurveRefineSaturation":         (0.0, 100.0),  # ⚠ MEDIUM — pending LR UI verification
    # Tone curves — all axes 0-255 (X = input level, Y = output level)
    **{f: (0.0, 255.0) for f in SLIDER_FIELDS if f.startswith("ToneCurve")},
}

assert set(SLIDER_FIELDS).issubset(SLIDER_RANGES), (
    f"SLIDER_RANGES missing entries for: {set(SLIDER_FIELDS) - set(SLIDER_RANGES)}"
)

# Documented Lightroom Classic 15.3 defaults for v2-extension fields (idx 135-146).
# Used by scripts/migrate_labels_to_v2.py to backfill new label columns in
# pre-v2 parquet train splits. Primary migration path is RE-EXTRACTION from
# source XMPs (Q2 decision 2026-05-13); these defaults are the fallback for
# any row where re-extraction fails or the source XMP is missing the field.
#
# 6/12 defaults verified against real LR Classic 15.3 XMP export (Canon R6 +
# RF24-70mm, PV 15.4, 2026-05-13). 6 marked MEDIUM are based on standard LR
# convention and pending further verification:
#   - ColorNoiseReductionDetail/Smoothness: standard LR default = 50
#   - LensProfile{Distortion,Vignetting}Scale: standard LR default = 100
#     when lens profile is enabled
#   - CurveRefineSaturation: assumed default = 100, range 0-100 (awaiting
#     LR UI verification)
SLIDER_DEFAULTS: dict[str, float] = {
    "ColorNoiseReductionDetail":     50.0,    # MEDIUM (standard LR default)
    "ColorNoiseReductionSmoothness": 50.0,    # MEDIUM (standard LR default)
    "DefringePurpleAmount":          0.0,     # verified
    "DefringePurpleHueLo":           30.0,    # verified in real XMP
    "DefringePurpleHueHi":           70.0,    # verified in real XMP
    "DefringeGreenAmount":           0.0,     # verified
    "DefringeGreenHueLo":            40.0,    # verified in real XMP
    "DefringeGreenHueHi":            60.0,    # verified in real XMP
    "LensProfileDistortionScale":    100.0,   # MEDIUM (standard LR default when profile enabled)
    "LensProfileVignettingScale":    100.0,   # MEDIUM (standard LR default when profile enabled)
    "ShadowTint":                    0.0,     # verified
    "CurveRefineSaturation":         100.0,   # ⚠ MEDIUM — pending LR UI verification
}

assert set(SLIDER_DEFAULTS.keys()) == set(SLIDER_FIELDS[135:]), (
    f"SLIDER_DEFAULTS keys must match v2 extension fields (idx 135-146) exactly. "
    f"Missing: {set(SLIDER_FIELDS[135:]) - set(SLIDER_DEFAULTS.keys())}. "
    f"Extra: {set(SLIDER_DEFAULTS.keys()) - set(SLIDER_FIELDS[135:])}."
)

# Loss weights — mostly 1.0 (Neutral Learner principle).
# Range normalization is done inside WeightedSliderLoss (Task 3.2) by normalizing
# each slider to [0,1] before MSE: normalized = (value - lo) / (hi - lo).
# Temperature uses log-space bounds: lo=log(2000), hi=log(50000).
#
# v1.0.2-candidate bumps (Stage 2): targeted at audit-identified timid fields.
# All other fields stay at 1.0 (Neutral Learner principle).
#
# Curve weights revised after the catalog-parser fix (commit 662cfbd) revealed
# v1.0.1 had been trained on profile-stub tone curves. Re-baselined v1.0.1
# audit against corrected labels shows BIASED (not just TIMID) curve fields
# with 30–70 unit systematic offset. Bumped to 3.0 to overcome that bias.
#
#   Tint        4.0 — worst scalar field (cr=0.03 in Stage 1 audit)
#   Temperature 3.0 — modest bump; bucket loss does the heavy lifting
#   Composite tone curve Pt2_Y..Pt5_Y  3.0 — model emits one fixed S-curve;
#                                            real truth varies per photo
#   R/G/B per-channel curve Pt2_Y..Pt5_Y (12 fields) 3.0 — model predicts
#                                            identity; real truth pushed per
#                                            photo (84.7% of train rows)
#   HSL Hue/Sat/Lum (24 fields) 1.5 — uniformly TIMID in Stage 1
#   Camera Calibration Red/Green/BlueHue 2.0
#
# v1.1.0-c3k-tuned bumps (final pre-12.9K cycle): full-loss audit on
# v1.1.0-c3k showed 7/95 HEALTHY, 24 TIMID, 19 BROKEN. This pass applies
# 1.5× bumps to the 24 TIMID fields, reduces ToneCurveBlue Pt2-5_Y from
# 3.0 → 2.0 (audit flagged Blue overshooting with bias +1.16-1.25σ), and
# raises TINT_BUCKET_LOSS_WEIGHT 1.0 → 2.0 below.
#
# Explicitly NOT bumped (direction-at-chance — deferred to full-12.9K
# retrain, where data volume is expected to do the work):
#   Saturation, SaturationAdjustmentRed/Yellow/Green, ToneCurve_Pt4_X/Y,
#   GrainFrequency. Tint stays at 4.0 (already at audit ceiling); the new
#   sign-wrong term + bumped Tint bucket carry that field.
SLIDER_LOSS_WEIGHTS: dict[str, float] = {field: 1.0 for field in SLIDER_FIELDS}
SLIDER_LOSS_WEIGHTS["Temperature"] = 3.0
SLIDER_LOSS_WEIGHTS["Tint"] = 4.0
for _pt in (2, 3, 4, 5):
    SLIDER_LOSS_WEIGHTS[f"ToneCurve_Pt{_pt}_Y"]      = 3.0
    SLIDER_LOSS_WEIGHTS[f"ToneCurveRed_Pt{_pt}_Y"]   = 3.0
    SLIDER_LOSS_WEIGHTS[f"ToneCurveGreen_Pt{_pt}_Y"] = 3.0
    # ToneCurveBlue Y dropped 3.0 → 2.0 (audit: BROKEN with bias +1.16-1.25σ).
    SLIDER_LOSS_WEIGHTS[f"ToneCurveBlue_Pt{_pt}_Y"]  = 2.0
for _c in ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"):
    SLIDER_LOSS_WEIGHTS[f"HueAdjustment{_c}"]        = 1.5
    SLIDER_LOSS_WEIGHTS[f"SaturationAdjustment{_c}"] = 1.5
    SLIDER_LOSS_WEIGHTS[f"LuminanceAdjustment{_c}"]  = 1.5
SLIDER_LOSS_WEIGHTS["RedHue"]   = 2.0
SLIDER_LOSS_WEIGHTS["GreenHue"] = 2.0
SLIDER_LOSS_WEIGHTS["BlueHue"]  = 2.0

# ── v1.1.0-c3k-tuned: TIMID-field bumps (1.5× each, clamped to 6.0) ──
# Source: /tmp/saha_full_loss_audit.md TIMID list (24 fields).
_TUNED_TIMID_BUMPS: dict[str, float] = {
    "Blacks2012":                    1.5,
    "BlueHue":                       3.0,
    "Clarity2012":                   1.5,
    "ColorGradeHighlightLum":        1.5,
    "ColorGradeMidtoneHue":          1.5,
    "ColorGradeMidtoneSat":          1.5,
    "ColorGradeShadowLum":           1.5,
    "Contrast2012":                  1.5,
    "GreenHue":                      3.0,
    "Highlights2012":                1.5,
    "HueAdjustmentAqua":             2.25,
    "HueAdjustmentBlue":             2.25,
    "HueAdjustmentOrange":           2.25,
    "HueAdjustmentRed":              2.25,
    "HueAdjustmentYellow":           2.25,
    "LuminanceAdjustmentBlue":       2.25,
    "LuminanceAdjustmentGreen":      2.25,
    "LuminanceAdjustmentOrange":     2.25,
    "Shadows2012":                   1.5,
    "SplitToningHighlightHue":       1.5,
    "SplitToningHighlightSaturation": 1.5,
    "ToneCurveGreen_Pt5_Y":          4.5,
    "ToneCurveRed_Pt2_Y":            4.5,
    "Whites2012":                    1.5,
}
for _f, _w in _TUNED_TIMID_BUMPS.items():
    SLIDER_LOSS_WEIGHTS[_f] = _w

# ── Stage 2 loss-term coefficients ────────────────────────────────────────
# Spread penalty: one-sided hinge on per-field std (pred under-predicts spread).
# Per-bucket Temperature: bucket by AsShot Kelvin, penalise mean(pred_corr - truth_corr).
# Per-bucket Tint: bucket by ground-truth Tint magnitude.
#
# TEMPERATURE_BUCKET_LOSS_WEIGHT lowered 1.0 → 0.10 after the v1.0.2-halted-
# epoch1 diagnostic run. The original 1.0 produced a term that was 19.9× MSE
# at end of epoch 1 (vs my pre-train estimate of 0.5× MSE), tripping the
# LossComponentBalanceCallback. The arithmetic miss came from assuming a
# per-bucket log-K gap of ~0.04 — reality at epoch 1 is ~0.22 because the
# Temperature head is still ~1000 K off the truth at that point. 0.10 brings
# the end-of-epoch-1 ratio to ~2× MSE, well under the 5× threshold.
# Per-bucket Tint and spread terms came in fine at 0.10× and 0.05× MSE.
SPREAD_LOSS_WEIGHT: float = 0.5
TEMPERATURE_BUCKET_LOSS_WEIGHT: float = 0.10
# v1.1.0-c3k-tuned bumped 1.0 → 2.0; dialled back to 1.5 pre-12.9K after
# the 3K diagnostic overfit: the 2.0 weight + sign-wrong 0.3 jointly
# rewarded sign-flipped predictions when train signal was insufficient.
# 1.5 still meaningfully bumps the bucket term above its v1.0.1 baseline
# but with less aggressive directional pressure.
TINT_BUCKET_LOSS_WEIGHT: float = 1.5

# Symmetric sign-wrong penalty for the two AsShot-referenced fields
# (Temperature, Tint). Hinge² in range-normalised space; non-zero only when
# pred has opposite sign from truth (relative to AsShot). Coefficient
# expected to contribute ~1–2% of total loss — meant to redirect, not
# dominate. See losses.WeightedSliderLoss._sign_wrong_term.
#
# Halved 0.3 → 0.15 pre-12.9K after the 3K diagnostic overfit at epoch 6
# with Temperature 75% wrong-direction on small data. 0.15 keeps the
# directional pressure but reduces the gradient magnitude that drove the
# heads to confident-but-wrong predictions on 3K rows.
SIGN_WRONG_PENALTY_WEIGHT: float = 0.15


# ── Confidence (used by the API to map per-slider MC-dropout std → scalar) ──
# The Basic-panel sliders are the most photographer-visible. Reducing across
# this fixed set (rather than all 135) keeps the confidence number readable
# and stable: noise on low-impact sliders shouldn't dominate the score.
KEY_CONFIDENCE_SLIDERS: list[str] = [
    "Exposure2012", "Temperature", "Tint",
    "Shadows2012", "Highlights2012", "Whites2012", "Blacks2012", "Contrast2012",
]

# Per-slider normalisation stds. Confidence formula divides each slider's
# MC-dropout std by the matching value here; a slider whose uncertainty
# matches the training-set spread contributes ~1.0 (saturating to "no
# confidence"), while well-bounded uncertainty stays near zero.
# TODO: replace these placeholders with values computed from the v1.0.1
# training set once the audit pipeline exposes them.
CONFIDENCE_NORM_STDS: dict[str, float] = {
    "Exposure2012":   0.5,    # stops
    "Temperature":  500.0,    # Kelvin
    "Tint":          15.0,
    "Shadows2012":   20.0,
    "Highlights2012": 20.0,
    "Whites2012":    20.0,
    "Blacks2012":    20.0,
    "Contrast2012":  15.0,
}

# Path to the original training split. The /api/finetune endpoint reads this
# when building the combined fine-tune parquet.
ORIGINAL_TRAIN_PARQUET = CHECKPOINTS_DIR / "dataset" / "splits_v2_stratified" / "train.parquet"
