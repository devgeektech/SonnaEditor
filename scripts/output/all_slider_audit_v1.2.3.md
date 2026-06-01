# All-Slider Behaviour Audit — v1.2.3 (dp-event-v1.2.3)

**Model:** `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor\v1_learning\model-v2.0.0.ckpt`  
**Test split:** `v1_learning\dataset\splits_v2_stratified\test.parquet`  (30 photos)  
**Generated:** 2026-06-01T15:20:19  
**Architecture:** v1, 13 heads, 135 outputs  

Read-only diagnostic against the live production checkpoint. Raw
model predictions, no postprocess clamping. Temperature (idx 11)
is in log-K in the model's prediction space; analysed in both
log-K and Kelvin views.

## 1. Summary

| Category | Count | % of 135 |
|---|---:|---:|
| HEALTHY | 3 | 2.2% |
| HIGH ERROR | 70 | 51.9% |
| COLLAPSED | 13 | 9.6% |
| WRONG DIRECTION | 0 | 0.0% |
| SPARSE TARGET | 49 | 36.3% |

## 2. Per-panel breakdown (all 135 sliders)

Columns: MAE in native units; std_ratio = std(pred)/std(target);
dir = sign-agreement % on signed-range sliders; corr = Pearson;
sparse = fraction of test rows at default.

### Tone (idx 0-7)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | Exposure2012 | [-5, 5] | HIGH ERROR | 0.240 | 0.26 | 0.96 | 0.59 | 20% |
| 1 | Contrast2012 | [-100, 100] | HIGH ERROR | 0.637 | 0.32 | 1.00 | 0.53 | 0% |
| 2 | Highlights2012 | [-100, 100] | HIGH ERROR | 6.679 | 0.13 | 1.00 | -0.73 | 0% |
| 3 | Shadows2012 | [-100, 100] | HIGH ERROR | 6.989 | 0.17 | 1.00 | -0.77 | 0% |
| 4 | Whites2012 | [-100, 100] | HIGH ERROR | 2.859 | 0.33 | 1.00 | -0.38 | 0% |
| 5 | Blacks2012 | [-100, 100] | HEALTHY | 0.901 | 1.32 | 1.00 | 0.42 | 0% |
| 6 | Clarity2012 | [-100, 100] | HIGH ERROR | 0.307 | — | 1.00 | — | 0% |
| 7 | Dehaze | [-100, 100] | COLLAPSED | 3.105 | 0.08 | 1.00 | 0.18 | 7% |

### Presence (idx 8-10)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 8 | Texture | [-100, 100] | HIGH ERROR | 0.158 | — | — | — | 0% |
| 9 | Vibrance | [-100, 100] | HIGH ERROR | 0.643 | 0.12 | 1.00 | 0.34 | 0% |
| 10 | Saturation | [-100, 100] | HIGH ERROR | 0.651 | 0.17 | 1.00 | -0.37 | 3% |

### WB (idx 11-12)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 11 | Temperature | [2000, 50000] | HEALTHY | 222.518 | 1.00 | — | 0.99 | 53% |
| 12 | Tint | [-150, 150] | HEALTHY | 2.166 | 1.03 | 0.90 | 0.93 | 30% |

### HSL Hue (idx 13-20)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 13 | HueAdjustmentRed | [-100, 100] | HIGH ERROR | 0.496 | — | 1.00 | — | 0% |
| 14 | HueAdjustmentOrange | [-100, 100] | HIGH ERROR | 1.942 | 0.22 | 1.00 | -0.05 | 0% |
| 15 | HueAdjustmentYellow | [-100, 100] | COLLAPSED | 9.218 | 0.02 | 0.92 | 0.03 | 7% |
| 16 | HueAdjustmentGreen | [-100, 100] | COLLAPSED | 5.113 | 0.02 | 0.88 | 0.24 | 7% |
| 17 | HueAdjustmentAqua | [-100, 100] | COLLAPSED | 3.051 | 0.06 | 1.00 | 0.36 | 7% |
| 18 | HueAdjustmentBlue | [-100, 100] | COLLAPSED | 1.609 | 0.02 | 1.00 | -0.08 | 43% |
| 19 | HueAdjustmentPurple | [-100, 100] | HIGH ERROR | 1.927 | 0.48 | 1.00 | -0.56 | 0% |
| 20 | HueAdjustmentMagenta | [-100, 100] | HIGH ERROR | 0.272 | — | 1.00 | — | 0% |

### HSL Saturation (idx 21-28)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 21 | SaturationAdjustmentRed | [-100, 100] | HIGH ERROR | 0.808 | 0.12 | 1.00 | -0.34 | 3% |
| 22 | SaturationAdjustmentOrange | [-100, 100] | COLLAPSED | 1.279 | 0.04 | 1.00 | 0.13 | 23% |
| 23 | SaturationAdjustmentYellow | [-100, 100] | COLLAPSED | 1.740 | 0.02 | 0.92 | -0.22 | 27% |
| 24 | SaturationAdjustmentGreen | [-100, 100] | HIGH ERROR | 1.641 | 0.11 | 1.00 | -0.52 | 3% |
| 25 | SaturationAdjustmentAqua | [-100, 100] | HIGH ERROR | 0.257 | — | 1.00 | — | 0% |
| 26 | SaturationAdjustmentBlue | [-100, 100] | COLLAPSED | 1.790 | 0.05 | 1.00 | 0.41 | 17% |
| 27 | SaturationAdjustmentPurple | [-100, 100] | SPARSE TARGET | 1.123 | 0.03 | 1.00 | 0.48 | 90% |
| 28 | SaturationAdjustmentMagenta | [-100, 100] | COLLAPSED | 1.522 | 0.03 | 0.95 | -0.41 | 10% |

### HSL Luminance (idx 29-36)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 29 | LuminanceAdjustmentRed | [-100, 100] | HIGH ERROR | 0.744 | — | 1.00 | — | 0% |
| 30 | LuminanceAdjustmentOrange | [-100, 100] | COLLAPSED | 1.279 | 0.04 | 1.00 | 0.01 | 17% |
| 31 | LuminanceAdjustmentYellow | [-100, 100] | COLLAPSED | 6.598 | 0.05 | 0.93 | 0.30 | 3% |
| 32 | LuminanceAdjustmentGreen | [-100, 100] | COLLAPSED | 13.584 | 0.03 | 1.00 | -0.02 | 7% |
| 33 | LuminanceAdjustmentAqua | [-100, 100] | SPARSE TARGET | 0.117 | — | — | — | 100% |
| 34 | LuminanceAdjustmentBlue | [-100, 100] | HIGH ERROR | 1.335 | 0.20 | 1.00 | -0.21 | 0% |
| 35 | LuminanceAdjustmentPurple | [-100, 100] | SPARSE TARGET | 0.657 | 0.10 | 1.00 | 0.49 | 87% |
| 36 | LuminanceAdjustmentMagenta | [-100, 100] | COLLAPSED | 2.024 | 0.02 | 0.95 | -0.42 | 10% |

### Parametric Tone Curve (idx 37-43)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 37 | ParametricHighlights | [-100, 100] | SPARSE TARGET | 0.448 | — | — | — | 100% |
| 38 | ParametricLights | [-100, 100] | SPARSE TARGET | 0.306 | — | — | — | 100% |
| 39 | ParametricDarks | [-100, 100] | SPARSE TARGET | 0.350 | — | — | — | 100% |
| 40 | ParametricShadows | [-100, 100] | SPARSE TARGET | 0.090 | — | — | — | 100% |
| 41 | ParametricHighlightSplit | [0, 100] | SPARSE TARGET | 3.592 | — | — | — | 100% |
| 42 | ParametricMidtoneSplit | [0, 100] | HIGH ERROR | 2.767 | — | — | — | 0% |
| 43 | ParametricShadowSplit | [0, 100] | HIGH ERROR | 0.588 | — | — | — | 0% |

### Color Grading (idx 44-57)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 44 | SplitToningShadowHue | [0, 360] | HIGH ERROR | 5.571 | — | — | — | 0% |
| 45 | SplitToningShadowSaturation | [0, 100] | HIGH ERROR | 0.640 | — | — | — | 0% |
| 46 | ColorGradeShadowLum | [-100, 100] | HIGH ERROR | 1.061 | — | 1.00 | — | 0% |
| 47 | ColorGradeMidtoneHue | [0, 360] | HIGH ERROR | 2.804 | — | — | — | 0% |
| 48 | ColorGradeMidtoneSat | [0, 100] | HIGH ERROR | 0.618 | — | — | — | 0% |
| 49 | ColorGradeMidtoneLum | [-100, 100] | HIGH ERROR | 1.490 | — | 1.00 | — | 0% |
| 50 | SplitToningHighlightHue | [0, 360] | HIGH ERROR | 8.726 | — | — | — | 0% |
| 51 | SplitToningHighlightSaturation | [0, 100] | SPARSE TARGET | 0.158 | — | — | — | 100% |
| 52 | ColorGradeHighlightLum | [-100, 100] | HIGH ERROR | 0.253 | — | — | — | 0% |
| 53 | ColorGradeBlending | [0, 100] | SPARSE TARGET | 2.161 | — | — | — | 100% |
| 54 | ColorGradeGlobalHue | [0, 360] | SPARSE TARGET | 0.169 | — | — | — | 100% |
| 55 | ColorGradeGlobalSat | [0, 100] | SPARSE TARGET | 0.043 | — | — | — | 100% |
| 56 | ColorGradeGlobalLum | [-100, 100] | SPARSE TARGET | 0.287 | — | — | — | 100% |
| 57 | SplitToningBalance | [-100, 100] | HIGH ERROR | 3.337 | — | 1.00 | — | 0% |

### Calibration (idx 58-63)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 58 | RedHue | [-100, 100] | HIGH ERROR | 0.526 | — | 1.00 | — | 0% |
| 59 | RedSaturation | [-100, 100] | SPARSE TARGET | 0.352 | — | — | — | 100% |
| 60 | GreenHue | [-100, 100] | HIGH ERROR | 2.249 | — | 1.00 | — | 0% |
| 61 | GreenSaturation | [-100, 100] | HIGH ERROR | 0.346 | — | 1.00 | — | 0% |
| 62 | BlueHue | [-100, 100] | HIGH ERROR | 0.650 | — | 1.00 | — | 0% |
| 63 | BlueSaturation | [-100, 100] | HIGH ERROR | 0.261 | — | 1.00 | — | 0% |

### Detail Sharpening (idx 64-67)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 64 | Sharpness | [0, 150] | HIGH ERROR | 27.815 | 0.35 | — | 0.48 | 0% |
| 65 | SharpenRadius | [0, 3] | SPARSE TARGET | 0.049 | — | — | — | 100% |
| 66 | SharpenDetail | [0, 100] | SPARSE TARGET | 11.124 | — | — | — | 100% |
| 67 | SharpenEdgeMasking | [0, 100] | HIGH ERROR | 52.416 | — | — | — | 0% |

### Detail Noise Reduction (idx 68-71)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 68 | LuminanceSmoothing | [0, 100] | HIGH ERROR | 0.615 | — | — | — | 0% |
| 69 | LuminanceNoiseReductionDetail | [0, 100] | SPARSE TARGET | 2.187 | — | — | — | 100% |
| 70 | LuminanceNoiseReductionContrast | [0, 100] | SPARSE TARGET | 0.131 | — | — | — | 100% |
| 71 | ColorNoiseReduction | [0, 100] | SPARSE TARGET | 1.227 | — | — | — | 100% |

### Effects (idx 72-79)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 72 | PostCropVignetteAmount | [-100, 100] | SPARSE TARGET | 0.193 | — | — | — | 100% |
| 73 | PostCropVignetteMidpoint | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 74 | PostCropVignetteRoundness | [-100, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 75 | PostCropVignetteFeather | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 76 | PostCropVignetteHighlightContrast | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 77 | GrainAmount | [0, 100] | SPARSE TARGET | 0.569 | — | — | — | 100% |
| 78 | GrainSize | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 79 | GrainFrequency | [0, 100] | HIGH ERROR | 0.702 | — | — | — | 0% |

### Lens Corrections (idx 80-81)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 80 | LensManualDistortionAmount | [-100, 100] | SPARSE TARGET | 0.041 | — | — | — | 100% |
| 81 | VignetteAmount | [-100, 100] | SPARSE TARGET | 0.061 | — | — | — | 100% |

### Transform (idx 82-86)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 82 | PerspectiveVertical | [-100, 100] | SPARSE TARGET | 0.038 | — | — | — | 100% |
| 83 | PerspectiveHorizontal | [-100, 100] | SPARSE TARGET | 0.245 | — | — | — | 100% |
| 84 | PerspectiveRotate | [-10, 10] | SPARSE TARGET | 0.318 | — | — | — | 100% |
| 85 | PerspectiveScale | [50, 150] | SPARSE TARGET | 11.986 | — | — | — | 100% |
| 86 | PerspectiveAspect | [-100, 100] | SPARSE TARGET | 0.044 | — | — | — | 100% |

### Tone Curves (Composite) (idx 87-98)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 87 | ToneCurve_Pt1_X | [0, 255] | SPARSE TARGET | 0.504 | — | — | — | 100% |
| 88 | ToneCurve_Pt1_Y | [0, 255] | HIGH ERROR | 0.320 | — | — | — | 0% |
| 89 | ToneCurve_Pt2_X | [0, 255] | HIGH ERROR | 0.753 | — | — | — | 0% |
| 90 | ToneCurve_Pt2_Y | [0, 255] | HIGH ERROR | 0.841 | — | — | — | 0% |
| 91 | ToneCurve_Pt3_X | [0, 255] | HIGH ERROR | 2.517 | — | — | — | 0% |
| 92 | ToneCurve_Pt3_Y | [0, 255] | HIGH ERROR | 2.638 | — | — | — | 0% |
| 93 | ToneCurve_Pt4_X | [0, 255] | HIGH ERROR | 5.957 | — | — | — | 0% |
| 94 | ToneCurve_Pt4_Y | [0, 255] | HIGH ERROR | 6.080 | — | — | — | 0% |
| 95 | ToneCurve_Pt5_X | [0, 255] | SPARSE TARGET | 7.592 | — | — | — | 100% |
| 96 | ToneCurve_Pt5_Y | [0, 255] | HIGH ERROR | 6.646 | — | — | — | 0% |
| 97 | ToneCurve_Pt6_X | [0, 255] | SPARSE TARGET | 8.609 | — | — | — | 100% |
| 98 | ToneCurve_Pt6_Y | [0, 255] | HIGH ERROR | 12.714 | — | — | — | 0% |

### Tone Curves (Red) (idx 99-110)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 99 | ToneCurveRed_Pt1_X | [0, 255] | SPARSE TARGET | 0.624 | — | — | — | 100% |
| 100 | ToneCurveRed_Pt1_Y | [0, 255] | SPARSE TARGET | 0.489 | — | — | — | 100% |
| 101 | ToneCurveRed_Pt2_X | [0, 255] | SPARSE TARGET | 1.700 | — | — | — | 100% |
| 102 | ToneCurveRed_Pt2_Y | [0, 255] | HIGH ERROR | 0.951 | — | — | — | 0% |
| 103 | ToneCurveRed_Pt3_X | [0, 255] | HIGH ERROR | 2.976 | — | — | — | 0% |
| 104 | ToneCurveRed_Pt3_Y | [0, 255] | HIGH ERROR | 2.420 | — | — | — | 0% |
| 105 | ToneCurveRed_Pt4_X | [0, 255] | HIGH ERROR | 4.081 | — | — | — | 0% |
| 106 | ToneCurveRed_Pt4_Y | [0, 255] | HIGH ERROR | 4.874 | — | — | — | 0% |
| 107 | ToneCurveRed_Pt5_X | [0, 255] | HIGH ERROR | 5.732 | — | — | — | 0% |
| 108 | ToneCurveRed_Pt5_Y | [0, 255] | HIGH ERROR | 6.581 | — | — | — | 0% |
| 109 | ToneCurveRed_Pt6_X | [0, 255] | SPARSE TARGET | 41.810 | — | — | — | 100% |
| 110 | ToneCurveRed_Pt6_Y | [0, 255] | SPARSE TARGET | 8.657 | — | — | — | 100% |

### Tone Curves (Green) (idx 111-122)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 111 | ToneCurveGreen_Pt1_X | [0, 255] | SPARSE TARGET | 1.122 | — | — | — | 100% |
| 112 | ToneCurveGreen_Pt1_Y | [0, 255] | SPARSE TARGET | 0.540 | — | — | — | 100% |
| 113 | ToneCurveGreen_Pt2_X | [0, 255] | HIGH ERROR | 2.435 | — | — | — | 0% |
| 114 | ToneCurveGreen_Pt2_Y | [0, 255] | HIGH ERROR | 1.715 | — | — | — | 0% |
| 115 | ToneCurveGreen_Pt3_X | [0, 255] | HIGH ERROR | 2.993 | — | — | — | 0% |
| 116 | ToneCurveGreen_Pt3_Y | [0, 255] | HIGH ERROR | 3.298 | — | — | — | 0% |
| 117 | ToneCurveGreen_Pt4_X | [0, 255] | HIGH ERROR | 5.045 | — | — | — | 0% |
| 118 | ToneCurveGreen_Pt4_Y | [0, 255] | HIGH ERROR | 4.798 | — | — | — | 0% |
| 119 | ToneCurveGreen_Pt5_X | [0, 255] | HIGH ERROR | 6.063 | — | — | — | 0% |
| 120 | ToneCurveGreen_Pt5_Y | [0, 255] | HIGH ERROR | 7.051 | — | — | — | 0% |
| 121 | ToneCurveGreen_Pt6_X | [0, 255] | SPARSE TARGET | 20.683 | — | — | — | 100% |
| 122 | ToneCurveGreen_Pt6_Y | [0, 255] | SPARSE TARGET | 38.621 | — | — | — | 100% |

### Tone Curves (Blue) (idx 123-134)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 123 | ToneCurveBlue_Pt1_X | [0, 255] | SPARSE TARGET | 0.740 | — | — | — | 100% |
| 124 | ToneCurveBlue_Pt1_Y | [0, 255] | SPARSE TARGET | 0.659 | — | — | — | 100% |
| 125 | ToneCurveBlue_Pt2_X | [0, 255] | HIGH ERROR | 1.535 | — | — | — | 0% |
| 126 | ToneCurveBlue_Pt2_Y | [0, 255] | HIGH ERROR | 0.723 | — | — | — | 0% |
| 127 | ToneCurveBlue_Pt3_X | [0, 255] | HIGH ERROR | 2.911 | — | — | — | 0% |
| 128 | ToneCurveBlue_Pt3_Y | [0, 255] | HIGH ERROR | 2.496 | — | — | — | 0% |
| 129 | ToneCurveBlue_Pt4_X | [0, 255] | HIGH ERROR | 4.509 | — | — | — | 0% |
| 130 | ToneCurveBlue_Pt4_Y | [0, 255] | HIGH ERROR | 4.823 | — | — | — | 0% |
| 131 | ToneCurveBlue_Pt5_X | [0, 255] | HIGH ERROR | 5.772 | — | — | — | 0% |
| 132 | ToneCurveBlue_Pt5_Y | [0, 255] | HIGH ERROR | 6.964 | — | — | — | 0% |
| 133 | ToneCurveBlue_Pt6_X | [0, 255] | SPARSE TARGET | 46.390 | — | — | — | 100% |
| 134 | ToneCurveBlue_Pt6_Y | [0, 255] | SPARSE TARGET | 26.595 | — | — | — | 100% |

## 3. Detailed view — non-HEALTHY non-SPARSE sliders

### `LuminanceAdjustmentGreen` — COLLAPSED  (idx 32)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `13.584` (median `10.544`, p95 `38.881`)
- norm_mae: `0.0679` (mae / range_span)
- std(pred)=0.543, std(target)=17.413, ratio=0.031
- mean(pred)=-21.293, mean(target)=-20.800, gap=-0.493
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.022`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `HueAdjustmentYellow` — COLLAPSED  (idx 15)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `9.218` (median `6.250`, p95 `24.907`)
- norm_mae: `0.0461` (mae / range_span)
- std(pred)=0.280, std(target)=11.818, ratio=0.024
- mean(pred)=11.137, mean(target)=10.733, gap=0.404
- direction correct on signed-range subset: `92.3%`
- Pearson corr(pred, target): `0.028`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `LuminanceAdjustmentYellow` — COLLAPSED  (idx 31)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.598` (median `4.942`, p95 `15.686`)
- norm_mae: `0.0330` (mae / range_span)
- std(pred)=0.374, std(target)=8.154, ratio=0.046
- mean(pred)=13.273, mean(target)=14.100, gap=-0.827
- direction correct on signed-range subset: `92.9%`
- Pearson corr(pred, target): `0.298`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `HueAdjustmentGreen` — COLLAPSED  (idx 16)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `5.113` (median `4.553`, p95 `12.061`)
- norm_mae: `0.0256` (mae / range_span)
- std(pred)=0.138, std(target)=5.963, ratio=0.023
- mean(pred)=-3.785, mean(target)=-5.200, gap=1.415
- direction correct on signed-range subset: `87.5%`
- Pearson corr(pred, target): `0.244`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `Dehaze` — COLLAPSED  (idx 7)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.105` (median `2.855`, p95 `6.543`)
- norm_mae: `0.0155` (mae / range_span)
- std(pred)=0.299, std(target)=3.587, ratio=0.083
- mean(pred)=8.824, mean(target)=10.000, gap=-1.176
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.183`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `HueAdjustmentAqua` — COLLAPSED  (idx 17)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.051` (median `2.339`, p95 `7.734`)
- norm_mae: `0.0153` (mae / range_span)
- std(pred)=0.237, std(target)=4.031, ratio=0.059
- mean(pred)=8.326, mean(target)=8.533, gap=-0.207
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.360`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `LuminanceAdjustmentMagenta` — COLLAPSED  (idx 36)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `2.024` (median `2.146`, p95 `2.579`)
- norm_mae: `0.0101` (mae / range_span)
- std(pred)=0.040, std(target)=1.941, ratio=0.020
- mean(pred)=0.844, mean(target)=2.033, gap=-1.189
- direction correct on signed-range subset: `95.0%`
- Pearson corr(pred, target): `-0.423`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SaturationAdjustmentBlue` — COLLAPSED  (idx 26)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.790` (median `1.266`, p95 `3.870`)
- norm_mae: `0.0089` (mae / range_span)
- std(pred)=0.120, std(target)=2.291, ratio=0.053
- mean(pred)=-4.286, mean(target)=-3.533, gap=-0.753
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.414`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SaturationAdjustmentYellow` — COLLAPSED  (idx 23)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.740` (median `1.177`, p95 `4.800`)
- norm_mae: `0.0087` (mae / range_span)
- std(pred)=0.053, std(target)=2.596, ratio=0.021
- mean(pred)=1.745, mean(target)=2.167, gap=-0.422
- direction correct on signed-range subset: `92.3%`
- Pearson corr(pred, target): `-0.220`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `HueAdjustmentBlue` — COLLAPSED  (idx 18)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.609` (median `1.167`, p95 `4.416`)
- norm_mae: `0.0080` (mae / range_span)
- std(pred)=0.042, std(target)=2.040, ratio=0.021
- mean(pred)=-1.172, mean(target)=-1.800, gap=0.628
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.085`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SaturationAdjustmentMagenta` — COLLAPSED  (idx 28)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.522` (median `1.636`, p95 `2.108`)
- norm_mae: `0.0076` (mae / range_span)
- std(pred)=0.041, std(target)=1.499, ratio=0.027
- mean(pred)=-1.351, mean(target)=-2.233, gap=0.882
- direction correct on signed-range subset: `95.0%`
- Pearson corr(pred, target): `-0.410`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SaturationAdjustmentOrange` — COLLAPSED  (idx 22)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.279` (median `1.038`, p95 `2.946`)
- norm_mae: `0.0064` (mae / range_span)
- std(pred)=0.063, std(target)=1.611, ratio=0.039
- mean(pred)=-2.014, mean(target)=-2.267, gap=0.253
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.127`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `LuminanceAdjustmentOrange` — COLLAPSED  (idx 30)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.279` (median `0.765`, p95 `3.752`)
- norm_mae: `0.0064` (mae / range_span)
- std(pred)=0.071, std(target)=1.778, ratio=0.040
- mean(pred)=2.730, mean(target)=2.800, gap=-0.070
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.013`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SharpenEdgeMasking` — HIGH ERROR  (idx 67)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `52.416` (median `52.417`, p95 `54.189`)
- norm_mae: `0.5242` (mae / range_span)
- std(pred)=1.430, std(target)=0.000, ratio=—
- mean(pred)=37.584, mean(target)=90.000, gap=-52.416
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Sharpness` — HIGH ERROR  (idx 64)

- range: `[0.00, 150.00]`, default: `25.00`
- MAE: `27.815` (median `28.344`, p95 `29.952`)
- norm_mae: `0.1854` (mae / range_span)
- std(pred)=1.023, std(target)=2.904, ratio=0.352
- mean(pred)=27.152, mean(target)=54.967, gap=-27.815
- Pearson corr(pred, target): `0.475`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt6_Y` — HIGH ERROR  (idx 98)

- range: `[0.00, 255.00]`, default: `255.00`
- MAE: `12.714` (median `12.539`, p95 `23.202`)
- norm_mae: `0.0499` (mae / range_span)
- std(pred)=8.771, std(target)=0.000, ratio=—
- mean(pred)=239.529, mean(target)=252.000, gap=-12.471
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Shadows2012` — HIGH ERROR  (idx 3)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.989` (median `5.191`, p95 `17.871`)
- norm_mae: `0.0349` (mae / range_span)
- std(pred)=1.113, std(target)=6.513, ratio=0.171
- mean(pred)=32.936, mean(target)=38.667, gap=-5.730
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.768`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Highlights2012` — HIGH ERROR  (idx 2)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.679` (median `4.097`, p95 `17.129`)
- norm_mae: `0.0334` (mae / range_span)
- std(pred)=0.946, std(target)=7.165, ratio=0.132
- mean(pred)=-27.934, mean(target)=-32.000, gap=4.066
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.729`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ParametricMidtoneSplit` — HIGH ERROR  (idx 42)

- range: `[0.00, 100.00]`, default: `50.00`
- MAE: `2.767` (median `2.642`, p95 `5.213`)
- norm_mae: `0.0277` (mae / range_span)
- std(pred)=2.103, std(target)=0.000, ratio=—
- mean(pred)=57.374, mean(target)=60.000, gap=-2.626
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt5_Y` — HIGH ERROR  (idx 120)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `7.051` (median `7.138`, p95 `12.656`)
- norm_mae: `0.0277` (mae / range_span)
- std(pred)=7.102, std(target)=0.000, ratio=—
- mean(pred)=194.997, mean(target)=199.000, gap=-4.003
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt5_Y` — HIGH ERROR  (idx 132)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `6.964` (median `7.066`, p95 `12.537`)
- norm_mae: `0.0273` (mae / range_span)
- std(pred)=6.998, std(target)=0.000, ratio=—
- mean(pred)=191.009, mean(target)=195.000, gap=-3.991
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt5_Y` — HIGH ERROR  (idx 96)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `6.646` (median `6.795`, p95 `11.874`)
- norm_mae: `0.0261` (mae / range_span)
- std(pred)=6.742, std(target)=0.000, ratio=—
- mean(pred)=184.349, mean(target)=188.000, gap=-3.651
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt5_Y` — HIGH ERROR  (idx 108)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `6.581` (median `7.028`, p95 `11.647`)
- norm_mae: `0.0258` (mae / range_span)
- std(pred)=6.776, std(target)=0.000, ratio=—
- mean(pred)=184.622, mean(target)=188.000, gap=-3.378
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningHighlightHue` — HIGH ERROR  (idx 50)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `8.726` (median `8.762`, p95 `10.388`)
- norm_mae: `0.0242` (mae / range_span)
- std(pred)=1.351, std(target)=0.000, ratio=—
- mean(pred)=37.274, mean(target)=46.000, gap=-8.726
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Exposure2012` — HIGH ERROR  (idx 0)

- range: `[-5.00, 5.00]`, default: `0.00`
- MAE: `0.240` (median `0.178`, p95 `0.648`)
- norm_mae: `0.0240` (mae / range_span)
- std(pred)=0.071, std(target)=0.272, ratio=0.262
- mean(pred)=0.118, mean(target)=0.318, gap=-0.200
- direction correct on signed-range subset: `95.8%`
- Pearson corr(pred, target): `0.593`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt4_Y` — HIGH ERROR  (idx 94)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `6.080` (median `5.239`, p95 `11.245`)
- norm_mae: `0.0238` (mae / range_span)
- std(pred)=5.864, std(target)=0.000, ratio=—
- mean(pred)=159.917, mean(target)=164.000, gap=-4.083
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt5_X` — HIGH ERROR  (idx 119)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `6.063` (median `6.457`, p95 `9.933`)
- norm_mae: `0.0238` (mae / range_span)
- std(pred)=6.549, std(target)=0.000, ratio=—
- mean(pred)=179.072, mean(target)=181.000, gap=-1.928
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt4_X` — HIGH ERROR  (idx 93)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `5.957` (median `6.299`, p95 `10.031`)
- norm_mae: `0.0234` (mae / range_span)
- std(pred)=6.344, std(target)=0.000, ratio=—
- mean(pred)=172.718, mean(target)=175.000, gap=-2.282
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt5_X` — HIGH ERROR  (idx 131)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `5.772` (median `6.094`, p95 `9.430`)
- norm_mae: `0.0226` (mae / range_span)
- std(pred)=6.413, std(target)=0.000, ratio=—
- mean(pred)=175.208, mean(target)=176.000, gap=-0.792
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt5_X` — HIGH ERROR  (idx 107)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `5.732` (median `6.136`, p95 `9.471`)
- norm_mae: `0.0225` (mae / range_span)
- std(pred)=6.169, std(target)=0.000, ratio=—
- mean(pred)=168.073, mean(target)=170.000, gap=-1.927
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt4_X` — HIGH ERROR  (idx 117)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `5.045` (median `4.391`, p95 `9.627`)
- norm_mae: `0.0198` (mae / range_span)
- std(pred)=4.314, std(target)=0.000, ratio=—
- mean(pred)=117.653, mean(target)=122.000, gap=-4.347
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt4_Y` — HIGH ERROR  (idx 106)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `4.874` (median `3.806`, p95 `9.169`)
- norm_mae: `0.0191` (mae / range_span)
- std(pred)=4.572, std(target)=0.000, ratio=—
- mean(pred)=124.421, mean(target)=128.000, gap=-3.579
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt4_Y` — HIGH ERROR  (idx 130)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `4.823` (median `3.855`, p95 `9.152`)
- norm_mae: `0.0189` (mae / range_span)
- std(pred)=4.378, std(target)=0.000, ratio=—
- mean(pred)=120.182, mean(target)=124.000, gap=-3.818
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt4_Y` — HIGH ERROR  (idx 118)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `4.798` (median `4.036`, p95 `8.880`)
- norm_mae: `0.0188` (mae / range_span)
- std(pred)=4.595, std(target)=0.000, ratio=—
- mean(pred)=125.714, mean(target)=129.000, gap=-3.286
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt4_X` — HIGH ERROR  (idx 129)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `4.509` (median `3.559`, p95 `8.459`)
- norm_mae: `0.0177` (mae / range_span)
- std(pred)=4.229, std(target)=0.000, ratio=—
- mean(pred)=115.682, mean(target)=119.000, gap=-3.318
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningBalance` — HIGH ERROR  (idx 57)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.337` (median `3.364`, p95 `4.554`)
- norm_mae: `0.0167` (mae / range_span)
- std(pred)=0.982, std(target)=0.000, ratio=—
- mean(pred)=26.663, mean(target)=30.000, gap=-3.337
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt4_X` — HIGH ERROR  (idx 105)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `4.081` (median `4.317`, p95 `6.568`)
- norm_mae: `0.0160` (mae / range_span)
- std(pred)=4.446, std(target)=0.000, ratio=—
- mean(pred)=121.867, mean(target)=123.000, gap=-1.133
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningShadowHue` — HIGH ERROR  (idx 44)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `5.571` (median `5.585`, p95 `6.872`)
- norm_mae: `0.0155` (mae / range_span)
- std(pred)=1.072, std(target)=0.000, ratio=—
- mean(pred)=30.429, mean(target)=36.000, gap=-5.571
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Whites2012` — HIGH ERROR  (idx 4)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `2.859` (median `2.879`, p95 `3.764`)
- norm_mae: `0.0143` (mae / range_span)
- std(pred)=0.609, std(target)=1.856, ratio=0.328
- mean(pred)=-18.181, mean(target)=-20.433, gap=2.253
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.380`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt3_Y` — HIGH ERROR  (idx 116)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `3.298` (median `2.999`, p95 `6.241`)
- norm_mae: `0.0129` (mae / range_span)
- std(pred)=2.670, std(target)=0.000, ratio=—
- mean(pred)=73.000, mean(target)=76.000, gap=-3.000
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt3_X` — HIGH ERROR  (idx 115)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.993` (median `3.181`, p95 `5.306`)
- norm_mae: `0.0117` (mae / range_span)
- std(pred)=3.082, std(target)=0.000, ratio=—
- mean(pred)=84.465, mean(target)=86.000, gap=-1.535
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt3_X` — HIGH ERROR  (idx 103)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.976` (median `3.009`, p95 `5.400`)
- norm_mae: `0.0117` (mae / range_span)
- std(pred)=2.982, std(target)=0.000, ratio=—
- mean(pred)=81.264, mean(target)=83.000, gap=-1.736
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt3_X` — HIGH ERROR  (idx 127)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.911` (median `2.512`, p95 `5.567`)
- norm_mae: `0.0114` (mae / range_span)
- std(pred)=2.508, std(target)=0.000, ratio=—
- mean(pred)=68.506, mean(target)=71.000, gap=-2.494
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `GreenHue` — HIGH ERROR  (idx 60)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `2.249` (median `2.190`, p95 `4.080`)
- norm_mae: `0.0112` (mae / range_span)
- std(pred)=1.607, std(target)=0.000, ratio=—
- mean(pred)=47.809, mean(target)=50.000, gap=-2.191
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt3_Y` — HIGH ERROR  (idx 92)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.638` (median `2.649`, p95 `4.733`)
- norm_mae: `0.0103` (mae / range_span)
- std(pred)=2.640, std(target)=0.000, ratio=—
- mean(pred)=72.465, mean(target)=74.000, gap=-1.535
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt3_X` — HIGH ERROR  (idx 91)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.517` (median `1.962`, p95 `4.716`)
- norm_mae: `0.0099` (mae / range_span)
- std(pred)=2.360, std(target)=0.000, ratio=—
- mean(pred)=64.149, mean(target)=66.000, gap=-1.851
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt3_Y` — HIGH ERROR  (idx 128)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.496` (median `2.248`, p95 `4.728`)
- norm_mae: `0.0098` (mae / range_span)
- std(pred)=2.073, std(target)=0.000, ratio=—
- mean(pred)=56.773, mean(target)=59.000, gap=-2.227
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentOrange` — HIGH ERROR  (idx 14)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.942` (median `1.838`, p95 `4.589`)
- norm_mae: `0.0097` (mae / range_span)
- std(pred)=0.511, std(target)=2.276, ratio=0.224
- mean(pred)=18.552, mean(target)=19.233, gap=-0.681
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.051`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentPurple` — HIGH ERROR  (idx 19)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.927` (median `2.018`, p95 `2.802`)
- norm_mae: `0.0096` (mae / range_span)
- std(pred)=0.491, std(target)=1.024, ratio=0.479
- mean(pred)=-17.936, mean(target)=-19.533, gap=1.597
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.564`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt2_X` — HIGH ERROR  (idx 113)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `2.435` (median `2.280`, p95 `4.601`)
- norm_mae: `0.0095` (mae / range_span)
- std(pred)=1.899, std(target)=0.000, ratio=—
- mean(pred)=51.724, mean(target)=54.000, gap=-2.276
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt3_Y` — HIGH ERROR  (idx 104)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `2.420` (median `1.817`, p95 `5.552`)
- norm_mae: `0.0095` (mae / range_span)
- std(pred)=2.548, std(target)=0.000, ratio=—
- mean(pred)=69.631, mean(target)=68.000, gap=1.631
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentGreen` — HIGH ERROR  (idx 24)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.641` (median `1.426`, p95 `2.486`)
- norm_mae: `0.0082` (mae / range_span)
- std(pred)=0.173, std(target)=1.602, ratio=0.108
- mean(pred)=-6.721, mean(target)=-7.633, gap=0.912
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.524`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeMidtoneHue` — HIGH ERROR  (idx 47)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `2.804` (median `2.838`, p95 `4.095`)
- norm_mae: `0.0078` (mae / range_span)
- std(pred)=1.052, std(target)=0.000, ratio=—
- mean(pred)=29.196, mean(target)=32.000, gap=-2.804
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeMidtoneLum` — HIGH ERROR  (idx 49)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.490` (median `1.534`, p95 `2.392`)
- norm_mae: `0.0075` (mae / range_span)
- std(pred)=0.732, std(target)=0.000, ratio=—
- mean(pred)=20.510, mean(target)=22.000, gap=-1.490
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `GrainFrequency` — HIGH ERROR  (idx 79)

- range: `[0.00, 100.00]`, default: `50.00`
- MAE: `0.702` (median `0.683`, p95 `1.541`)
- norm_mae: `0.0070` (mae / range_span)
- std(pred)=0.697, std(target)=0.000, ratio=—
- mean(pred)=0.479, mean(target)=0.000, gap=0.479
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt2_Y` — HIGH ERROR  (idx 114)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `1.715` (median `1.614`, p95 `3.234`)
- norm_mae: `0.0067` (mae / range_span)
- std(pred)=1.311, std(target)=0.000, ratio=—
- mean(pred)=35.377, mean(target)=37.000, gap=-1.623
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentBlue` — HIGH ERROR  (idx 34)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.335` (median `1.268`, p95 `1.752`)
- norm_mae: `0.0067` (mae / range_span)
- std(pred)=0.301, std(target)=1.499, ratio=0.201
- mean(pred)=10.672, mean(target)=11.433, gap=-0.762
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.215`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningShadowSaturation` — HIGH ERROR  (idx 45)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `0.640` (median `0.635`, p95 `1.092`)
- norm_mae: `0.0064` (mae / range_span)
- std(pred)=0.373, std(target)=0.000, ratio=—
- mean(pred)=10.360, mean(target)=11.000, gap=-0.640
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeMidtoneSat` — HIGH ERROR  (idx 48)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `0.618` (median `0.621`, p95 `1.065`)
- norm_mae: `0.0062` (mae / range_span)
- std(pred)=0.366, std(target)=0.000, ratio=—
- mean(pred)=10.382, mean(target)=11.000, gap=-0.618
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceSmoothing` — HIGH ERROR  (idx 68)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `0.615` (median `0.656`, p95 `1.106`)
- norm_mae: `0.0061` (mae / range_span)
- std(pred)=0.376, std(target)=0.000, ratio=—
- mean(pred)=11.389, mean(target)=12.000, gap=-0.611
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt2_X` — HIGH ERROR  (idx 125)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `1.535` (median `1.414`, p95 `2.906`)
- norm_mae: `0.0060` (mae / range_span)
- std(pred)=1.250, std(target)=0.000, ratio=—
- mean(pred)=34.613, mean(target)=36.000, gap=-1.387
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ParametricShadowSplit` — HIGH ERROR  (idx 43)

- range: `[0.00, 100.00]`, default: `25.00`
- MAE: `0.588` (median `0.516`, p95 `1.131`)
- norm_mae: `0.0059` (mae / range_span)
- std(pred)=0.494, std(target)=0.000, ratio=—
- mean(pred)=13.487, mean(target)=14.000, gap=-0.513
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeShadowLum` — HIGH ERROR  (idx 46)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `1.061` (median `1.090`, p95 `1.795`)
- norm_mae: `0.0053` (mae / range_span)
- std(pred)=0.592, std(target)=0.000, ratio=—
- mean(pred)=-16.939, mean(target)=-18.000, gap=1.061
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentRed` — HIGH ERROR  (idx 21)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.808` (median `0.670`, p95 `1.218`)
- norm_mae: `0.0040` (mae / range_span)
- std(pred)=0.113, std(target)=0.971, ratio=0.117
- mean(pred)=-4.344, mean(target)=-4.700, gap=0.356
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.342`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt2_Y` — HIGH ERROR  (idx 102)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `0.951` (median `1.010`, p95 `1.480`)
- norm_mae: `0.0037` (mae / range_span)
- std(pred)=1.046, std(target)=0.000, ratio=—
- mean(pred)=28.801, mean(target)=29.000, gap=-0.199
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentRed` — HIGH ERROR  (idx 29)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.744` (median `0.747`, p95 `1.123`)
- norm_mae: `0.0037` (mae / range_span)
- std(pred)=0.294, std(target)=0.000, ratio=—
- mean(pred)=11.256, mean(target)=12.000, gap=-0.744
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt2_Y` — HIGH ERROR  (idx 90)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `0.841` (median `0.663`, p95 `1.566`)
- norm_mae: `0.0033` (mae / range_span)
- std(pred)=0.791, std(target)=0.000, ratio=—
- mean(pred)=21.386, mean(target)=22.000, gap=-0.614
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Saturation` — HIGH ERROR  (idx 10)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.651` (median `0.594`, p95 `0.705`)
- norm_mae: `0.0033` (mae / range_span)
- std(pred)=0.125, std(target)=0.748, ratio=0.167
- mean(pred)=4.482, mean(target)=4.800, gap=-0.318
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `-0.371`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `BlueHue` — HIGH ERROR  (idx 62)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.650` (median `0.632`, p95 `1.100`)
- norm_mae: `0.0032` (mae / range_span)
- std(pred)=0.389, std(target)=0.000, ratio=—
- mean(pred)=-11.350, mean(target)=-12.000, gap=0.650
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Vibrance` — HIGH ERROR  (idx 9)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.643` (median `0.431`, p95 `1.047`)
- norm_mae: `0.0032` (mae / range_span)
- std(pred)=0.156, std(target)=1.303, ratio=0.120
- mean(pred)=5.427, mean(target)=5.367, gap=0.061
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.342`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Contrast2012` — HIGH ERROR  (idx 1)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.637` (median `0.522`, p95 `1.423`)
- norm_mae: `0.0032` (mae / range_span)
- std(pred)=0.256, std(target)=0.795, ratio=0.322
- mean(pred)=-7.306, mean(target)=-7.633, gap=0.327
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.529`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt2_X` — HIGH ERROR  (idx 89)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `0.753` (median `0.806`, p95 `1.287`)
- norm_mae: `0.0030` (mae / range_span)
- std(pred)=0.799, std(target)=0.000, ratio=—
- mean(pred)=21.694, mean(target)=22.000, gap=-0.306
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt2_Y` — HIGH ERROR  (idx 126)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `0.723` (median `0.737`, p95 `1.328`)
- norm_mae: `0.0028` (mae / range_span)
- std(pred)=0.731, std(target)=0.000, ratio=—
- mean(pred)=19.584, mean(target)=20.000, gap=-0.416
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `RedHue` — HIGH ERROR  (idx 58)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.526` (median `0.456`, p95 `1.029`)
- norm_mae: `0.0026` (mae / range_span)
- std(pred)=0.453, std(target)=0.000, ratio=—
- mean(pred)=13.543, mean(target)=14.000, gap=-0.457
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentRed` — HIGH ERROR  (idx 13)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.496` (median `0.505`, p95 `0.735`)
- norm_mae: `0.0025` (mae / range_span)
- std(pred)=0.186, std(target)=0.000, ratio=—
- mean(pred)=6.504, mean(target)=7.000, gap=-0.496
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `GreenSaturation` — HIGH ERROR  (idx 61)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.346` (median `0.356`, p95 `0.640`)
- norm_mae: `0.0017` (mae / range_span)
- std(pred)=0.228, std(target)=0.000, ratio=—
- mean(pred)=-7.655, mean(target)=-8.000, gap=0.345
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Clarity2012` — HIGH ERROR  (idx 6)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.307` (median `0.320`, p95 `0.522`)
- norm_mae: `0.0015` (mae / range_span)
- std(pred)=0.157, std(target)=0.000, ratio=—
- mean(pred)=-4.693, mean(target)=-5.000, gap=0.307
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentMagenta` — HIGH ERROR  (idx 20)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.272` (median `0.271`, p95 `0.446`)
- norm_mae: `0.0014` (mae / range_span)
- std(pred)=0.135, std(target)=0.000, ratio=—
- mean(pred)=-4.728, mean(target)=-5.000, gap=0.272
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `BlueSaturation` — HIGH ERROR  (idx 63)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.261` (median `0.227`, p95 `0.507`)
- norm_mae: `0.0013` (mae / range_span)
- std(pred)=0.228, std(target)=0.000, ratio=—
- mean(pred)=-6.790, mean(target)=-7.000, gap=0.210
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentAqua` — HIGH ERROR  (idx 25)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.257` (median `0.258`, p95 `0.365`)
- norm_mae: `0.0013` (mae / range_span)
- std(pred)=0.076, std(target)=0.000, ratio=—
- mean(pred)=-2.743, mean(target)=-3.000, gap=0.257
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeHighlightLum` — HIGH ERROR  (idx 52)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.253` (median `0.265`, p95 `0.387`)
- norm_mae: `0.0013` (mae / range_span)
- std(pred)=0.087, std(target)=0.000, ratio=—
- mean(pred)=-2.253, mean(target)=-2.000, gap=-0.253
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt1_Y` — HIGH ERROR  (idx 88)

- range: `[0.00, 255.00]`, default: `0.00`
- MAE: `0.320` (median `0.286`, p95 `0.676`)
- norm_mae: `0.0013` (mae / range_span)
- std(pred)=0.350, std(target)=0.000, ratio=—
- mean(pred)=9.141, mean(target)=9.000, gap=0.141
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Texture` — HIGH ERROR  (idx 8)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.158` (median `0.155`, p95 `0.237`)
- norm_mae: `0.0008` (mae / range_span)
- std(pred)=0.058, std(target)=0.000, ratio=—
- mean(pred)=1.842, mean(target)=2.000, gap=-0.158
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

## 4. Tone curve identity-collapse check

Documented issue (HANDOVER Part 6 item 9): the model converged to near-identity curve predictions across all 4 channels in v1.0.1. Quantifying here against the current shipping v1.2.3.

| Channel | mean L2 (pred ↔ identity) | mean L2 (target ↔ identity) | identity bias |
|---|---:|---:|---|
| Composite | 21.08 | 21.00 | OK |
| Red | 28.94 | 31.86 | OK |
| Green | 26.78 | 27.60 | OK |
| Blue | 25.17 | 28.04 | OK |

_Distance = mean over photos of L2-norm between (Pt2_Y..Pt5_Y) vs identity (Pt_n_Y == Pt_n_X)._ A near-zero pred-vs-identity distance with non-zero target-vs-identity distance confirms collapse.

## 5. Temperature dual-view (log-K + Kelvin)

- log-K MAE (prediction space): `0.0485` (typical training-loss scale)
- Kelvin MAE (user-facing): `223 K` (target was <250 K per HANDOVER)
- Kelvin p95 abs error: `426 K`

## 6. Worst-offender photos (top 20 by weighted error)

Weighting: sum over fields of `|pred - target| / range_span` (Temperature uses Kelvin/range_K).

**1. `0ca63f0d51dcb41a...`** — shoot `19181_Canon_EOS_R6`, total weighted error `4.358`
   - ISO 100, Canon EOS R6, focal 28.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-54.831), ToneCurveBlue_Pt6_X (Δ=-59.545), ToneCurveRed_Pt6_X (Δ=-55.280), ToneCurveGreen_Pt6_Y (Δ=-52.313), PerspectiveScale (Δ=-17.743)

**2. `43611e04110bd6c7...`** — shoot `19181_Canon_EOS_R6`, total weighted error `4.296`
   - ISO 250, Canon EOS R6, focal 44.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.948), ToneCurveBlue_Pt6_X (Δ=-54.636), LuminanceAdjustmentGreen (Δ=+40.170), Sharpness (Δ=-29.924), ToneCurveRed_Pt6_X (Δ=-50.306)

**3. `ad0b917b28b1c39c...`** — shoot `19181_Canon_EOS_R6`, total weighted error `4.176`
   - ISO 125, Canon EOS R6, focal 70.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-54.096), ToneCurveBlue_Pt6_X (Δ=-55.623), ToneCurveRed_Pt6_X (Δ=-51.282), Sharpness (Δ=-29.059), ToneCurveGreen_Pt6_Y (Δ=-48.295)

**4. `b8ae1c9c4b6ac222...`** — shoot `19181_Canon_EOS_R6`, total weighted error `4.139`
   - ISO 200, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.639), LuminanceAdjustmentGreen (Δ=+44.202), ToneCurveBlue_Pt6_X (Δ=-52.624), Sharpness (Δ=-29.767), ToneCurveRed_Pt6_X (Δ=-48.237)

**5. `18d81a2635f718da...`** — shoot `19181_Canon_EOS_R6`, total weighted error `4.025`
   - ISO 200, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.852), ToneCurveBlue_Pt6_X (Δ=-53.146), ToneCurveRed_Pt6_X (Δ=-48.760), LuminanceAdjustmentGreen (Δ=+37.307), Sharpness (Δ=-27.881)

**6. `b8fbe210598fe03c...`** — shoot `19181_Canon_EOS-1D_X_Mark_II`, total weighted error `3.986`
   - ISO 160, Canon EOS-1D X Mark II, focal 134.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-54.242), ToneCurveBlue_Pt6_X (Δ=-55.786), ToneCurveRed_Pt6_X (Δ=-51.430), ToneCurveGreen_Pt6_Y (Δ=-48.484), PerspectiveScale (Δ=-16.116)

**7. `b5adbe174c245a18...`** — shoot `18523_Canon_EOS_R6`, total weighted error `3.750`
   - ISO 320, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-54.123), ToneCurveBlue_Pt6_X (Δ=-54.966), Sharpness (Δ=-30.091), ToneCurveRed_Pt6_X (Δ=-50.638), ToneCurveGreen_Pt6_Y (Δ=-47.580)

**8. `c721e915fb4e9050...`** — shoot `19181_Canon_EOS_R6`, total weighted error `3.721`
   - ISO 200, Canon EOS R6, focal 28.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.980), ToneCurveBlue_Pt6_X (Δ=-54.711), ToneCurveRed_Pt6_X (Δ=-50.337), ToneCurveGreen_Pt6_Y (Δ=-47.307), Sharpness (Δ=-23.972)

**9. `65823e939ffd2273...`** — shoot `18523_Canon_EOS_R6`, total weighted error `3.588`
   - ISO 320, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-54.000), ToneCurveBlue_Pt6_X (Δ=-54.328), Sharpness (Δ=-29.975), ToneCurveRed_Pt6_X (Δ=-49.983), ToneCurveGreen_Pt6_Y (Δ=-46.915)

**10. `927a6cce66fdc7a2...`** — shoot `19241_Canon_EOS_R6`, total weighted error `3.380`
   - ISO 800, Canon EOS R6, focal 40.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.431), ToneCurveBlue_Pt6_X (Δ=-52.041), Sharpness (Δ=-29.514), ToneCurveRed_Pt6_X (Δ=-47.595), ToneCurveGreen_Pt6_Y (Δ=-44.436)

**11. `bca8f6f01d65ac5f...`** — shoot `19181_Canon_EOS_R6`, total weighted error `3.322`
   - ISO 250, Canon EOS R6, focal 55.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.044), ToneCurveBlue_Pt6_X (Δ=-49.853), Sharpness (Δ=-29.291), ToneCurveRed_Pt6_X (Δ=-45.404), ToneCurveGreen_Pt6_Y (Δ=-42.272)

**12. `33e6fa8e91bfebbe...`** — shoot `19241_Canon_EOS_R6`, total weighted error `3.312`
   - ISO 800, Canon EOS R6, focal 36.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.385), ToneCurveBlue_Pt6_X (Δ=-51.762), Sharpness (Δ=-29.519), ToneCurveRed_Pt6_X (Δ=-47.316), ToneCurveGreen_Pt6_Y (Δ=-44.164)

**13. `5383db8a2128882d...`** — shoot `18589_Canon_EOS_R6`, total weighted error `3.256`
   - ISO 320, Canon EOS R6, focal 60.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.210), ToneCurveBlue_Pt6_X (Δ=-51.587), Sharpness (Δ=-29.460), ToneCurveRed_Pt6_X (Δ=-47.117), ToneCurveGreen_Pt6_Y (Δ=-44.010)

**14. `84a08c4cb1cf5efc...`** — shoot `19241_Canon_EOS_R6`, total weighted error `3.232`
   - ISO 800, Canon EOS R6, focal 35.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-53.173), ToneCurveBlue_Pt6_X (Δ=-50.697), Sharpness (Δ=-29.360), ToneCurveRed_Pt6_X (Δ=-46.225), ToneCurveGreen_Pt6_Y (Δ=-43.062)

**15. `e85d8d0e9b29f7bf...`** — shoot `18755_Canon_EOS_R6`, total weighted error `2.615`
   - ISO 1600, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-52.448), Sharpness (Δ=-28.871), ToneCurveBlue_Pt6_X (Δ=-46.745), ToneCurveRed_Pt6_X (Δ=-42.165), ToneCurveGreen_Pt6_Y (Δ=-38.978)

**16. `814a4a5e80f2dc6b...`** — shoot `18589_Canon_EOS_R6`, total weighted error `2.574`
   - ISO 320, Canon EOS R6, focal 70.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-52.386), Sharpness (Δ=-28.870), ToneCurveBlue_Pt6_X (Δ=-46.154), ToneCurveRed_Pt6_X (Δ=-41.553), ToneCurveGreen_Pt6_Y (Δ=-38.390)

**17. `cb57e0f27be9d428...`** — shoot `18755_Canon_EOS_R6`, total weighted error `2.554`
   - ISO 4000, Canon EOS R6, focal 35.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-50.614), Sharpness (Δ=-27.521), ToneCurveBlue_Pt6_X (Δ=-36.365), ToneCurveRed_Pt6_X (Δ=-31.580), ToneCurveGreen_Pt6_Y (Δ=-28.181)

**18. `4d6c072722428dc9...`** — shoot `18755_Canon_EOS_R6`, total weighted error `2.497`
   - ISO 6400, Canon EOS R6, focal 46.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-50.602), Sharpness (Δ=-27.546), ToneCurveBlue_Pt6_X (Δ=-36.821), ToneCurveRed_Pt6_X (Δ=-31.969), LuminanceAdjustmentGreen (Δ=-22.653)

**19. `6d65843916dbd1b8...`** — shoot `18755_Canon_EOS_R6`, total weighted error `2.478`
   - ISO 2500, Canon EOS R6, focal 32.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-51.991), Sharpness (Δ=-28.560), ToneCurveBlue_Pt6_X (Δ=-44.459), ToneCurveRed_Pt6_X (Δ=-39.797), ToneCurveGreen_Pt6_Y (Δ=-36.569)

**20. `326be5a183526e91...`** — shoot `18755_Canon_EOS_R6`, total weighted error `2.469`
   - ISO 3200, Canon EOS R6, focal 64.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=-51.092), Sharpness (Δ=-27.865), ToneCurveBlue_Pt6_X (Δ=-39.802), ToneCurveRed_Pt6_X (Δ=-35.003), ToneCurveGreen_Pt6_Y (Δ=-31.688)

## 7. Surprises / contradictions

- Temperature is classified HEALTHY despite the documented 730K-MAE failure mode in HANDOVER. Current Kelvin MAE: 223 K.
- Tone curve Composite is NOT showing the documented identity-collapse (pred-vs-identity 21.08 vs target-vs-identity 21.00). HANDOVER says all 4 channels collapse — verify.
- Tone curve Red is NOT showing the documented identity-collapse (pred-vs-identity 28.94 vs target-vs-identity 31.86). HANDOVER says all 4 channels collapse — verify.
- Tone curve Green is NOT showing the documented identity-collapse (pred-vs-identity 26.78 vs target-vs-identity 27.60). HANDOVER says all 4 channels collapse — verify.
- Tone curve Blue is NOT showing the documented identity-collapse (pred-vs-identity 25.17 vs target-vs-identity 28.04). HANDOVER says all 4 channels collapse — verify.
- COLLAPSED sliders found OUTSIDE the tone-curve panels: ['Dehaze', 'HueAdjustmentYellow', 'HueAdjustmentGreen', 'HueAdjustmentAqua', 'HueAdjustmentBlue', 'SaturationAdjustmentOrange', 'SaturationAdjustmentYellow', 'SaturationAdjustmentBlue', 'SaturationAdjustmentMagenta', 'LuminanceAdjustmentOrange', 'LuminanceAdjustmentYellow', 'LuminanceAdjustmentGreen', 'LuminanceAdjustmentMagenta']. HANDOVER documents tone-curve collapse only — this is new.
