# All-Slider Behaviour Audit — v1.2.3 (dp-event-v1.2.3)

**Model:** `v1_learning/model-v1.2.3-prod256.ckpt`  
**Test split:** `v1_learning/dataset/splits_v2_stratified/test.parquet`  (1694 photos)  
**Generated:** 2026-05-13T16:22:11  
**Architecture:** v1, 13 heads, 135 outputs  

Read-only diagnostic against the live production checkpoint. Raw
model predictions, no postprocess clamping. Temperature (idx 11)
is in log-K in the model's prediction space; analysed in both
log-K and Kelvin views.

## 1. Summary

| Category | Count | % of 135 |
|---|---:|---:|
| HEALTHY | 11 | 8.1% |
| HIGH ERROR | 50 | 37.0% |
| COLLAPSED | 11 | 8.1% |
| WRONG DIRECTION | 11 | 8.1% |
| SPARSE TARGET | 52 | 38.5% |

## 2. Per-panel breakdown (all 135 sliders)

Columns: MAE in native units; std_ratio = std(pred)/std(target);
dir = sign-agreement % on signed-range sliders; corr = Pearson;
sparse = fraction of test rows at default.

### Tone (idx 0-7)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | Exposure2012 | [-5, 5] | HIGH ERROR | 0.342 | 0.54 | 0.78 | 0.45 | 14% |
| 1 | Contrast2012 | [-100, 100] | WRONG DIRECTION | 7.396 | 0.39 | 0.53 | 0.17 | 5% |
| 2 | Highlights2012 | [-100, 100] | HIGH ERROR | 7.323 | 0.36 | 1.00 | 0.17 | 4% |
| 3 | Shadows2012 | [-100, 100] | HIGH ERROR | 15.162 | 0.35 | 1.00 | 0.36 | 7% |
| 4 | Whites2012 | [-100, 100] | WRONG DIRECTION | 15.383 | 0.52 | 0.38 | 0.25 | 38% |
| 5 | Blacks2012 | [-100, 100] | HIGH ERROR | 10.265 | 0.50 | 0.78 | 0.35 | 8% |
| 6 | Clarity2012 | [-100, 100] | HIGH ERROR | 3.933 | 0.23 | 1.00 | 0.06 | 3% |
| 7 | Dehaze | [-100, 100] | HIGH ERROR | 4.507 | 0.39 | 0.99 | 0.41 | 41% |

### Presence (idx 8-10)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 8 | Texture | [-100, 100] | WRONG DIRECTION | 3.443 | 0.36 | 0.20 | 0.17 | 41% |
| 9 | Vibrance | [-100, 100] | HIGH ERROR | 6.907 | 0.48 | 0.99 | 0.17 | 8% |
| 10 | Saturation | [-100, 100] | WRONG DIRECTION | 5.883 | 0.43 | 0.30 | 0.27 | 12% |

### WB (idx 11-12)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 11 | Temperature | [2000, 50000] | HEALTHY | 731.276 | 0.93 | — | 0.71 | 22% |
| 12 | Tint | [-150, 150] | COLLAPSED | 6.146 | 0.04 | 0.97 | -0.00 | 13% |

### HSL Hue (idx 13-20)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 13 | HueAdjustmentRed | [-100, 100] | HIGH ERROR | 4.287 | 0.41 | 1.00 | 0.28 | 33% |
| 14 | HueAdjustmentOrange | [-100, 100] | HIGH ERROR | 6.396 | 0.46 | 1.00 | 0.12 | 8% |
| 15 | HueAdjustmentYellow | [-100, 100] | HIGH ERROR | 7.184 | 0.75 | 0.78 | 0.52 | 21% |
| 16 | HueAdjustmentGreen | [-100, 100] | WRONG DIRECTION | 8.144 | 0.69 | 0.49 | 0.54 | 44% |
| 17 | HueAdjustmentAqua | [-100, 100] | HIGH ERROR | 6.379 | 0.45 | 1.00 | 0.10 | 29% |
| 18 | HueAdjustmentBlue | [-100, 100] | HIGH ERROR | 7.154 | 0.35 | 0.66 | 0.42 | 33% |
| 19 | HueAdjustmentPurple | [-100, 100] | HEALTHY | 7.660 | 0.72 | 0.99 | -0.16 | 29% |
| 20 | HueAdjustmentMagenta | [-100, 100] | HEALTHY | 0.689 | 0.94 | 1.00 | -0.22 | 29% |

### HSL Saturation (idx 21-28)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 21 | SaturationAdjustmentRed | [-100, 100] | WRONG DIRECTION | 7.502 | 0.73 | 0.55 | 0.04 | 3% |
| 22 | SaturationAdjustmentOrange | [-100, 100] | HIGH ERROR | 17.178 | 0.31 | 0.58 | -0.02 | 10% |
| 23 | SaturationAdjustmentYellow | [-100, 100] | HIGH ERROR | 9.815 | 0.47 | 0.76 | 0.17 | 8% |
| 24 | SaturationAdjustmentGreen | [-100, 100] | WRONG DIRECTION | 9.050 | 0.60 | 0.54 | 0.08 | 3% |
| 25 | SaturationAdjustmentAqua | [-100, 100] | HEALTHY | 2.296 | 0.58 | 1.00 | 0.30 | 3% |
| 26 | SaturationAdjustmentBlue | [-100, 100] | HEALTHY | 8.481 | 0.53 | 0.99 | 0.37 | 29% |
| 27 | SaturationAdjustmentPurple | [-100, 100] | HIGH ERROR | 11.049 | 0.50 | 0.59 | 0.18 | 11% |
| 28 | SaturationAdjustmentMagenta | [-100, 100] | WRONG DIRECTION | 11.124 | 0.44 | 0.48 | 0.26 | 3% |

### HSL Luminance (idx 29-36)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 29 | LuminanceAdjustmentRed | [-100, 100] | HIGH ERROR | 3.325 | 0.47 | 1.00 | 0.59 | 3% |
| 30 | LuminanceAdjustmentOrange | [-100, 100] | HIGH ERROR | 10.295 | 0.41 | 0.89 | 0.43 | 25% |
| 31 | LuminanceAdjustmentYellow | [-100, 100] | HIGH ERROR | 10.353 | 0.48 | 0.91 | 0.34 | 4% |
| 32 | LuminanceAdjustmentGreen | [-100, 100] | HEALTHY | 6.231 | 0.89 | 0.98 | 0.27 | 11% |
| 33 | LuminanceAdjustmentAqua | [-100, 100] | SPARSE TARGET | 1.933 | 0.67 | 0.00 | -0.31 | 100% |
| 34 | LuminanceAdjustmentBlue | [-100, 100] | WRONG DIRECTION | 7.371 | 0.56 | 0.47 | 0.08 | 3% |
| 35 | LuminanceAdjustmentPurple | [-100, 100] | WRONG DIRECTION | 6.513 | 0.28 | 0.48 | -0.05 | 35% |
| 36 | LuminanceAdjustmentMagenta | [-100, 100] | HIGH ERROR | 9.748 | 0.30 | 0.55 | 0.04 | 4% |

### Parametric Tone Curve (idx 37-43)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 37 | ParametricHighlights | [-100, 100] | SPARSE TARGET | 0.204 | — | — | — | 100% |
| 38 | ParametricLights | [-100, 100] | SPARSE TARGET | 0.148 | — | — | — | 100% |
| 39 | ParametricDarks | [-100, 100] | SPARSE TARGET | 0.158 | — | — | — | 100% |
| 40 | ParametricShadows | [-100, 100] | SPARSE TARGET | 0.021 | — | — | — | 100% |
| 41 | ParametricHighlightSplit | [0, 100] | SPARSE TARGET | 1.404 | — | — | — | 100% |
| 42 | ParametricMidtoneSplit | [0, 100] | HIGH ERROR | 1.759 | — | — | — | 41% |
| 43 | ParametricShadowSplit | [0, 100] | HIGH ERROR | 0.657 | — | — | — | 41% |

### Color Grading (idx 44-57)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 44 | SplitToningShadowHue | [0, 360] | COLLAPSED | 48.087 | 0.03 | — | -0.03 | 3% |
| 45 | SplitToningShadowSaturation | [0, 100] | HIGH ERROR | 4.179 | 0.24 | — | -0.14 | 54% |
| 46 | ColorGradeShadowLum | [-100, 100] | HIGH ERROR | 15.444 | 0.48 | 0.70 | 0.19 | 30% |
| 47 | ColorGradeMidtoneHue | [0, 360] | COLLAPSED | 10.763 | 0.05 | — | -0.13 | 3% |
| 48 | ColorGradeMidtoneSat | [0, 100] | HIGH ERROR | 5.270 | 0.33 | — | 0.30 | 28% |
| 49 | ColorGradeMidtoneLum | [-100, 100] | HIGH ERROR | 7.352 | 0.23 | 1.00 | 0.16 | 30% |
| 50 | SplitToningHighlightHue | [0, 360] | HIGH ERROR | 5.781 | 0.33 | — | 0.15 | 3% |
| 51 | SplitToningHighlightSaturation | [0, 100] | HEALTHY | 2.857 | 0.89 | — | 0.30 | 67% |
| 52 | ColorGradeHighlightLum | [-100, 100] | WRONG DIRECTION | 0.856 | 0.39 | 0.00 | 0.05 | 30% |
| 53 | ColorGradeBlending | [0, 100] | SPARSE TARGET | 0.621 | 0.32 | — | 0.02 | 100% |
| 54 | ColorGradeGlobalHue | [0, 360] | SPARSE TARGET | 0.017 | — | — | — | 100% |
| 55 | ColorGradeGlobalSat | [0, 100] | SPARSE TARGET | 0.040 | — | — | — | 100% |
| 56 | ColorGradeGlobalLum | [-100, 100] | SPARSE TARGET | 0.022 | — | — | — | 100% |
| 57 | SplitToningBalance | [-100, 100] | HIGH ERROR | 0.403 | — | 1.00 | — | 34% |

### Calibration (idx 58-63)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 58 | RedHue | [-100, 100] | HIGH ERROR | 7.923 | 0.16 | 1.00 | 0.35 | 19% |
| 59 | RedSaturation | [-100, 100] | HIGH ERROR | 7.592 | 0.24 | 0.98 | 0.11 | 62% |
| 60 | GreenHue | [-100, 100] | HIGH ERROR | 20.710 | 0.54 | 1.00 | 0.45 | 20% |
| 61 | GreenSaturation | [-100, 100] | HIGH ERROR | 8.092 | 0.30 | 0.71 | 0.38 | 21% |
| 62 | BlueHue | [-100, 100] | HIGH ERROR | 4.593 | 0.28 | 1.00 | 0.42 | 46% |
| 63 | BlueSaturation | [-100, 100] | HIGH ERROR | 5.092 | 0.22 | 0.67 | 0.49 | 20% |

### Detail Sharpening (idx 64-67)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 64 | Sharpness | [0, 150] | HIGH ERROR | 14.843 | 0.14 | — | 0.31 | 1% |
| 65 | SharpenRadius | [0, 3] | SPARSE TARGET | 0.025 | — | — | — | 100% |
| 66 | SharpenDetail | [0, 100] | SPARSE TARGET | 0.678 | — | — | — | 100% |
| 67 | SharpenEdgeMasking | [0, 100] | HIGH ERROR | 10.570 | 0.42 | — | 0.51 | 8% |

### Detail Noise Reduction (idx 68-71)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 68 | LuminanceSmoothing | [0, 100] | HIGH ERROR | 5.295 | 0.38 | — | 0.08 | 39% |
| 69 | LuminanceNoiseReductionDetail | [0, 100] | SPARSE TARGET | 1.974 | — | — | — | 100% |
| 70 | LuminanceNoiseReductionContrast | [0, 100] | SPARSE TARGET | 0.017 | — | — | — | 100% |
| 71 | ColorNoiseReduction | [0, 100] | SPARSE TARGET | 0.260 | 0.32 | — | 0.01 | 100% |

### Effects (idx 72-79)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 72 | PostCropVignetteAmount | [-100, 100] | SPARSE TARGET | 0.173 | 0.20 | 0.00 | -0.00 | 100% |
| 73 | PostCropVignetteMidpoint | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 74 | PostCropVignetteRoundness | [-100, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 75 | PostCropVignetteFeather | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 76 | PostCropVignetteHighlightContrast | [0, 100] | SPARSE TARGET | — | — | — | — | 100% |
| 77 | GrainAmount | [0, 100] | SPARSE TARGET | 5.216 | 0.45 | — | 0.76 | 93% |
| 78 | GrainSize | [0, 100] | SPARSE TARGET | 1.710 | 0.67 | — | 0.54 | 93% |
| 79 | GrainFrequency | [0, 100] | SPARSE TARGET | 9.929 | 0.48 | — | 0.60 | 85% |

### Lens Corrections (idx 80-81)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 80 | LensManualDistortionAmount | [-100, 100] | SPARSE TARGET | 0.010 | — | — | — | 100% |
| 81 | VignetteAmount | [-100, 100] | SPARSE TARGET | 0.013 | — | — | — | 100% |

### Transform (idx 82-86)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 82 | PerspectiveVertical | [-100, 100] | SPARSE TARGET | 0.007 | — | — | — | 100% |
| 83 | PerspectiveHorizontal | [-100, 100] | SPARSE TARGET | 0.014 | — | — | — | 100% |
| 84 | PerspectiveRotate | [-10, 10] | SPARSE TARGET | 0.033 | — | — | — | 100% |
| 85 | PerspectiveScale | [50, 150] | SPARSE TARGET | 0.934 | — | — | — | 100% |
| 86 | PerspectiveAspect | [-100, 100] | SPARSE TARGET | 0.046 | — | — | — | 100% |

### Tone Curves (Composite) (idx 87-98)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 87 | ToneCurve_Pt1_X | [0, 255] | SPARSE TARGET | 0.027 | — | — | — | 100% |
| 88 | ToneCurve_Pt1_Y | [0, 255] | COLLAPSED | 3.298 | 0.05 | — | -0.01 | 9% |
| 89 | ToneCurve_Pt2_X | [0, 255] | COLLAPSED | 5.014 | 0.07 | — | 0.23 | 4% |
| 90 | ToneCurve_Pt2_Y | [0, 255] | COLLAPSED | 6.387 | 0.06 | — | 0.14 | 3% |
| 91 | ToneCurve_Pt3_X | [0, 255] | COLLAPSED | 8.248 | 0.08 | — | 0.12 | 4% |
| 92 | ToneCurve_Pt3_Y | [0, 255] | COLLAPSED | 12.174 | 0.06 | — | 0.07 | 3% |
| 93 | ToneCurve_Pt4_X | [0, 255] | HIGH ERROR | 16.189 | 0.14 | — | -0.03 | 39% |
| 94 | ToneCurve_Pt4_Y | [0, 255] | HIGH ERROR | 9.430 | 0.22 | — | 0.03 | 38% |
| 95 | ToneCurve_Pt5_X | [0, 255] | HEALTHY | 4.677 | 0.73 | — | 0.24 | 16% |
| 96 | ToneCurve_Pt5_Y | [0, 255] | HIGH ERROR | 5.869 | 0.38 | — | 0.24 | 6% |
| 97 | ToneCurve_Pt6_X | [0, 255] | SPARSE TARGET | 3.402 | — | — | — | 100% |
| 98 | ToneCurve_Pt6_Y | [0, 255] | HEALTHY | 5.483 | 0.68 | — | 0.05 | 6% |

### Tone Curves (Red) (idx 99-110)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 99 | ToneCurveRed_Pt1_X | [0, 255] | SPARSE TARGET | 0.011 | — | — | — | 100% |
| 100 | ToneCurveRed_Pt1_Y | [0, 255] | SPARSE TARGET | 0.044 | — | — | — | 100% |
| 101 | ToneCurveRed_Pt2_X | [0, 255] | HEALTHY | 2.226 | 0.52 | — | -0.09 | 65% |
| 102 | ToneCurveRed_Pt2_Y | [0, 255] | COLLAPSED | 5.466 | 0.09 | — | 0.14 | 3% |
| 103 | ToneCurveRed_Pt3_X | [0, 255] | HIGH ERROR | 9.423 | 0.16 | — | 0.14 | 53% |
| 104 | ToneCurveRed_Pt3_Y | [0, 255] | HIGH ERROR | 15.212 | 0.11 | — | 0.19 | 10% |
| 105 | ToneCurveRed_Pt4_X | [0, 255] | HIGH ERROR | 14.992 | 0.15 | — | 0.15 | 53% |
| 106 | ToneCurveRed_Pt4_Y | [0, 255] | HIGH ERROR | 15.233 | 0.17 | — | 0.16 | 3% |
| 107 | ToneCurveRed_Pt5_X | [0, 255] | HIGH ERROR | 16.669 | 0.19 | — | 0.15 | 53% |
| 108 | ToneCurveRed_Pt5_Y | [0, 255] | HIGH ERROR | 11.285 | 0.29 | — | 0.14 | 3% |
| 109 | ToneCurveRed_Pt6_X | [0, 255] | SPARSE TARGET | 3.275 | — | — | — | 100% |
| 110 | ToneCurveRed_Pt6_Y | [0, 255] | SPARSE TARGET | 3.476 | — | — | — | 100% |

### Tone Curves (Green) (idx 111-122)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 111 | ToneCurveGreen_Pt1_X | [0, 255] | SPARSE TARGET | 0.041 | — | — | — | 100% |
| 112 | ToneCurveGreen_Pt1_Y | [0, 255] | SPARSE TARGET | 0.016 | — | — | — | 100% |
| 113 | ToneCurveGreen_Pt2_X | [0, 255] | SPARSE TARGET | 1.228 | 0.50 | — | -0.02 | 88% |
| 114 | ToneCurveGreen_Pt2_Y | [0, 255] | HIGH ERROR | 5.149 | 0.10 | — | 0.07 | 3% |
| 115 | ToneCurveGreen_Pt3_X | [0, 255] | SPARSE TARGET | 8.886 | 0.31 | — | 0.06 | 88% |
| 116 | ToneCurveGreen_Pt3_Y | [0, 255] | HIGH ERROR | 13.112 | 0.21 | — | 0.08 | 45% |
| 117 | ToneCurveGreen_Pt4_X | [0, 255] | SPARSE TARGET | 17.064 | 0.23 | — | 0.06 | 88% |
| 118 | ToneCurveGreen_Pt4_Y | [0, 255] | HIGH ERROR | 18.409 | 0.23 | — | 0.02 | 3% |
| 119 | ToneCurveGreen_Pt5_X | [0, 255] | SPARSE TARGET | 13.420 | 0.41 | — | 0.04 | 88% |
| 120 | ToneCurveGreen_Pt5_Y | [0, 255] | HEALTHY | 9.024 | 0.56 | — | -0.01 | 3% |
| 121 | ToneCurveGreen_Pt6_X | [0, 255] | SPARSE TARGET | 3.301 | — | — | — | 100% |
| 122 | ToneCurveGreen_Pt6_Y | [0, 255] | SPARSE TARGET | 3.413 | — | — | — | 100% |

### Tone Curves (Blue) (idx 123-134)

| idx | field | range | category | mae | std_ratio | dir | corr | sparse |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 123 | ToneCurveBlue_Pt1_X | [0, 255] | SPARSE TARGET | 0.088 | — | — | — | 100% |
| 124 | ToneCurveBlue_Pt1_Y | [0, 255] | SPARSE TARGET | 0.014 | 0.03 | — | -0.02 | 100% |
| 125 | ToneCurveBlue_Pt2_X | [0, 255] | SPARSE TARGET | 7.007 | 0.16 | — | 0.14 | 80% |
| 126 | ToneCurveBlue_Pt2_Y | [0, 255] | COLLAPSED | 10.896 | 0.07 | — | 0.16 | 3% |
| 127 | ToneCurveBlue_Pt3_X | [0, 255] | SPARSE TARGET | 17.909 | 0.10 | — | 0.12 | 80% |
| 128 | ToneCurveBlue_Pt3_Y | [0, 255] | COLLAPSED | 24.552 | 0.08 | — | 0.15 | 49% |
| 129 | ToneCurveBlue_Pt4_X | [0, 255] | SPARSE TARGET | 17.133 | 0.20 | — | 0.13 | 80% |
| 130 | ToneCurveBlue_Pt4_Y | [0, 255] | HIGH ERROR | 20.147 | 0.16 | — | 0.10 | 3% |
| 131 | ToneCurveBlue_Pt5_X | [0, 255] | SPARSE TARGET | 14.833 | 0.29 | — | 0.12 | 80% |
| 132 | ToneCurveBlue_Pt5_Y | [0, 255] | HIGH ERROR | 10.479 | 0.40 | — | 0.09 | 3% |
| 133 | ToneCurveBlue_Pt6_X | [0, 255] | SPARSE TARGET | 3.317 | — | — | — | 100% |
| 134 | ToneCurveBlue_Pt6_Y | [0, 255] | SPARSE TARGET | 3.422 | — | — | — | 100% |

## 3. Detailed view — non-HEALTHY non-SPARSE sliders

### `SplitToningShadowHue` — COLLAPSED  (idx 44)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `48.087` (median `9.240`, p95 `160.696`)
- norm_mae: `0.1336` (mae / range_span)
- std(pred)=2.585, std(target)=89.414, ratio=0.029
- mean(pred)=38.252, mean(target)=93.865, gap=-55.613
- Pearson corr(pred, target): `-0.030`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurveBlue_Pt3_Y` — COLLAPSED  (idx 128)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `24.552` (median `25.359`, p95 `31.181`)
- norm_mae: `0.0963` (mae / range_span)
- std(pred)=1.690, std(target)=20.367, ratio=0.083
- mean(pred)=76.292, mean(target)=90.943, gap=-14.651
- Pearson corr(pred, target): `0.150`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurve_Pt3_Y` — COLLAPSED  (idx 92)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `12.174` (median `7.238`, p95 `47.544`)
- norm_mae: `0.0477` (mae / range_span)
- std(pred)=1.150, std(target)=18.375, ratio=0.063
- mean(pred)=73.554, mean(target)=73.182, gap=0.372
- Pearson corr(pred, target): `0.072`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurveBlue_Pt2_Y` — COLLAPSED  (idx 126)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `10.896` (median `10.198`, p95 `17.943`)
- norm_mae: `0.0427` (mae / range_span)
- std(pred)=0.601, std(target)=9.206, ratio=0.065
- mean(pred)=28.778, mean(target)=36.684, gap=-7.906
- Pearson corr(pred, target): `0.156`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurve_Pt3_X` — COLLAPSED  (idx 91)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `8.248` (median `4.021`, p95 `33.750`)
- norm_mae: `0.0323` (mae / range_span)
- std(pred)=1.147, std(target)=13.907, ratio=0.082
- mean(pred)=66.031, mean(target)=64.570, gap=1.461
- Pearson corr(pred, target): `0.119`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ColorGradeMidtoneHue` — COLLAPSED  (idx 47)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `10.763` (median `6.842`, p95 `36.652`)
- norm_mae: `0.0299` (mae / range_span)
- std(pred)=0.749, std(target)=14.510, ratio=0.052
- mean(pred)=34.791, mean(target)=41.898, gap=-7.107
- Pearson corr(pred, target): `-0.131`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurve_Pt2_Y` — COLLAPSED  (idx 90)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `6.387` (median `3.800`, p95 `19.752`)
- norm_mae: `0.0250` (mae / range_span)
- std(pred)=0.556, std(target)=8.769, ratio=0.063
- mean(pred)=25.280, mean(target)=25.594, gap=-0.315
- Pearson corr(pred, target): `0.139`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurveRed_Pt2_Y` — COLLAPSED  (idx 102)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `5.466` (median `3.240`, p95 `14.630`)
- norm_mae: `0.0214` (mae / range_span)
- std(pred)=0.621, std(target)=7.004, ratio=0.089
- mean(pred)=32.119, mean(target)=36.280, gap=-4.161
- Pearson corr(pred, target): `0.136`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `Tint` — COLLAPSED  (idx 12)

- range: `[-150.00, 150.00]`, default: `0.00`
- MAE: `6.146` (median `5.681`, p95 `13.519`)
- norm_mae: `0.0205` (mae / range_span)
- std(pred)=0.277, std(target)=7.183, ratio=0.039
- mean(pred)=8.407, mean(target)=8.015, gap=0.392
- direction correct on signed-range subset: `97.1%`
- Pearson corr(pred, target): `-0.001`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurve_Pt2_X` — COLLAPSED  (idx 89)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `5.014` (median `3.100`, p95 `11.047`)
- norm_mae: `0.0197` (mae / range_span)
- std(pred)=0.504, std(target)=7.489, ratio=0.067
- mean(pred)=24.415, mean(target)=24.215, gap=0.200
- Pearson corr(pred, target): `0.225`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `ToneCurve_Pt1_Y` — COLLAPSED  (idx 88)

- range: `[0.00, 255.00]`, default: `0.00`
- MAE: `3.298` (median `2.824`, p95 `10.190`)
- norm_mae: `0.0129` (mae / range_span)
- std(pred)=0.221, std(target)=4.390, ratio=0.050
- mean(pred)=10.141, mean(target)=10.208, gap=-0.067
- Pearson corr(pred, target): `-0.006`
- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.

### `SharpenEdgeMasking` — HIGH ERROR  (idx 67)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `10.570` (median `4.093`, p95 `63.478`)
- norm_mae: `0.1057` (mae / range_span)
- std(pred)=10.292, std(target)=24.321, ratio=0.423
- mean(pred)=83.364, mean(target)=80.146, gap=3.218
- Pearson corr(pred, target): `0.509`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `GreenHue` — HIGH ERROR  (idx 60)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `20.710` (median `20.885`, p95 `36.137`)
- norm_mae: `0.1036` (mae / range_span)
- std(pred)=10.609, std(target)=19.583, ratio=0.542
- mean(pred)=29.797, mean(target)=15.327, gap=14.470
- direction correct on signed-range subset: `99.8%`
- Pearson corr(pred, target): `0.451`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Sharpness` — HIGH ERROR  (idx 64)

- range: `[0.00, 150.00]`, default: `25.00`
- MAE: `14.843` (median `5.477`, p95 `43.891`)
- norm_mae: `0.0990` (mae / range_span)
- std(pred)=2.718, std(target)=20.092, ratio=0.135
- mean(pred)=52.003, mean(target)=40.870, gap=11.133
- Pearson corr(pred, target): `0.307`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentOrange` — HIGH ERROR  (idx 22)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `17.178` (median `20.174`, p95 `33.396`)
- norm_mae: `0.0859` (mae / range_span)
- std(pred)=6.004, std(target)=19.584, ratio=0.307
- mean(pred)=-0.210, mean(target)=-0.074, gap=-0.136
- direction correct on signed-range subset: `58.4%`
- Pearson corr(pred, target): `-0.021`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt4_Y` — HIGH ERROR  (idx 130)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `20.147` (median `19.786`, p95 `28.203`)
- norm_mae: `0.0790` (mae / range_span)
- std(pred)=2.531, std(target)=15.488, ratio=0.163
- mean(pred)=139.879, mean(target)=153.982, gap=-14.103
- Pearson corr(pred, target): `0.103`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeShadowLum` — HIGH ERROR  (idx 46)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `15.444` (median `10.429`, p95 `50.459`)
- norm_mae: `0.0772` (mae / range_span)
- std(pred)=9.495, std(target)=19.592, ratio=0.485
- mean(pred)=-9.480, mean(target)=-1.051, gap=-8.429
- direction correct on signed-range subset: `70.0%`
- Pearson corr(pred, target): `0.194`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Shadows2012` — HIGH ERROR  (idx 3)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `15.162` (median `11.610`, p95 `34.090`)
- norm_mae: `0.0758` (mae / range_span)
- std(pred)=5.951, std(target)=17.021, ratio=0.350
- mean(pred)=28.592, mean(target)=18.153, gap=10.439
- direction correct on signed-range subset: `99.9%`
- Pearson corr(pred, target): `0.356`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt4_Y` — HIGH ERROR  (idx 118)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `18.409` (median `19.175`, p95 `26.437`)
- norm_mae: `0.0722` (mae / range_span)
- std(pred)=2.643, std(target)=11.538, ratio=0.229
- mean(pred)=142.453, mean(target)=157.704, gap=-15.250
- Pearson corr(pred, target): `0.023`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt5_X` — HIGH ERROR  (idx 107)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `16.669` (median `17.082`, p95 `25.823`)
- norm_mae: `0.0654` (mae / range_span)
- std(pred)=3.266, std(target)=16.839, ratio=0.194
- mean(pred)=181.739, mean(target)=188.192, gap=-6.454
- Pearson corr(pred, target): `0.153`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt4_X` — HIGH ERROR  (idx 93)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `16.189` (median `17.700`, p95 `41.539`)
- norm_mae: `0.0635` (mae / range_span)
- std(pred)=2.413, std(target)=17.664, ratio=0.137
- mean(pred)=155.835, mean(target)=145.370, gap=10.465
- Pearson corr(pred, target): `-0.032`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt4_Y` — HIGH ERROR  (idx 106)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `15.233` (median `14.028`, p95 `24.961`)
- norm_mae: `0.0597` (mae / range_span)
- std(pred)=2.626, std(target)=15.653, ratio=0.168
- mean(pred)=139.537, mean(target)=144.377, gap=-4.839
- Pearson corr(pred, target): `0.162`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt3_Y` — HIGH ERROR  (idx 104)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `15.212` (median `15.065`, p95 `21.490`)
- norm_mae: `0.0597` (mae / range_span)
- std(pred)=1.644, std(target)=15.301, ratio=0.107
- mean(pred)=79.336, mean(target)=84.223, gap=-4.887
- Pearson corr(pred, target): `0.194`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt4_X` — HIGH ERROR  (idx 105)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `14.992` (median `16.439`, p95 `22.833`)
- norm_mae: `0.0588` (mae / range_span)
- std(pred)=2.285, std(target)=14.976, ratio=0.153
- mean(pred)=132.716, mean(target)=138.917, gap=-6.201
- Pearson corr(pred, target): `0.147`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentPurple` — HIGH ERROR  (idx 27)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `11.049` (median `10.071`, p95 `21.992`)
- norm_mae: `0.0552` (mae / range_span)
- std(pred)=5.879, std(target)=11.662, ratio=0.504
- mean(pred)=1.254, mean(target)=6.810, gap=-5.556
- direction correct on signed-range subset: `58.9%`
- Pearson corr(pred, target): `0.183`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceSmoothing` — HIGH ERROR  (idx 68)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `5.295` (median `4.264`, p95 `10.696`)
- norm_mae: `0.0529` (mae / range_span)
- std(pred)=1.638, std(target)=4.354, ratio=0.376
- mean(pred)=13.510, mean(target)=17.957, gap=-4.447
- Pearson corr(pred, target): `0.076`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeMidtoneSat` — HIGH ERROR  (idx 48)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `5.270` (median `4.934`, p95 `11.775`)
- norm_mae: `0.0527` (mae / range_span)
- std(pred)=2.090, std(target)=6.240, ratio=0.335
- mean(pred)=8.156, mean(target)=6.113, gap=2.044
- Pearson corr(pred, target): `0.299`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentYellow` — HIGH ERROR  (idx 31)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `10.353` (median `10.080`, p95 `22.067`)
- norm_mae: `0.0518` (mae / range_span)
- std(pred)=6.134, std(target)=12.732, ratio=0.482
- mean(pred)=17.659, mean(target)=15.371, gap=2.288
- direction correct on signed-range subset: `91.3%`
- Pearson corr(pred, target): `0.336`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentOrange` — HIGH ERROR  (idx 30)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `10.295` (median `8.724`, p95 `23.154`)
- norm_mae: `0.0515` (mae / range_span)
- std(pred)=4.853, std(target)=11.767, ratio=0.412
- mean(pred)=6.902, mean(target)=13.666, gap=-6.764
- direction correct on signed-range subset: `89.4%`
- Pearson corr(pred, target): `0.428`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt3_Y` — HIGH ERROR  (idx 116)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `13.112` (median `13.452`, p95 `16.819`)
- norm_mae: `0.0514` (mae / range_span)
- std(pred)=1.636, std(target)=7.905, ratio=0.207
- mean(pred)=86.054, mean(target)=96.798, gap=-10.743
- Pearson corr(pred, target): `0.079`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Blacks2012` — HIGH ERROR  (idx 5)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `10.265` (median `9.766`, p95 `23.476`)
- norm_mae: `0.0513` (mae / range_span)
- std(pred)=6.169, std(target)=12.367, ratio=0.499
- mean(pred)=8.369, mean(target)=13.287, gap=-4.918
- direction correct on signed-range subset: `78.2%`
- Pearson corr(pred, target): `0.348`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SaturationAdjustmentYellow` — HIGH ERROR  (idx 23)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `9.815` (median `8.599`, p95 `24.526`)
- norm_mae: `0.0491` (mae / range_span)
- std(pred)=4.103, std(target)=8.679, ratio=0.473
- mean(pred)=-3.167, mean(target)=-10.804, gap=7.637
- direction correct on signed-range subset: `76.2%`
- Pearson corr(pred, target): `0.166`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentMagenta` — HIGH ERROR  (idx 36)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `9.748` (median `8.723`, p95 `20.817`)
- norm_mae: `0.0487` (mae / range_span)
- std(pred)=3.458, std(target)=11.434, ratio=0.302
- mean(pred)=0.665, mean(target)=-2.494, gap=3.158
- direction correct on signed-range subset: `55.4%`
- Pearson corr(pred, target): `0.041`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt5_Y` — HIGH ERROR  (idx 108)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `11.285` (median `10.459`, p95 `21.369`)
- norm_mae: `0.0443` (mae / range_span)
- std(pred)=3.472, std(target)=11.781, ratio=0.295
- mean(pred)=195.728, mean(target)=200.398, gap=-4.670
- Pearson corr(pred, target): `0.139`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningShadowSaturation` — HIGH ERROR  (idx 45)

- range: `[0.00, 100.00]`, default: `0.00`
- MAE: `4.179` (median `5.133`, p95 `6.245`)
- norm_mae: `0.0418` (mae / range_span)
- std(pred)=0.813, std(target)=3.394, ratio=0.239
- mean(pred)=9.237, mean(target)=6.226, gap=3.010
- Pearson corr(pred, target): `-0.137`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveBlue_Pt5_Y` — HIGH ERROR  (idx 132)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `10.479` (median `10.776`, p95 `17.813`)
- norm_mae: `0.0411` (mae / range_span)
- std(pred)=3.286, std(target)=8.186, ratio=0.401
- mean(pred)=201.290, mean(target)=208.975, gap=-7.685
- Pearson corr(pred, target): `0.085`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `GreenSaturation` — HIGH ERROR  (idx 61)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `8.092` (median `5.943`, p95 `19.243`)
- norm_mae: `0.0405` (mae / range_span)
- std(pred)=3.164, std(target)=10.656, ratio=0.297
- mean(pred)=-3.518, mean(target)=-1.054, gap=-2.464
- direction correct on signed-range subset: `71.4%`
- Pearson corr(pred, target): `0.379`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `RedHue` — HIGH ERROR  (idx 58)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.923` (median `8.078`, p95 `12.844`)
- norm_mae: `0.0396` (mae / range_span)
- std(pred)=1.388, std(target)=8.645, ratio=0.161
- mean(pred)=14.135, mean(target)=17.351, gap=-3.216
- direction correct on signed-range subset: `99.9%`
- Pearson corr(pred, target): `0.349`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `RedSaturation` — HIGH ERROR  (idx 59)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.592` (median `6.239`, p95 `17.136`)
- norm_mae: `0.0380` (mae / range_span)
- std(pred)=1.934, std(target)=8.225, ratio=0.235
- mean(pred)=-3.215, mean(target)=-6.490, gap=3.276
- direction correct on signed-range subset: `97.7%`
- Pearson corr(pred, target): `0.105`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt4_Y` — HIGH ERROR  (idx 94)

- range: `[0.00, 255.00]`, default: `153.00`
- MAE: `9.430` (median `10.585`, p95 `18.606`)
- norm_mae: `0.0370` (mae / range_span)
- std(pred)=2.307, std(target)=10.582, ratio=0.218
- mean(pred)=152.863, mean(target)=149.341, gap=3.522
- Pearson corr(pred, target): `0.031`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveRed_Pt3_X` — HIGH ERROR  (idx 103)

- range: `[0.00, 255.00]`, default: `102.00`
- MAE: `9.423` (median `9.497`, p95 `13.428`)
- norm_mae: `0.0370` (mae / range_span)
- std(pred)=1.522, std(target)=9.483, ratio=0.161
- mean(pred)=90.323, mean(target)=93.083, gap=-2.760
- Pearson corr(pred, target): `0.139`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ColorGradeMidtoneLum` — HIGH ERROR  (idx 49)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.352` (median `3.115`, p95 `20.356`)
- norm_mae: `0.0368` (mae / range_span)
- std(pred)=2.325, std(target)=9.989, ratio=0.233
- mean(pred)=19.126, mean(target)=15.132, gap=3.994
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.161`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Highlights2012` — HIGH ERROR  (idx 2)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.323` (median `5.604`, p95 `24.569`)
- norm_mae: `0.0366` (mae / range_span)
- std(pred)=3.656, std(target)=10.041, ratio=0.364
- mean(pred)=-28.014, mean(target)=-29.423, gap=1.409
- direction correct on signed-range subset: `99.7%`
- Pearson corr(pred, target): `0.165`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentYellow` — HIGH ERROR  (idx 15)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.184` (median `6.543`, p95 `15.651`)
- norm_mae: `0.0359` (mae / range_span)
- std(pred)=5.930, std(target)=7.875, ratio=0.753
- mean(pred)=2.678, mean(target)=7.811, gap=-5.133
- direction correct on signed-range subset: `78.4%`
- Pearson corr(pred, target): `0.517`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentBlue` — HIGH ERROR  (idx 18)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.154` (median `7.289`, p95 `12.844`)
- norm_mae: `0.0358` (mae / range_span)
- std(pred)=2.775, std(target)=7.977, ratio=0.348
- mean(pred)=0.065, mean(target)=3.690, gap=-3.625
- direction correct on signed-range subset: `66.3%`
- Pearson corr(pred, target): `0.417`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Vibrance` — HIGH ERROR  (idx 9)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.907` (median `6.268`, p95 `15.216`)
- norm_mae: `0.0345` (mae / range_span)
- std(pred)=3.783, std(target)=7.880, ratio=0.480
- mean(pred)=9.115, mean(target)=6.942, gap=2.173
- direction correct on signed-range subset: `98.9%`
- Pearson corr(pred, target): `0.173`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Exposure2012` — HIGH ERROR  (idx 0)

- range: `[-5.00, 5.00]`, default: `0.00`
- MAE: `0.342` (median `0.293`, p95 `0.841`)
- norm_mae: `0.0342` (mae / range_span)
- std(pred)=0.256, std(target)=0.475, ratio=0.538
- mean(pred)=0.352, mean(target)=0.272, gap=0.080
- direction correct on signed-range subset: `78.2%`
- Pearson corr(pred, target): `0.445`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentOrange` — HIGH ERROR  (idx 14)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.396` (median `6.412`, p95 `13.724`)
- norm_mae: `0.0320` (mae / range_span)
- std(pred)=2.904, std(target)=6.259, ratio=0.464
- mean(pred)=13.651, mean(target)=9.659, gap=3.991
- direction correct on signed-range subset: `99.9%`
- Pearson corr(pred, target): `0.124`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentAqua` — HIGH ERROR  (idx 17)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.379` (median `5.625`, p95 `13.001`)
- norm_mae: `0.0319` (mae / range_span)
- std(pred)=3.228, std(target)=7.153, ratio=0.451
- mean(pred)=9.602, mean(target)=10.455, gap=-0.852
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.100`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `BlueSaturation` — HIGH ERROR  (idx 63)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `5.092` (median `2.394`, p95 `12.568`)
- norm_mae: `0.0255` (mae / range_span)
- std(pred)=1.514, std(target)=7.011, ratio=0.216
- mean(pred)=-5.865, mean(target)=-3.329, gap=-2.537
- direction correct on signed-range subset: `66.9%`
- Pearson corr(pred, target): `0.486`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurve_Pt5_Y` — HIGH ERROR  (idx 96)

- range: `[0.00, 255.00]`, default: `204.00`
- MAE: `5.869` (median `4.384`, p95 `16.544`)
- norm_mae: `0.0230` (mae / range_span)
- std(pred)=2.943, std(target)=7.775, ratio=0.378
- mean(pred)=188.182, mean(target)=187.645, gap=0.538
- Pearson corr(pred, target): `0.239`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `BlueHue` — HIGH ERROR  (idx 62)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `4.593` (median `3.257`, p95 `10.758`)
- norm_mae: `0.0230` (mae / range_span)
- std(pred)=1.727, std(target)=6.067, ratio=0.285
- mean(pred)=-9.179, mean(target)=-7.734, gap=-1.445
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.422`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Dehaze` — HIGH ERROR  (idx 7)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `4.507` (median `4.073`, p95 `9.613`)
- norm_mae: `0.0225` (mae / range_span)
- std(pred)=2.392, std(target)=6.058, ratio=0.395
- mean(pred)=7.781, mean(target)=8.551, gap=-0.770
- direction correct on signed-range subset: `98.8%`
- Pearson corr(pred, target): `0.407`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `HueAdjustmentRed` — HIGH ERROR  (idx 13)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `4.287` (median `1.784`, p95 `13.862`)
- norm_mae: `0.0214` (mae / range_span)
- std(pred)=2.571, std(target)=6.295, ratio=0.408
- mean(pred)=8.152, mean(target)=10.491, gap=-2.338
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `0.279`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ToneCurveGreen_Pt2_Y` — HIGH ERROR  (idx 114)

- range: `[0.00, 255.00]`, default: `51.00`
- MAE: `5.149` (median `2.610`, p95 `12.199`)
- norm_mae: `0.0202` (mae / range_span)
- std(pred)=0.666, std(target)=6.637, ratio=0.100
- mean(pred)=36.093, mean(target)=37.347, gap=-1.254
- Pearson corr(pred, target): `0.071`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Clarity2012` — HIGH ERROR  (idx 6)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.933` (median `2.324`, p95 `9.877`)
- norm_mae: `0.0197` (mae / range_span)
- std(pred)=1.006, std(target)=4.437, ratio=0.227
- mean(pred)=-5.789, mean(target)=-8.887, gap=3.097
- direction correct on signed-range subset: `99.5%`
- Pearson corr(pred, target): `0.057`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ParametricMidtoneSplit` — HIGH ERROR  (idx 42)

- range: `[0.00, 100.00]`, default: `50.00`
- MAE: `1.759` (median `1.832`, p95 `2.552`)
- norm_mae: `0.0176` (mae / range_span)
- std(pred)=0.601, std(target)=0.000, ratio=—
- mean(pred)=61.747, mean(target)=60.000, gap=1.747
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `LuminanceAdjustmentRed` — HIGH ERROR  (idx 29)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.325` (median `2.023`, p95 `15.642`)
- norm_mae: `0.0166` (mae / range_span)
- std(pred)=3.205, std(target)=6.837, ratio=0.469
- mean(pred)=12.282, mean(target)=12.747, gap=-0.464
- direction correct on signed-range subset: `99.7%`
- Pearson corr(pred, target): `0.593`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningHighlightHue` — HIGH ERROR  (idx 50)

- range: `[0.00, 360.00]`, default: `0.00`
- MAE: `5.781` (median `3.175`, p95 `14.475`)
- norm_mae: `0.0161` (mae / range_span)
- std(pred)=3.190, std(target)=9.754, ratio=0.327
- mean(pred)=45.803, mean(target)=43.104, gap=2.699
- Pearson corr(pred, target): `0.147`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `ParametricShadowSplit` — HIGH ERROR  (idx 43)

- range: `[0.00, 100.00]`, default: `25.00`
- MAE: `0.657` (median `0.675`, p95 `0.849`)
- norm_mae: `0.0066` (mae / range_span)
- std(pred)=0.140, std(target)=0.000, ratio=—
- mean(pred)=14.657, mean(target)=14.000, gap=0.657
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `SplitToningBalance` — HIGH ERROR  (idx 57)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.403` (median `0.346`, p95 `0.938`)
- norm_mae: `0.0020` (mae / range_span)
- std(pred)=0.422, std(target)=0.000, ratio=—
- mean(pred)=29.743, mean(target)=30.000, gap=-0.257
- direction correct on signed-range subset: `100.0%`
- Pearson corr(pred, target): `—`
- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.

### `Whites2012` — WRONG DIRECTION  (idx 4)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `15.383` (median `14.610`, p95 `28.846`)
- norm_mae: `0.0769` (mae / range_span)
- std(pred)=8.284, std(target)=15.789, ratio=0.525
- mean(pred)=-9.849, mean(target)=-1.185, gap=-8.664
- direction correct on signed-range subset: `38.2%`
- Pearson corr(pred, target): `0.254`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `SaturationAdjustmentMagenta` — WRONG DIRECTION  (idx 28)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `11.124` (median `11.367`, p95 `21.482`)
- norm_mae: `0.0556` (mae / range_span)
- std(pred)=5.439, std(target)=12.250, ratio=0.444
- mean(pred)=-1.959, mean(target)=4.011, gap=-5.970
- direction correct on signed-range subset: `47.7%`
- Pearson corr(pred, target): `0.259`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `SaturationAdjustmentGreen` — WRONG DIRECTION  (idx 24)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `9.050` (median `7.190`, p95 `20.449`)
- norm_mae: `0.0452` (mae / range_span)
- std(pred)=6.233, std(target)=10.390, ratio=0.600
- mean(pred)=-1.382, mean(target)=2.287, gap=-3.668
- direction correct on signed-range subset: `54.0%`
- Pearson corr(pred, target): `0.076`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `HueAdjustmentGreen` — WRONG DIRECTION  (idx 16)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `8.144` (median `7.613`, p95 `16.691`)
- norm_mae: `0.0407` (mae / range_span)
- std(pred)=5.891, std(target)=8.592, ratio=0.686
- mean(pred)=-3.478, mean(target)=2.571, gap=-6.049
- direction correct on signed-range subset: `49.4%`
- Pearson corr(pred, target): `0.539`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `SaturationAdjustmentRed` — WRONG DIRECTION  (idx 21)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.502` (median `7.094`, p95 `15.318`)
- norm_mae: `0.0375` (mae / range_span)
- std(pred)=3.932, std(target)=5.360, ratio=0.734
- mean(pred)=1.177, mean(target)=6.916, gap=-5.739
- direction correct on signed-range subset: `54.9%`
- Pearson corr(pred, target): `0.039`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `Contrast2012` — WRONG DIRECTION  (idx 1)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.396` (median `7.343`, p95 `13.167`)
- norm_mae: `0.0370` (mae / range_span)
- std(pred)=3.150, std(target)=8.011, ratio=0.393
- mean(pred)=-3.452, mean(target)=-1.861, gap=-1.591
- direction correct on signed-range subset: `53.3%`
- Pearson corr(pred, target): `0.173`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `LuminanceAdjustmentBlue` — WRONG DIRECTION  (idx 34)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `7.371` (median `7.443`, p95 `14.730`)
- norm_mae: `0.0369` (mae / range_span)
- std(pred)=3.852, std(target)=6.905, ratio=0.558
- mean(pred)=6.402, mean(target)=1.631, gap=4.771
- direction correct on signed-range subset: `46.6%`
- Pearson corr(pred, target): `0.078`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `LuminanceAdjustmentPurple` — WRONG DIRECTION  (idx 35)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `6.513` (median `7.115`, p95 `12.963`)
- norm_mae: `0.0326` (mae / range_span)
- std(pred)=2.279, std(target)=8.240, ratio=0.277
- mean(pred)=-1.207, mean(target)=-2.254, gap=1.047
- direction correct on signed-range subset: `47.7%`
- Pearson corr(pred, target): `-0.049`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `Saturation` — WRONG DIRECTION  (idx 10)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `5.883` (median `5.219`, p95 `15.739`)
- norm_mae: `0.0294` (mae / range_span)
- std(pred)=2.193, std(target)=5.068, ratio=0.433
- mean(pred)=2.823, mean(target)=-2.415, gap=5.238
- direction correct on signed-range subset: `30.0%`
- Pearson corr(pred, target): `0.268`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `Texture` — WRONG DIRECTION  (idx 8)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `3.443` (median `1.711`, p95 `8.446`)
- norm_mae: `0.0172` (mae / range_span)
- std(pred)=1.295, std(target)=3.610, ratio=0.359
- mean(pred)=1.514, mean(target)=-1.483, gap=2.997
- direction correct on signed-range subset: `19.7%`
- Pearson corr(pred, target): `0.172`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

### `ColorGradeHighlightLum` — WRONG DIRECTION  (idx 52)

- range: `[-100.00, 100.00]`, default: `0.00`
- MAE: `0.856` (median `0.535`, p95 `2.190`)
- norm_mae: `0.0043` (mae / range_span)
- std(pred)=0.452, std(target)=1.144, ratio=0.395
- mean(pred)=-1.822, mean(target)=-1.378, gap=-0.443
- direction correct on signed-range subset: `0.0%`
- Pearson corr(pred, target): `0.050`
- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.

## 4. Tone curve identity-collapse check

Documented issue (HANDOVER Part 6 item 9): the model converged to near-identity curve predictions across all 4 channels in v1.0.1. Quantifying here against the current shipping v1.2.3.

| Channel | mean L2 (pred ↔ identity) | mean L2 (target ↔ identity) | identity bias |
|---|---:|---:|---|
| Composite | 15.82 | 19.05 | OK |
| Red | 26.00 | 22.81 | OK |
| Green | 23.27 | 19.84 | OK |
| Blue | 22.14 | 18.80 | OK |

_Distance = mean over photos of L2-norm between (Pt2_Y..Pt5_Y) vs identity (Pt_n_Y == Pt_n_X)._ A near-zero pred-vs-identity distance with non-zero target-vs-identity distance confirms collapse.

## 5. Temperature dual-view (log-K + Kelvin)

- log-K MAE (prediction space): `0.1752` (typical training-loss scale)
- Kelvin MAE (user-facing): `731 K` (target was <250 K per HANDOVER)
- Kelvin p95 abs error: `1889 K`

## 6. Worst-offender photos (top 20 by weighted error)

Weighting: sum over fields of `|pred - target| / range_span` (Temperature uses Kelvin/range_K).

**1. `9afc601330b519e7...`** — shoot `13679_Canon_EOS_5D_Mark_IV`, total weighted error `6.233`
   - ISO 160, Canon EOS 5D Mark IV, focal 42.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+56.537), ColorGradeMidtoneHue (Δ=-168.330), SplitToningShadowHue (Δ=-166.994), LuminanceAdjustmentAqua (Δ=+68.956), LuminanceAdjustmentBlue (Δ=+66.461)

**2. `968d8398e2962acf...`** — shoot `18685_Canon_EOS_R6`, total weighted error `6.088`
   - ISO 500, Canon EOS R6, focal 48.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+94.849), ColorGradeShadowLum (Δ=-51.574), SaturationAdjustmentMagenta (Δ=+35.573), Whites2012 (Δ=+33.942), SaturationAdjustmentPurple (Δ=+31.996)

**3. `90d38592400bd938...`** — shoot `18689_Canon_EOS_R6`, total weighted error `6.010`
   - ISO 125, Canon EOS R6, focal 41.0 mm
   - top contributing sliders: LuminanceAdjustmentPurple (Δ=+97.693), SaturationAdjustmentPurple (Δ=+91.028), SaturationAdjustmentMagenta (Δ=+86.504), SharpenEdgeMasking (Δ=-31.517), LuminanceAdjustmentMagenta (Δ=+61.807)

**4. `28fa219c694897f4...`** — shoot `13679_Canon_EOS_5D_Mark_IV`, total weighted error `5.966`
   - ISO 160, Canon EOS 5D Mark IV, focal 35.0 mm
   - top contributing sliders: SplitToningShadowHue (Δ=-177.413), ColorGradeMidtoneHue (Δ=-167.552), LuminanceAdjustmentAqua (Δ=+76.984), LuminanceAdjustmentOrange (Δ=-60.744), LuminanceAdjustmentBlue (Δ=+57.977)

**5. `f25b2231626bc3a9...`** — shoot `18439_Canon_EOS_R5_C`, total weighted error `5.912`
   - ISO 10000, Canon EOS R5 C, focal 38.0 mm
   - top contributing sliders: ColorGradeBlending (Δ=-50.018), Blacks2012 (Δ=+46.597), ColorGradeShadowLum (Δ=-44.491), GrainFrequency (Δ=-18.362), LuminanceAdjustmentOrange (Δ=-34.886)

**6. `552decdd8e624b58...`** — shoot `19124_Canon_EOS_R6`, total weighted error `5.747`
   - ISO 320, Canon EOS R6, focal 69.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+90.207), Whites2012 (Δ=+53.296), SaturationAdjustmentMagenta (Δ=+44.589), ColorGradeShadowLum (Δ=-41.005), SaturationAdjustmentPurple (Δ=+34.958)

**7. `b3e1b22479f997bc...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.619`
   - ISO 800, Canon EOS R5, focal 79.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+89.590), Whites2012 (Δ=+60.334), ColorGradeShadowLum (Δ=-42.481), SaturationAdjustmentOrange (Δ=+31.536), SaturationAdjustmentMagenta (Δ=+27.752)

**8. `030f865d392bd85e...`** — shoot `18689_Canon_EOS_R6`, total weighted error `5.601`
   - ISO 125, Canon EOS R6, focal 41.0 mm
   - top contributing sliders: LuminanceAdjustmentPurple (Δ=+97.466), SaturationAdjustmentPurple (Δ=+91.003), SaturationAdjustmentMagenta (Δ=+86.458), SharpenEdgeMasking (Δ=-32.466), LuminanceAdjustmentMagenta (Δ=+61.672)

**9. `043dcad0d4d46364...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.500`
   - ISO 800, Canon EOS R5, focal 92.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+87.413), Whites2012 (Δ=+67.374), ColorGradeShadowLum (Δ=-44.163), SaturationAdjustmentOrange (Δ=+32.028), SaturationAdjustmentMagenta (Δ=+25.679)

**10. `0385fad0f58cd8b8...`** — shoot `18689_Canon_EOS_R6`, total weighted error `5.488`
   - ISO 125, Canon EOS R6, focal 41.0 mm
   - top contributing sliders: SaturationAdjustmentPurple (Δ=+100.179), LuminanceAdjustmentPurple (Δ=+99.650), SaturationAdjustmentMagenta (Δ=+98.414), LuminanceAdjustmentMagenta (Δ=+62.400), Sharpness (Δ=+29.155)

**11. `3c714cbc9d212969...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.484`
   - ISO 1250, Canon EOS R5, focal 89.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+84.491), Whites2012 (Δ=+59.147), ColorGradeShadowLum (Δ=-43.329), SaturationAdjustmentMagenta (Δ=+25.964), SaturationAdjustmentOrange (Δ=+25.543)

**12. `3b3e0eb997297afb...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.477`
   - ISO 800, Canon EOS R5, focal 92.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+87.095), Whites2012 (Δ=+65.494), ColorGradeShadowLum (Δ=-42.090), SaturationAdjustmentOrange (Δ=+31.279), LuminanceSmoothing (Δ=+12.857)

**13. `45bbbcc1bad24eb4...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.397`
   - ISO 2500, Canon EOS R5, focal 70.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+85.454), Whites2012 (Δ=+71.939), ColorGradeShadowLum (Δ=-43.558), SaturationAdjustmentOrange (Δ=+32.521), LuminanceSmoothing (Δ=+13.765)

**14. `48376c232167bf1d...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.339`
   - ISO 2000, Canon EOS R5, focal 28.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+89.027), Whites2012 (Δ=+64.068), ColorGradeShadowLum (Δ=-34.934), LuminanceSmoothing (Δ=+12.657), SaturationAdjustmentOrange (Δ=+25.138)

**15. `6a42862535a0b43f...`** — shoot `18594_Canon_EOS_R6`, total weighted error `5.272`
   - ISO 4000, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: GrainFrequency (Δ=-24.140), GreenHue (Δ=-44.485), ToneCurve_Pt4_X (Δ=+43.999), Blacks2012 (Δ=-33.860), ColorGradeShadowLum (Δ=-32.649)

**16. `3b37a8ad9a8049f1...`** — shoot `18522_Canon_EOS_R6`, total weighted error `5.267`
   - ISO 6400, Canon EOS R6, focal 24.0 mm
   - top contributing sliders: GrainFrequency (Δ=-25.080), ColorGradeShadowLum (Δ=-40.209), ToneCurve_Pt4_X (Δ=+39.364), ColorGradeMidtoneSat (Δ=-14.998), Sharpness (Δ=+22.069)

**17. `b04e6fa6923f658f...`** — shoot `18767_Canon_EOS_R6`, total weighted error `5.261`
   - ISO 2000, Canon EOS R6, focal 50.0 mm
   - top contributing sliders: SplitToningShadowHue (Δ=-197.445), Sharpness (Δ=+44.065), GreenHue (Δ=+43.870), SaturationAdjustmentOrange (Δ=+35.911), Shadows2012 (Δ=+34.249)

**18. `a474bc6149c22fd5...`** — shoot `18522_Canon_EOS_R6`, total weighted error `5.258`
   - ISO 6400, Canon EOS R6, focal 35.0 mm
   - top contributing sliders: GrainFrequency (Δ=-26.164), ColorGradeShadowLum (Δ=-37.918), ToneCurve_Pt4_X (Δ=+40.269), ColorGradeMidtoneSat (Δ=-14.960), Blacks2012 (Δ=-29.622)

**19. `3183a5159702ab02...`** — shoot `19120_Canon_EOS_R5`, total weighted error `5.249`
   - ISO 500, Canon EOS R5, focal 70.0 mm
   - top contributing sliders: SharpenEdgeMasking (Δ=+81.088), Whites2012 (Δ=+55.380), ColorGradeShadowLum (Δ=-36.231), SaturationAdjustmentOrange (Δ=+23.759), SaturationAdjustmentMagenta (Δ=+23.025)

**20. `6df1264f3fdc5d2f...`** — shoot `18685_Canon_EOS_R6`, total weighted error `5.246`
   - ISO 1250, Canon EOS R6, focal 37.0 mm
   - top contributing sliders: SaturationAdjustmentPurple (Δ=+49.973), ColorGradeShadowLum (Δ=-46.977), SaturationAdjustmentMagenta (Δ=+39.140), SaturationAdjustmentYellow (Δ=-29.853), Highlights2012 (Δ=+26.504)

## 7. Surprises / contradictions

- Temperature is classified HEALTHY despite the documented 730K-MAE failure mode in HANDOVER. Current Kelvin MAE: 731 K.
- Tone curve Composite is NOT showing the documented identity-collapse (pred-vs-identity 15.82 vs target-vs-identity 19.05). HANDOVER says all 4 channels collapse — verify.
- Tone curve Red is NOT showing the documented identity-collapse (pred-vs-identity 26.00 vs target-vs-identity 22.81). HANDOVER says all 4 channels collapse — verify.
- Tone curve Green is NOT showing the documented identity-collapse (pred-vs-identity 23.27 vs target-vs-identity 19.84). HANDOVER says all 4 channels collapse — verify.
- Tone curve Blue is NOT showing the documented identity-collapse (pred-vs-identity 22.14 vs target-vs-identity 18.80). HANDOVER says all 4 channels collapse — verify.
- COLLAPSED sliders found OUTSIDE the tone-curve panels: ['Tint', 'SplitToningShadowHue', 'ColorGradeMidtoneHue']. HANDOVER documents tone-curve collapse only — this is new.
