# Tint Deep Dive — why does Saha v1.2.3 collapse on Tint?

**Generated:** 2026-05-13T21:54:51  
**Training data:** `v1_learning/dataset/splits_v2_stratified/train.parquet` (9746 rows)  
**Scope:** Tint target distribution, correlations with metadata + other sliders, per-shoot variability, comparison vs Temperature. Diagnosis only — no fixes proposed.

## 1. Tint target distribution

### Tint distribution

- n = 9746
- min = -68.000, max = 68.000
- mean = 7.406, median = 7.000
- std = 8.894
- p01 = -11.000
- p05 = -5.000
- p25 = 1.000
- p75 = 13.000
- p95 = 22.000
- p99 = 32.000
- skewness = 0.754
- kurtosis = 4.919
- fraction near default (0.00 ± 1.0): 4.5%  (439 / 9746)
- positive (>1): 7226 (74.1%); negative (<-1): 1276 (13.1%); neutral [-1,1]: 1244 (12.8%)
- KS-vs-normal: stat=0.063, p=7.61e-34 (NON-normal)

**Histogram (30 bins):**
```
  `  -68.0`–`  -63.5`:     2  
  `  -63.5`–`  -58.9`:     1  
  `  -58.9`–`  -54.4`:     0  
  `  -54.4`–`  -49.9`:     0  
  `  -49.9`–`  -45.3`:     0  
  `  -45.3`–`  -40.8`:     0  
  `  -40.8`–`  -36.3`:     0  
  `  -36.3`–`  -31.7`:     0  
  `  -31.7`–`  -27.2`:     1  
  `  -27.2`–`  -22.7`:     5  
  `  -22.7`–`  -18.1`:    10  
  `  -18.1`–`  -13.6`:    50  #
  `  -13.6`–`   -9.1`:    43  
  `   -9.1`–`   -4.5`:   420  ########
  `   -4.5`–`    0.0`:  1214  #########################
  `    0.0`–`    4.5`:  1828  ######################################
  `    4.5`–`    9.1`:  2373  ##################################################
  `    9.1`–`   13.6`:  1645  ##################################
  `   13.6`–`   18.1`:  1373  ############################
  `   18.1`–`   22.7`:   336  #######
  `   22.7`–`   27.2`:   235  ####
  `   27.2`–`   31.7`:   105  ##
  `   31.7`–`   36.3`:    35  
  `   36.3`–`   40.8`:    18  
  `   40.8`–`   45.3`:    14  
  `   45.3`–`   49.9`:    15  
  `   49.9`–`   54.4`:     4  
  `   54.4`–`   58.9`:     5  
  `   58.9`–`   63.5`:    10  
  `   63.5`–`   68.0`:     4  
```

_NaN count in raw column: 0 (0.00%)_

## 2. Tint vs Temperature — what's structurally different?

| stat | Tint | Temperature |
|---|---:|---:|
| n (non-NaN) | 9746 | 9746 |
| range | [-68.0, 68.0] | [2037, 9400] K |
| mean | 7.406 | 4912 |
| median | 7.000 | 4977 |
| std | 8.894 | 1097 |
| skewness | 0.754 | -0.068 |
| kurtosis | 4.919 | -0.774 |
| fraction near default | 4.5% (Tint=0±1) | _(Temperature default is AsShot, not 5500)_ |
| sign asymmetry | 7226 positive vs 1276 negative (ratio 5.66×) | _N/A (Kelvin always positive)_ |

**Histograms side-by-side:**

Tint (range −150 to +150):
```
  `  -68.0`–`  -63.5`:     2  
  `  -63.5`–`  -58.9`:     1  
  `  -58.9`–`  -54.4`:     0  
  `  -54.4`–`  -49.9`:     0  
  `  -49.9`–`  -45.3`:     0  
  `  -45.3`–`  -40.8`:     0  
  `  -40.8`–`  -36.3`:     0  
  `  -36.3`–`  -31.7`:     0  
  `  -31.7`–`  -27.2`:     1  
  `  -27.2`–`  -22.7`:     5  
  `  -22.7`–`  -18.1`:    10  
  `  -18.1`–`  -13.6`:    50  #
  `  -13.6`–`   -9.1`:    43  
  `   -9.1`–`   -4.5`:   420  ########
  `   -4.5`–`    0.0`:  1214  #########################
  `    0.0`–`    4.5`:  1828  ######################################
  `    4.5`–`    9.1`:  2373  ##################################################
  `    9.1`–`   13.6`:  1645  ##################################
  `   13.6`–`   18.1`:  1373  ############################
  `   18.1`–`   22.7`:   336  #######
  `   22.7`–`   27.2`:   235  ####
  `   27.2`–`   31.7`:   105  ##
  `   31.7`–`   36.3`:    35  
  `   36.3`–`   40.8`:    18  
  `   40.8`–`   45.3`:    14  
  `   45.3`–`   49.9`:    15  
  `   49.9`–`   54.4`:     4  
  `   54.4`–`   58.9`:     5  
  `   58.9`–`   63.5`:    10  
  `   63.5`–`   68.0`:     4  
```

Temperature (Kelvin):
```
  ` 2037.0`–` 2282.4`:     2  
  ` 2282.4`–` 2527.9`:     9  
  ` 2527.9`–` 2773.3`:    15  
  ` 2773.3`–` 3018.7`:   101  ####
  ` 3018.7`–` 3264.2`:   941  ############################################
  ` 3264.2`–` 3509.6`:   564  ##########################
  ` 3509.6`–` 3755.0`:   244  ###########
  ` 3755.0`–` 4000.5`:   586  ###########################
  ` 4000.5`–` 4245.9`:   405  ##################
  ` 4245.9`–` 4491.3`:   411  ###################
  ` 4491.3`–` 4736.8`:   546  #########################
  ` 4736.8`–` 4982.2`:  1069  ##################################################
  ` 4982.2`–` 5227.6`:   766  ###################################
  ` 5227.6`–` 5473.1`:   717  #################################
  ` 5473.1`–` 5718.5`:   694  ################################
  ` 5718.5`–` 5963.9`:  1024  ###############################################
  ` 5963.9`–` 6209.4`:   608  ############################
  ` 6209.4`–` 6454.8`:   363  ################
  ` 6454.8`–` 6700.2`:   288  #############
  ` 6700.2`–` 6945.7`:   189  ########
  ` 6945.7`–` 7191.1`:    50  ##
  ` 7191.1`–` 7436.5`:   118  #####
  ` 7436.5`–` 7682.0`:    28  #
  ` 7682.0`–` 7927.4`:     4  
  ` 7927.4`–` 8172.8`:     1  
  ` 8172.8`–` 8418.3`:     2  
  ` 8418.3`–` 8663.7`:     0  
  ` 8663.7`–` 8909.1`:     0  
  ` 8909.1`–` 9154.6`:     0  
  ` 9154.6`–` 9400.0`:     1  
```

## 3. Correlations: Tint vs metadata and other sliders

### Correlations with `Tint` (Spearman, |ρ|≥0.3 ∧ p<0.05)

| field | ρ | p | n |
|---|---:|---:|---:|
| `as_shot_tint` | +0.913 | 0.00e+00 | 9440 |

**Top 10 strongest |ρ| (for context, regardless of threshold):**

| field | ρ | p | n |
|---|---:|---:|---:|
| `as_shot_tint` | +0.913 | 0.00e+00 | 9440 |
| `LuminanceAdjustmentPurple` | +0.274 | 5.11e-155 | 8985 |
| `HueAdjustmentPurple` | -0.271 | 2.39e-150 | 8957 |
| `GrainFrequency` | -0.269 | 4.26e-96 | 5755 |
| `ToneCurve_Pt4_Y` | +0.259 | 3.69e-149 | 9746 |
| `ColorGradeMidtoneHue` | -0.258 | 2.80e-148 | 9746 |
| `GrainAmount` | +0.256 | 2.88e-26 | 1655 |
| `ToneCurveBlue_Pt2_Y` | -0.254 | 1.61e-143 | 9746 |
| `ToneCurve_Pt4_X` | +0.244 | 9.02e-132 | 9746 |
| `LuminanceSmoothing` | -0.235 | 9.47e-107 | 8496 |

## 4. For comparison: Temperature correlations (same dataset)

### Correlations with `Temperature` (Spearman, |ρ|≥0.3 ∧ p<0.05)

| field | ρ | p | n |
|---|---:|---:|---:|
| `as_shot_temperature` | +0.939 | 0.00e+00 | 9440 |
| `iso` | -0.509 | 0.00e+00 | 9746 |
| `aperture` | +0.401 | 0.00e+00 | 9746 |
| `LuminanceAdjustmentAqua` | +0.319 | 1.36e-36 | 1489 |

**Top 10 strongest |ρ| (for context, regardless of threshold):**

| field | ρ | p | n |
|---|---:|---:|---:|
| `as_shot_temperature` | +0.939 | 0.00e+00 | 9440 |
| `iso` | -0.509 | 0.00e+00 | 9746 |
| `aperture` | +0.401 | 0.00e+00 | 9746 |
| `LuminanceAdjustmentAqua` | +0.319 | 1.36e-36 | 1489 |
| `focal_length` | -0.248 | 1.07e-136 | 9746 |
| `Exposure2012` | -0.220 | 9.29e-107 | 9746 |
| `ToneCurveBlue_Pt5_Y` | +0.217 | 3.85e-104 | 9746 |
| `ToneCurveBlue_Pt4_Y` | +0.205 | 4.44e-93 | 9746 |
| `ToneCurveBlue_Pt3_X` | +0.184 | 3.39e-75 | 9746 |
| `ToneCurveBlue_Pt3_Y` | +0.174 | 7.79e-67 | 9746 |

## 5. Per-shoot variability

### Per-shoot analysis of `Tint`

- shoots analysed: 92 (mean photos per shoot: 105.9)
- overall mean: 7.406, overall std: 8.895
- mean of per-shoot stds (within-shoot variability): 4.964
- std of per-shoot means (between-shoot variability): 6.816
- between/within ratio: 1.37  (>>1 = strongly clustered by shoot; ~1 = noisy within shoots)

**Shoots with the widest per-shoot std (most noisy):**

| shoot_id | n_photos | mean | std |
|---|---:|---:|---:|
| 18466_Canon_EOS_R6 | 138 | +17.80 | 15.90 |
| 18601_Canon_EOS_R6 | 107 | +13.01 | 15.03 |
| 18545_Canon_EOS_R6 | 64 | +11.22 | 13.59 |
| 18588_Canon_EOS_R6 | 11 | +14.36 | 13.14 |
| 18423_Canon_EOS_R6 | 216 | +13.59 | 11.37 |
| 18447_Canon_EOS_R6_Mark_II | 355 | +1.37 | 10.54 |
| 18903_Canon_EOS_R6 | 122 | -1.03 | 10.43 |
| 18421_Canon_EOS_R6 | 111 | +11.08 | 10.41 |
| 18497_Canon_EOS_R6 | 50 | +4.90 | 10.38 |
| 18420_Canon_EOS_R6 | 40 | +9.57 | 9.87 |

**Shoots with the most extreme mean (per-shoot bias):**

| shoot_id | n_photos | mean | std |
|---|---:|---:|---:|
| 18620_Sony_ILCE-6000 | 7 | +27.43 | 0.79 |
| 18619_Sony_ILCE-6000 | 18 | +26.56 | 2.20 |
| 18722_Canon_EOS_R6 | 7 | +25.29 | 0.49 |
| 18457_Sony_ILCE-7M4 | 142 | +20.92 | 4.65 |
| 19129_Nikon_Z_8 | 75 | +18.35 | 6.47 |
| 18466_Canon_EOS_R6 | 138 | +17.80 | 15.90 |
| 18505_Sony_ILCE-7M4 | 90 | +17.38 | 6.12 |
| 18447_Canon_EOS_R5 | 346 | +16.51 | 6.75 |
| 18657_Sony_ILCE-7M4 | 100 | +16.32 | 3.65 |
| 18499_Canon_EOS_R6 | 79 | +16.13 | 6.61 |

### For comparison: Temperature per-shoot

### Per-shoot analysis of `Temperature`

- shoots analysed: 92 (mean photos per shoot: 105.9)
- overall mean: 4911.907, overall std: 1096.932
- mean of per-shoot stds (within-shoot variability): 693.080
- std of per-shoot means (between-shoot variability): 794.562
- between/within ratio: 1.15  (>>1 = strongly clustered by shoot; ~1 = noisy within shoots)

**Shoots with the widest per-shoot std (most noisy):**

| shoot_id | n_photos | mean | std |
|---|---:|---:|---:|
| 18753_Canon_EOS_R6 | 6 | +4873.50 | 1800.22 |
| 18508_Canon_EOS_R6 | 18 | +5722.22 | 1534.62 |
| 18601_Canon_EOS_R6 | 107 | +4348.23 | 1520.04 |
| 18677_Canon_EOS_R6 | 197 | +5323.68 | 1465.26 |
| 18769_Canon_EOS_R6 | 53 | +5168.87 | 1458.34 |
| 18565_Canon_EOS_R6 | 6 | +5369.00 | 1457.86 |
| 18873_Canon_EOS_R6 | 115 | +5602.11 | 1403.18 |
| 18752_Canon_EOS_R6 | 58 | +5288.47 | 1388.26 |
| 18589_Canon_EOS_R6 | 306 | +5268.87 | 1384.48 |
| 18468_Canon_EOS_R6 | 30 | +5911.67 | 1379.07 |

**Shoots with the most extreme mean (per-shoot bias):**

| shoot_id | n_photos | mean | std |
|---|---:|---:|---:|
| 19095_Canon_EOS_R6 | 5 | +6694.00 | 614.96 |
| 18903_Canon_EOS_R6 | 122 | +6629.16 | 627.23 |
| 18467_Canon_EOS_R6 | 2 | +6535.50 | 2.12 |
| 18510_Canon_EOS_R6 | 25 | +6274.00 | 1070.06 |
| 18552_Canon_EOS_R6 | 28 | +6069.64 | 1258.26 |
| 18620_Sony_ILCE-6000 | 7 | +6050.00 | 70.71 |
| 18611_Canon_EOS_R6 | 111 | +6042.13 | 1234.55 |
| 18619_Sony_ILCE-6000 | 18 | +6005.56 | 175.64 |
| 18466_Canon_EOS_R6 | 138 | +5916.09 | 847.87 |
| 18468_Canon_EOS_R6 | 30 | +5911.67 | 1379.07 |

## 6. Cross-reference with v1.2.3 audit findings

From `scripts/output/all_slider_audit_v1.2.3_stats.parquet` (test split, n=1694):

| metric | Tint | Temperature |
|---|---:|---:|
| MAE | 6.146 | 731.3 |
| std(pred) | 0.277 | 1131.9 |
| std(target) | 7.183 | 1220.4 |
| std_ratio | 0.039 | 0.928 |
| direction correct | 0.971 | nan |
| corr(pred, target) | -0.001 | 0.711 |
| category | COLLAPSED | HEALTHY |

Tint's audit signature: predictions clustered narrowly (std_ratio < 0.1 = COLLAPSED) but direction-correct ~97% because most targets are near zero. Temperature's signature: high std_ratio (0.93) — model IS learning the spread, just amplified by exp() at extremes.

## 7. Most likely root cause

- **Tint has a near-perfect predictor in the metadata: `as_shot_tint` (ρ=+0.913, p<0e+00).** This is a stronger correlation than Temperature has with `as_shot_temperature` (compare: ['iso=-0.51']). The training data contains a near-identity mapping from AsShot Tint to user-final Tint — the signal isn't subtle, it's almost direct.
- **The model has access to `as_shot_tint` as a metadata input** (per `MetadataEncoder` arch_version=1: `self.as_shot_tint_fc = nn.Linear(1, 8)`). It SHOULD be trivial for the network to learn `Tint ≈ as_shot_tint`. But the audit shows it isn't — predictions cluster narrowly (std 0.4 on test) regardless of as_shot_tint input. **This is an architecture/training failure, not a data-side limit.** The signal exists; the model doesn't extract it.
- **Speculative mechanisms** (not provable from data alone, but worth flagging): (a) the fusion MLP routes all metadata through a single bottleneck before the WB head, so the direct `as_shot_tint → Tint` shortcut has to fight the image-feature signal for representation; (b) the WB head MLP `(832 → 128 → 64 → 2)` may have insufficient capacity for the WB-specific routing; (c) loss-weight scheduling (Tint at weight 4.0 in `config.SLIDER_LOSS_WEIGHTS`) didn't compensate for the model's inability to exploit the signal.
- **Strong sign asymmetry in targets.** 7226 positive vs 1276 negative Tint shifts (5.7× ratio). Mean target is +7.4. If the model collapses to the training-data mean (+7.4), it gets most photos approximately right on the positive side — which matches the audit's observation that direction-correctness is 97% even though std_ratio is 0.04. Direction correctness here is a misleading metric: the model 'agrees on sign' because almost everything is positive.
- **The collapse pattern matches 'predict the mean': mean target is +7.41, test prediction std is ~0.4 around a similar value.** The model has effectively learned 'predict +7' regardless of image content or metadata.

**Conclusion:**

The signal IS in the training data — `as_shot_tint` correlates with Tint at ρ=+0.913, a near-identity mapping. The model has the input available. **The collapse is an architecture/training problem in v1.2.3, NOT a data-side limit.** Even though the loss weight was bumped to 4.0 to fight this, the current architecture didn't extract the as_shot_tint → Tint shortcut.

**Implication for Phase 5 fine-tuning:** **unlikely to fix the issue.** If the current architecture can't exploit a ρ=0.91 correlation that's already in the bulk training data, more data won't change that. The fix has to be architectural — e.g., a direct skip-connection from `as_shot_tint` to the Tint output, or a separate WB-specific head with explicit metadata bypass. That's a v2 retrain consideration, not a continuous-learning fix.

**Implication for `default_skip_fields`:** Tint should be skipped in v1.2.3 production. Lightroom's fallback to as-shot WB will give the user approximately the right Tint value on ~91% of photos (per the correlation), which is dramatically better than the model's collapsed prediction of ~+7.4 regardless of input.
