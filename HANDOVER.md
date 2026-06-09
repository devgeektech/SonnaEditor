# Sonna Editor — Master Handover Document

**Project:** Internal AI photo editing tool for Sonna Studios
**Owner:** Darshil (Founder/Director, Sonna Studios)
**Platforms:** macOS, Windows, and Linux
**Reference hardware:** M1 Pro MacBook Pro, 32GB RAM
**Status:** v1.2.0 production shipped; v2 training prep active on the current Windows CUDA workstation; Phase 8 (team distribution) deferred
**Last updated:** 2026-06-05

---

## Current workspace state (2026-06-05)

This checkout is a Windows development/training workspace at `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`.

- **Environment:** Python 3.11.15 via uv 0.11.17. PyTorch is installed as `2.11.0+cu128`; `sonna_editor.runtime.preferred_torch_device()` returns `cuda` on the local NVIDIA GeForce RTX 3050. `scripts/verify_environment.py` passes 11/11 checks.
- **CUDA packaging:** `pyproject.toml` and `uv.lock` now pin `torch` and `torchvision` to the PyTorch CUDA 12.8 wheel index on Windows/Linux x86_64 so `uv sync --extra dev` does not fall back to CPU-only wheels.
- **Local dataset/checkpoints:** training/profile caches were intentionally cleared for a fresh dataset reset, and previous trained profile/foundation checkpoint files were cleared again before the new FiveK foundation run. The duplicate generated dataset folder `v1_learning\dataset` was removed. `v1_learning\` has no trained profile checkpoint and is reserved for frontend-visible published profile checkpoint/sidecar files only. `SonnaEditorFoundation\checkpoints\` is empty, and `foundation_manifest.json` is reset to an empty schema-v2 manifest. The next successful foundation training run will promote a new checkpoint as the default base for Personal AI / Mode A and Lite / Mode B.
- **Runtime layout:** the app now auto-creates repo-local working directories on startup, including `data\training_sources\`, `data\raw\`, `data\raw\sonna_training\`, `v1_learning\`, and `.saha\`. A fresh clone no longer depends on pre-existing gitignored folders or user-home `~\.saha` state.
- **Production profile UX boundary:** the frontend now exposes two profile creation paths: Personal AI from RAW+XMP warm-started from the hidden foundation checkpoint, and Lite from preset plus the six-question survey. Foundation model training is CLI-only with `scripts/train_foundation_model.py`.
- **Foundation/Lite boundary:** Lite profile creation no longer depends on the currently active Personal AI profile. It resolves the active foundation checkpoint from `SONNA_FOUNDATION_CHECKPOINT`, `SONNA_FOUNDATION_REPO/foundation_manifest.json`, or `SONNA_FOUNDATION_REPO/foundation.ckpt`; the default foundation folder is now the repo-local child folder `SonnaEditorFoundation\` unless overridden. The parent repo tracks this folder, with checkpoint binaries routed through Git LFS. The foundation CLI supports real Lightroom-label training from RAW+XMP folders or trusted catalog splits. The promoted foundation checkpoint is cumulative through native `SonnaEditor` slider-regression warm starts.
- **Foundation versioning:** every foundation run warm-starts from the active foundation checkpoint by default, writes a new versioned checkpoint, and promotes that new checkpoint as active. The manifest is now schema-v2 with `active_version`, `versions[]`, checkpoint SHA256, foundation type, capabilities, and training-source tags. Older foundation checkpoints are never overwritten; use `scripts\rollback_foundation.py` to activate a previous version. If the active checkpoint is removed, foundation resolution still falls back to the newest remaining checkpoint under `SonnaEditorFoundation\checkpoints\`.
- **Lite low-light exposure:** the preset/Lite exposure adjuster now gives low-light frames a stronger positive exposure lift when mean/median luminance are dark and highlights are not near clipping, matching the desired Imagen-like behavior more closely while keeping WB conservative unless explicitly enabled.
- **Training data location:** source RAW+XMP folders should live in separate child folders under repo-local `data\training_sources\` by default, or anywhere else you point the CLI/UI. Generated datasets, splits, thumbnails, audits, and run workspaces belong under `data\training_workspace\`. The repo-local `data\` folder remains gitignored and is auto-created as the local working area for source learning photos, generated datasets, captures, and foundation run workspaces. Promoted hidden foundation checkpoints live in the repo-local `SonnaEditorFoundation\` folder by default.
- **Foundation RAW+XMP docs:** `FOUNDATION_TRAINING.md` now spells out the RAW+XMP foundation prep path with direct script commands only: Lightroom sidecar export, source folder expectation, inspectable dataset/split build, audits, training from prepared splits, and direct `--raw-xmp-dir` shortcut.
- **FiveK local download and catalog review:** on 2026-06-05 the local FiveK folder at `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset` was inspected and smoke-tested. It contains the 5,000 DNG inputs and `raw_photos\fivek.lrcat`. The Lightroom catalog contains 60,000 catalog image rows over those 5,000 DNGs, with expert collections A/B/C/D/E at 5,000 rows each. A 20-row Expert C build into `data\training_workspace\fivek_catalog_verify_20260605` succeeded with 0 missing files and 0 parse errors. The catalog dataset builder supports `--collection-name "C"` plus `--include-unedited-looking` for the supported FiveK Expert C foundation route.
- **Foundation result triage and active rollback:** the `foundation-sonna-raw-xmp-001` continuation is not a good active foundation. It trained from only 132/27/30 Sonna RAW+XMP rows while 16.2M params were trainable, overfit (`test_loss / best_val_loss = 1.616x`), and failed key tone/color metrics. Collapse audit found collapsed `Highlights2012` and `Shadows2012`. The active manifest was rolled back to `foundation-fivek-catalog-expert-c-001`, whose 200-photo collapse audit found 0 collapsed sliders, though Saturation/Vibrance and Temperature still need visual review before treating it as production-quality direct Mode A output.
- **Foundation promotion guardrails:** `scripts\train_foundation_model.py` now blocks normal foundation training/promotion from fewer than 75 train rows unless `--allow-small-foundation-dataset` is explicitly supplied. It also blocks promotion when held-out quality metrics fail the gate (`test_loss` overfit ratio plus key MAE limits, using all-slider `test_per_field_mae` as a fallback) unless `--allow-quality-gate-failure` is supplied after deliberate visual review. Future `training_summary.json` files now embed split row counts, parquet paths, train-batch count, and all-slider `test_per_field_mae`; `scripts\quick_diagnostic.py` prints the all-parameter MAE check when that payload exists.
- **Foundation CUDA OOM handling and trainable capacity:** `scripts\train_foundation_model.py` defaults foundation runs to batch size 8 and automatically retries RAW+XMP slider-regression foundation training with smaller batch sizes after CUDA memory failures. Foundation startup uses adaptive capacity: catalog-scale/default runs start with `--backbone-trainable-layers stage:7`, while small splits below 500 train rows automatically use `--backbone-unfreeze-strategy custom --backbone-trainable-layers none` unless explicit backbone flags are supplied. Measured v2 trainable presets: `none` 1.92M, `block:7:2,stage:6` 7.86M, `block:7:1-2,stage:6` 12.63M, `stage:7` 16.21M. A one-epoch smoke on the local RTX 3050 previously passed at batch size 8 before this capacity change.
- **Training quality guard fixes:** warm-started profile/foundation runs now recalibrate output-head final biases from the current train split while preserving learned final weights, preventing stale foundation output priors from dominating small continuation datasets. The Lightroom catalog dataset builder now correctly honors its default unedited-looking filter; FiveK still disables that filter explicitly with `--include-unedited-looking`.
- **Dataset audit plotting:** `matplotlib` is now a base dependency so `scripts\audit_catalog.py` generates ISO and slider-distribution plots without optional-import warnings. The audit CLI prints ASCII statuses on Windows to avoid PowerShell emoji encoding errors.
- **Training runner structure:** the production training callable lives in `src/sonna_editor/training/profile_runner.py`; `scripts/train_profile.py` is a thin CLI wrapper. The API uses the packaged runner for frontend Personal AI training jobs, resolving the hidden foundation checkpoint as the warm start, with epoch progress, cancellation, and no publish on cancel. Personal AI sidecars record foundation provenance (`foundation_version`, checkpoint path, SHA256, type, capabilities, and source tags). Training startup logs and `training_summary.json` now include total/trainable/frozen parameters, trainable percentage, dataset row counts, batches per epoch, estimated optimizer steps, effective learning rates, sampler/cap status, and the backbone freeze/unfreeze summary.
- **Frontend profile deletion:** deleting a profile from the UI now asks for confirmation before removing local checkpoint/sidecar files. Active profile deletion remains blocked server-side.
- **Lite profile creation and processing:** fixed on 2026-06-02 for Imagen-aligned Lite output from the UI/CLI. `mode_b/checkpoint_builder.py` inherits the base checkpoint's native `slider_set_version` and writes a `mode_b_initial` sidecar. `inference/pipeline.py` now detects that sidecar for initial Lite runs, reloads the copied preset+survey, keeps preset look sliders fixed, applies per-photo Exposure/WB corrections only, and records those adjusted values in `sonna_predictions.json`. This prevents the active v2 base model's own Exposure/colour predictions from stacking on top of the uploaded preset while avoiding a constant bias-only preset clone.
- **Training script state:** `scripts/train_profile.py` uses the v2 recipe defaults (512px, fresh `arch_version=3`, WB metadata skip enabled, visual-priority weights: Exposure 5.0, Temperature/Tint 4.0, Contrast/Highlights/Shadows 3.0, Whites/Blacks/Saturation/Vibrance 2.0). Fresh models initialise output-head biases from training target medians, WB residual heads start at zero when AsShot WB skip is enabled, and default augmentation is geometry-only to avoid corrupting Exposure/WB labels. Fresh `arch_version=3` models consume six preview-derived luminance scene stats and use staged output-head conditioning; older checkpoints load with their saved architecture version. Default recipe values log as `Training recipe ...`; only explicitly supplied CLI flags log as `Override ...`.
- **Anti-collapse and retention diagnostics:** `scripts/analyse_prediction_collapse.py` reports per-slider prediction/target spread and collapsed sliders. `scripts/analyse_backbone_drift.py` compares ConvNeXt `backbone_features` tensors between a foundation checkpoint and a trained Personal AI checkpoint so FiveK/RAW+XMP feature retention can be measured instead of guessed. `scripts/audit_dataset_diversity.py` reports scene/edit diversity buckets. On the previous 27-photo validation split, existing `model-v2.0.0` showed 14 collapsed sliders and Exposure2012 std_ratio ~0.115; the rejected scene-stats candidate showed 29 collapsed sliders and near-zero Exposure spread despite lower test MAE. The previous diagnostic dataset was only 189 photos / 35 shoots, with 16 bright scenes and 9 cool-WB scenes. Those local dataset/checkpoint caches were later cleared for a fresh reset.
- **Dark low-light exposure failure:** Diagnosis on `0H5A4599` showed the expected/training XMP had `Exposure2012=+1.11`, while the then-active `model-v2.0.0` wrote about `+0.105`. Other key tone/WB sliders and curves were close, so the root cause was not XMP writing or tone-curve endpoints; it was Exposure2012 prediction collapse. Across the previous 189-row dataset, target Exposure std was ~0.454 but model output std was ~0.061, and the darkest luminance quartile needed ~`+0.695` while the model predicted ~`+0.090`.
- **Inference colour-cast fix:** `src/sonna_editor/inference/pipeline.py` now stabilises RGB tone-curve endpoints before writing XMP. Diagnosis from `0H5A3190A-2.xmp`: WB/Tint were close to the expected output, but Green/Blue tone-curve white endpoints below `255/255` made neutral whites render pink/red in Lightroom. The pipeline preserves RGB black endpoints at `0/0` and white endpoints at `255/255` while leaving middle curve points model-driven.
- **Verified this pass:** the latest focused training-capacity/diagnostics check passed with `uv run ruff check src\sonna_editor\model\architecture.py src\sonna_editor\training\module.py src\sonna_editor\training\profile_runner.py src\sonna_editor\training\diagnostics.py scripts\train_foundation_model.py tests\test_training.py tests\test_train_foundation_model.py` and focused pytest for the new freeze/diagnostic/foundation defaults (`10 passed`). Earlier full cleanup verification also passed `uv run ruff check .`, `uv run python -m compileall -q src scripts tests`, `npm run build:vite`, focused foundation/profile tests, focused mypy for the foundation CLI/helpers, and split full-suite pytest. Current split pytest result from that pass: `653 passed, 11 skipped` for the suite excluding `test_extract.py`/`test_xmp.py`, plus `71 passed, 34 skipped` for those fixture-dependent RAW/XMP tests. Private RAW/XMP fixture tests now skip when local fixtures are absent or unreadable.

---

## Current shipping state (2026-05-14)

**Active production model:** `DP Event v1.2.0 (full production, 12.9K, 256px)`
- Saha profile id: `dp-event-v1.2.3`
- Checkpoint: `v1_learning/model-v1.2.3-prod256.ckpt` (registered copy of `model-v1.2.0-full-production.ckpt`)
- Sidecar: `v1_learning/model-v1.2.3-prod256.json` (display name override + `default_skip_fields=["ColorGradeMidtoneHue", "SplitToningShadowHue", "Tint"]` + resolution=256). Sidecar is gitignored; canonical record of the field list lives in HANDOVER Part 6 item 14.
- Trained on 9,746 photos / 30 epochs max (EarlyStopping at e13, best e5), 256px input, ~111 min wall time
- Code state: Mode B Rebuild track complete (2026-05-14). Latest shipped includes the Mode B Step 3 validation commits (`b6b2d1e` predictions-sidecar schema, `7c93906` Mode B ckpt sidecar resolution) and the HANDOVER track-complete update for Mode B. Recent shipped changes since v1.2.3 production: Temperature epistemic clamp (`55aa70b`), Mode A WB AsShot substitution (`dac535e`), default_skip_fields expansion (`97194cf`), Mode B rebuild Steps 1-3 (`71fcf2b`, `a8d7c04`, `b6b2d1e`, `7c93906`), the slider-set shape-mismatch refactor for legacy compatibility, and the 2026-05-28 training-pipeline fix: current models now default to a direct AsShot Temperature/Tint WB metadata skip, training targets are slider-set-version aware, colour jitter is deliberately mild, and `scripts/train_profile.py` exports the best validation checkpoint as the native model.

**Production data pipeline:**
- Stratified BY-SHOOT splits at `data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/{train,val,test}.parquet` (75/11/14 photo proportions, 5-bucket quantile stratification on shoot_mean_delta_temp, seed=42). Old uniform splits at `data/training_workspace/sonna_personal_001_dataset/splits/` are deprecated; do not use for new training.
- 256px input resolution is locked for v1 production (validated in real-photo testing; 384 was tried in v1.2.0 smoke comparison and offered no quality improvement at the cost of 2.25× training time).

**UI toggle infrastructure:** generic `skip_fields` mechanism (commit `67062f2`) — sliders the user wants to omit from XMP writes are configured per-profile via the sidecar JSON's `default_skip_fields` list and per-job via the `[x] Skip <field>` checkboxes in the Output panel. Tint is the first slider exposed via this mechanism (default ON for v1.2.x profiles); the architecture is data-driven so adding more skip toggles is one entry in `SKIP_TOGGLE_OPTIONS`.

**Known v1.2.3 limitations (per the all-slider audit + three-way comparison + Tint deep dive — see Part 7 "Diagnostic reports"):**

- **Temperature: 731 K MAE** on the v1.2.3 held-out test set. Root cause: in-range mean-attraction bias on a bimodal target distribution (warm-light cluster ~3000–3500K + daylight cluster ~5000–6000K). The model DOES learn the log-space spread (std_ratio 0.93, Pearson 0.71) but the `exp()` postprocess amplifies in-distribution prediction error; not an extrapolation problem. **The epistemic clamp (commit `55aa70b`) was tested and confirmed to NOT reduce MAE** — it bounds extrapolation only (fires on 0.06% of test predictions). The v1.3 delta-from-AsShot experiment was an earlier attempt; failed in real-photo testing (uniformly over-cooled) and was reverted. **Do not retry the delta approach without a fundamentally different design.** 2026-05-28 update: the next v2 training run now uses the WB metadata skip-connection by default, so Temperature should be re-audited after training rather than accepted as a permanent v2 limit.

- **Tint: collapsed despite a near-perfect predictor available.** Audit categorisation: COLLAPSED (std_ratio 0.04, MAE 6.15). Tint targets in training have ρ=0.913 correlation with `as_shot_tint` — a near-identity mapping. The input is wired (`MetadataEncoder.as_shot_tint_fc`) but v1.2.3's fusion-MLP fails to extract it. **This is an architecture failure, not a data limit** — Phase 5 fine-tuning on similar data will not help. Mitigation for legacy v1.2.x: Tint is in `default_skip_fields`; Mode A's WB substitution (commit `dac535e`) writes the AsShot value explicitly to `crs:Tint`. 2026-05-28 update: new v2 models include an identity-initialised direct AsShot Tint route into the WB head.

- **SplitToningShadowHue + ColorGradeMidtoneHue: collapsed.** Both audit-categorised COLLAPSED. ColorGradeMidtoneHue is ALSO collapsed in Imagen Personal per the three-way comparison — **suggests a fundamental data limit, not Saha-specific** (the slider value may genuinely not be predictable from image content). Both added to `default_skip_fields` in this audit track.

- **Tone curves: per-control-point collapse, not per-channel.** The v1.0.1-era framing "all 4 channels predict near-identity" is disproven. Composite channel: predicted L2-deviation from identity is 83% of target deviation (partial learning, slight undershoot). R/G/B channels: predicted deviation actually *exceeds* target deviation (~114–118% — overshoot, predicting non-identity but in wrong directions). The collapse is concentrated on **lower-tone control points (Pt1–Pt3 Y values)** across multiple channels; Pt4–Pt5 partially learn. v2 fix scope narrower than originally planned — focus on lower-tone routing, not the whole curve head. See Part 6 item 9.

- **Imagen Personal (Imagen full Personal AI Profile) shows similar collapse patterns on different sliders** — three-way comparison found 73 fixed-value sliders in Imagen output across 117 photos. This is consistent with "slider regression model collapse on sparse-variance or metadata-bottlenecked targets is fundamental to the approach, not a Saha-specific bug." See Part 4 risk section.

**Active checkpoints / fallback tags:**
- `v1.2.0-production-shipped` ← current HEAD, the shipping point
- `checkpoint-pre-v1.3-delta-prediction` — same commit; kept for reference
- `checkpoint-pre-v1.2.1-stratified` — pre stratified-split rework
- `checkpoint-pre-v1.3-experiment-a` — pre temp_bucket bump (failed experiment)

---

## Model versioning

Trained model checkpoints follow `model-v{MAJOR}.{MINOR}.{ATTEMPT}[-{suffix}].ckpt`:

- **v1.0.0** — pilot smoke test (14K photos, 15 epochs). Confirmed model learns style.
- **v1.0.1** — first shippable v1 (12.9K photos, 20 epochs).
- **v1.1.0** — added camera make/model split, log focal/aperture, AsShot WB metadata, 512px resolution.
- **v1.2.0** — current production lineage. Recalibrated callbacks, deadband fix, OvercorrectionWarningCallback (warning-only). 256px replaces 512 as production resolution.
- **v1.2.0 256 smoke** (`dp-event-v1.2.0`) — first 3K validation; passed real-photo testing.
- **v1.2.0 384 smoke** (`dp-event-v1.2.1`) — 384px comparison; not better than 256, archived.
- **v1.2.1 stratified** (`dp-event-v1.2.2`) — same v1.2.0 model on the new stratified splits.
- **v1.2.0 full production** (`dp-event-v1.2.3`) — **current ACTIVE shipping model**, 9.7K full stratified train.
- **v1.3.0 (delta-AsShot)** — attempted experiment; failed real-photo test (over-cooled). REVERTED. Tag `checkpoint-pre-v1.3-delta-prediction` is the revert point. **Do not retry this approach.**
- **v2** — reserved for genuinely new features. Core bundle (now refined by v1.2.3 audit findings): (a) **WB head metadata skip-connection** to fix Tint architectural bottleneck and likely Temperature too (item 15, planned next-track experiment); (b) **lower-tone-curve-point routing fix** — collapse is on Pt1–Pt3 Y values specifically, not whole-channel identity (item 9 rewritten); (c) variable-length curves (6→8–12 pts); (d) Perspective/lens field handling; (e) local adjustments / AI Masks. Plus the 12 v2 extension slider fields already in `SLIDER_FIELDS` (idx 135–146). Do not use v2 label for incremental training runs.

---

## Build status

| Phase | Status | Notes |
|---|---|---|
| 0 | DONE | Environment verified, repo pushed |
| 1 | DONE | Tasks 1.1-1.6: XMP, DNG, extract, catalog reader, dataset builder, auditor |
| 2 | DONE (legacy rule-based pipeline) | Mode B preset pipeline: parser, adjuster, end-to-end CLI. **Superseded by Mode B rebuild track** (model-based Mode B). Remains available as fallback CLI tool. |
| 3 Task 3.1 | DONE | SonnaEditor architecture — ConvNeXt-Tiny backbone, 13 output heads, 135 sliders |
| 3 Task 3.2 | DONE | WeightedSliderLoss + augmentation (mode-aware resolution, accepts `resolution` arg) |
| 3 Task 3.3 | DONE | Training pipeline (datamodule + Lightning module + train scripts) |
| 3 Task 3.4 | DONE | v1.2.0 12.9K production trained, validated, shipped (id=`dp-event-v1.2.3`) |
| 3 (data) | DONE | Stratified BY-SHOOT split rework at `splits_v2_stratified/` (replaces old skewed splits) |
| 4 | DONE | Inference engine + pipeline; per-profile resolution from sidecar; `extra_skip_fields` filter in XMP writer |
| 5 | DONE for Mode A; Mode B-ready, unexercised | Continuous learning capture; finetune retrain pipeline. **Mode B Step 3 (2026-05-14) extended `sonna_predictions.json` with `profile_type` / `profile_id` / `base_checkpoint` / `slider_set_version` (`b6b2d1e`) so Phase 5 capture can branch on profile type when computing delta baselines.** The code paths are ready for Mode B; no real Mode B fine-tune has been exercised end-to-end yet (Phase 5 only runs on demand and no Mode B profile has accumulated capture data). Re-validation when the first Mode B fine-tune happens. See Decision 4 + item 17. |
| 6 | DONE | Profile registry: sidecar-driven discovery (`display_name`, `resolution`, `default_skip_fields`) |
| 7 | DONE | PyQt6 → Electron Saha app shipped: ProfileSelect dropdown with display_name override, generic `SKIP_TOGGLE_OPTIONS` toggle list (Tint default ON for v1.2.x), live job streaming via WS |
| Mode B rebuild | DONE 2026-05-14 (all three steps shipped) | (a) **style survey** (`71fcf2b`); (b) **preset-to-checkpoint converter** (`a8d7c04`); (c) **Mode B inference path** validated on real shoot 2026-05-14 — uses the same `inference/engine.py` + `inference/pipeline.py` as Mode A; produces XMP output matching preset+survey baseline within tolerance for every non-skipped scalar across all test photos. Two follow-up fixes surfaced during validation: predictions-sidecar schema extension recording profile metadata (`b6b2d1e`, closes item 19) and Mode B ckpt sidecar resolution inheritance from base ckpt (`7c93906`). See item 17 for full details and "Mode B usage" in Part 7 for the operational commands. |
| 8 | Deferred | Team distribution (code signing, distribution mechanism) |

**v2 prep (track complete 2026-05-13):** SLIDER_FIELDS locked at 147 (135 v1 + 12 v2 extensions, idx 135–146), 5 extension heads scaffolded behind `slider_set_version` flag (default "v2" for new instantiations, "v1" for the v1.2.3 ckpt). Migration script `scripts/migrate_labels_to_v2.py` ready to backfill labels via XMP re-extraction. v1.2.3 ckpt loads cleanly as v1 (production safety verified). Retrain on the 147-label dataset is the planned next major work; the skip-connection experiment (item 15) is the gating decision before committing to a full retrain.

**Audit / postprocess track (2026-05-13 to 2026-05-14):** v1.2.3 all-slider audit produced; Temperature epistemic clamp shipped (`55aa70b`, near-zero firing rate confirmed); Mode A WB AsShot substitution shipped (`dac535e`); default_skip_fields expanded (`97194cf`). **Extended 2026-05-14** with the slider_set v1/v2 shape-mismatch refactor: postprocess hotfix (`4133e8a`), version-aware helpers (`24756c1`), losses.py migration (`31e82ca`), predictions_to_dict migration (`8350903`), finetune capture/delta/retrain migrations (`7c58139`), production-pipeline integration test (`28f3290`). See Part 6 items 14–15 + 18 + Part 7 "Diagnostic reports".

### Current environment record

- Python 3.11.15 via uv 0.11.17 in the current Windows workspace
- PyTorch 2.11.0+cu128, CUDA confirmed available and working on NVIDIA GeForce RTX 3050
- macOS MPS remains supported by runtime device selection on Apple Silicon machines
- All deps installed: torchvision, pytorch-lightning, rawpy, pillow, lxml, pandas, pyarrow, tqdm, pyqt6, scikit-learn, pytest, ruff, mypy
- Adobe DNG Converter is discovered via `SONNA_DNG_CONVERTER`, OS default paths, or PATH
- GitHub repo: https://github.com/darshilp16-byte/sonnaeditor (SSH, pushed)
- Claude Code settings: use the current local checkout as the project root; avoid hardcoded user-home paths
- `scripts/verify_environment.py` - 11/11 checks pass in this workspace

### Before starting each session

At the start of a new session, read `SESSION_STATE.md` alongside this document, `project_knowledge.md`, and `SONNA_EDITOR_BUILD_SPEC.md` for instant context. At the end of meaningful work, update `SESSION_STATE.md` and any other Markdown file whose facts changed.

---

## How to use this document

This is the master reference for the Sonna Editor build. It contains everything an AI assistant or developer needs to pick up the project mid-stream and continue with full context.

If you are an AI assistant being handed this document: read it fully before responding to any technical question. The reasoning behind decisions matters as much as the decisions themselves. When making new recommendations, stay consistent with the principles laid out here unless given explicit reason to revisit them.

If you are Darshil returning to this project: this is your single source of truth. Keep it open in a tab.

---

## Part 1: Context

### Who Sonna Studios is

Sonna Studios is a premium photography and videography production company based in New Zealand, operating across Australasia with expansion into Los Angeles and London. The studio handles commercial, editorial, and event work for premium commercial clients including international watch brands and outdoor media networks. The team consists of Darshil (founder/director, handles creative direction and most operations), casual crew Chad and Sean, and part-time operations support Erin.

Editing volume is meaningful enough that AI-assisted editing has clear value. The motivation for building internally rather than using off-the-shelf solutions is roughly: 30% cost savings, 30% IP/control over the editing model, 30% learning value, 10% privacy advantage for sensitive client work.

### What we're building

A local desktop application that:

1. **Trains a personalised AI editing profile from historical photos (Mode A)** on Sonna's existing edited photos (~3,000+ photos from Lightroom catalog) — produces a `SonnaEditor` checkpoint trained from scratch (or warm-started from a prior version)
2. **Initialises an alternative profile from a preset + style survey (Mode B)** when historical training photos aren't available — preset and survey values seed a `mode_b_initial` profile package, with backbone weights warm-loaded from the active Mode A/Personal AI base checkpoint for later fine-tuning. Before fine-tuning, initial Lite processing uses preset+survey style plus adaptive per-photo Exposure/WB/tone corrections; after fine-tuning, Mode B uses the same model architecture and slider-set version as its base checkpoint.
3. **Applies either profile to new shoots** by predicting Lightroom slider values from image content + metadata, outputting XMP sidecars that Lightroom Classic auto-detects
4. **Improves continuously** via the same Phase 5 fine-tuning loop for both modes — Mode A and Mode B share architecture and capture/retrain pipeline; the only difference between profile types is the starting checkpoint

The conceptual model is a local, owned, Sonna-specific AI editing profile. It may not match general-purpose tools at first, but after fine-tuning it outperforms them on Sonna's specific work, because it sees only Sonna data.

### Quality target

Honest target: indistinguishable from manual edits 80-90% of the time on Sonna-typical event shoots, with the photographer tweaking the remaining 10-20% in Lightroom. Even well-trained models need occasional human review on the outliers.

The fine-tuning loop is the long-term quality lever. After 6 months of usage with consistent tweak-and-retrain cycles, the model will be increasingly tuned to Sonna's specific work, because it's trained exclusively on what Sonna shoots.

### Scope explicitly excluded from v1

- AI culling (separate problem, separate model)
- Subject masking and local adjustments
- Crop/straighten suggestions
- HDR merge
- Lightroom plugins / direct integration
- Cloud processing (everything runs locally)
- Team distribution (Phase 8, deferred)

---

## Part 2: Architecture & technical reasoning

### Core architectural decisions and why

**Decision 1: Slider regression, not pixel generation**

The model predicts Lightroom slider values, not pixels. This is the correct approach because:
- Lightroom edits are non-destructive and parametric
- Output is a small XMP file, integrates cleanly with existing workflows
- The ML problem is regression on ~37 numbers, which is far more tractable than image-to-image generation
- User can override any prediction in Lightroom afterward

**Decision 2: DNG as internal normalised format**

All RAW formats (CR3, NEF, ARW, etc.) are converted to DNG via Adobe DNG Converter at ingestion. Reasons:
- One format to handle in code, not 15
- DNG can embed XMP edit metadata
- Adobe DNG Converter is free and command-line scriptable
- Eliminates per-camera-body decoder maintenance

**Decision 3: Local-only, no cloud**

For a four-person team with local GPU/CPU hardware available, local processing is faster and cheaper than cloud:
- 1,000-photo shoot processes in 1-3 minutes locally vs. upload + process + download time
- Zero ongoing costs
- Client photos never leave Sonna's network (real privacy advantage)
- No infrastructure to maintain
- No per-photo fees ever

The only reason to revisit this is if the team scales significantly or starts running many shoots in parallel.

**Decision 4: Two modes, both model-based (Mode A trained, Mode B preset/survey-initialised) — revised 2026-05-14**

Both Mode A and Mode B produce a `SonnaEditor` checkpoint/profile package and are launched through the same processing entry point. The difference is how the first output is produced.

**Mode A — trained from historical photos**
- Backbone (ConvNeXt-Tiny) and 13 output heads trained from scratch (or warm-started from prior `SonnaEditor` ckpt) on Sonna's existing Lightroom-edited photos
- Produces the v1.2.3 production lineage and its successors
- Quality scales with training-data volume and tweak feedback through Phase 5

**Mode B — initialised from preset + style survey**
- *Initial state* = a Lightroom preset (baseline slider values) + a six-question style survey (Exposure2012, Temperature, Tint, Contrast2012, Saturation, Shadows2012; v2 extension fields are inherited/calibrated from preset defaults when the foundation checkpoint is v2)
- A **preset-to-checkpoint converter** (new build piece — Mode B rebuild step b) ingests the preset + survey and produces a `SonnaEditor` ckpt:
  - Output-head final weights zeroed and final biases set to the preset/survey values for each slider, so the checkpoint carries the uploaded calibration
  - Backbone, metadata encoder, and hidden head layers warm-loaded from the configured foundation checkpoint so future fine-tuning starts from useful features
  - Result is a UI-selectable `mode_b_initial` profile with copied preset/survey metadata
- Before the first fine-tune, Mode B initial processing uses the adaptive preset branch in `inference/pipeline.py`: preset controls the look, while per-photo automation adjusts only Exposure, Temperature, and Tint.
- After fine-tuning, Mode B *is* a model — same architecture, same native slider-set version as the foundation checkpoint, same `inference/engine.py` prediction path as Mode A
- Mode B fine-tunes via the **same Phase 5 mechanism** as Mode A: capture user tweaks → compute deltas → retrain → new versioned ckpt

**No graduation mechanism.** Mode B does not "become" Mode A. They are two profile types with different initialisation strategies, both improving over time through the same Phase 5 loop. A Mode B profile after 6 months of Phase 5 retrains is still a Mode B profile; it just happens to have a much better checkpoint than its initial state.

**Why this matters for the build plan:**
- Phase 2's preset parser/adjuster remains in the repo and is now reused by the initial Mode B Lite processing branch. The standalone preset CLI is still a fallback/diagnostic tool; the production UI flow is the `mode_b_initial` profile package described above.
- The Mode B rebuild track (style survey → preset-to-checkpoint converter → Mode B inference path) is sequenced **before Phase 5 redesign** so the capture/finetune mechanism can be validated against both profile types from the start.
- The Saha app's profile registry already supports multiple profiles by sidecar — Mode B profiles will appear in the same ProfileSelect dropdown as Mode A profiles. No UI restructure needed.

**Original (now-superseded) Decision 4 framing:** Mode B was a rule-based preset pipeline that shipped first to validate the input→XMP→Lightroom path before adding ML complexity, with Mode A described as "swap preset values for predicted values." That framing is preserved in the Phase 2 row of the build status table for historical context; the production direction has moved on.

**Decision 5: Image + metadata as model inputs (not just image)**

We analyse photo metadata alongside image content:
- Image input: 384px thumbnail (extracted from RAW's embedded preview)
- Metadata input: ISO, shutter, aperture, lens, focal length, camera body, capture white balance, camera profile, computed RGB histogram
- Late fusion: image features (768-d) + metadata features (64-d) → output heads

This is critical for handling Sonna's varied event lighting. The same scene at ISO 100 vs ISO 6400 should be edited differently, and the model needs ISO context to learn that.

**Decision 6: Slider-list-driven architecture**

All 135 sliders are configured in `config.py`, not hardcoded throughout. When Adobe adds new sliders or restructures the develop module, adapting requires:
1. Add slider name to config (one line)
2. Re-extract training data including the new field (one script run)
3. Retrain the model with new output dimension (one training run)

No code refactor. This is a deliberate design choice for long-term maintainability.

The v1 slider list covers 135 continuous sliders across all major Lightroom develop panels, including 48 point tone curve fields (4 channels × 6 control points × X/Y coordinates). Tone curves are stored in XMP as rdf:Seq child elements and normalised to 6 fixed control points on extraction. Deferred to v2: LensProfileEnable (binary flag, not continuous).

**Locked-append-only rule (added 2026-05-13, v2 prep).** Once a slider index is assigned in SLIDER_FIELDS, that index is frozen forever. Indices 0–134 ship with v1.2.3 and will never be reordered, repurposed, or removed in any future version. New fields are appended at the end of SLIDER_FIELDS only — never inserted, never interleaved.

This rule preserves checkpoint compatibility: any older model can be loaded into a newer architecture because the first N output indices (where N = older model's output count) retain identical meaning. New fields beyond N are predicted by extension heads that older checkpoints don't have weights for; those heads start random-init and learn from scratch during the next retrain (or warm-start fine-tune from the older base — see warm-start retrain decision).

The rule also gates the migration script: when SLIDER_FIELDS grows, existing parquet labels are extended by **re-extracting the new fields from source XMPs**, not by zero-filling. Documented Lightroom defaults (`config.SLIDER_DEFAULTS`) serve as the fallback for any row where re-extraction fails or the source XMP is missing.

**`slider_set_version` gate (added 2026-05-13).** The locked-append-only rule is implemented as a `slider_set_version` flag on `SonnaEditor.__init__`. `"v1"` builds 13 heads / 135 outputs (matches v1.2.3 shipping ckpt). `"v2"` builds 18 heads / 147 outputs (13 v1 + 5 extension heads). New instantiations default to `"v2"`. `save_checkpoint` persists the flag into arch_config so `from_checkpoint` can re-instantiate at the correct version. Cross-version load is one-way: v1 → v2 warm-start (strict=False, extension heads random-init) is supported; v2 → v1 raises ValueError.

**Decision 7: One model file per profile, versioned not overwritten** *(unchanged — see below)*

**Decision 8: Neutral Learner principle**

The model and all surrounding code must encode zero style opinions. Concretely:
- `SLIDER_LOSS_WEIGHTS` = all 1.0. No field gets a higher weight because it's "more important."
- Range normalisation (dividing each slider by its valid range before MSE) is done inside `WeightedSliderLoss` (Task 3.2), not via weight inflation.
- Postprocess clamps use full Lightroom valid ranges, not narrower "sensible" ranges.
- The content-aware adjuster (legacy rule-based Mode B / CLI tool) uses only image-derived signals for exposure/shadow/highlight recovery; it does not encode style. The model-based Mode B (rebuild track) inherits the Neutral Learner principle automatically because it uses the same `SonnaEditor` architecture and loss as Mode A.
- Augmentation never touches target slider values.
- No defensive output caps. Do not artificially limit slider values within Lightroom's valid ranges based on assumptions about "typical" or "reasonable" correction amounts. Examples of what NOT to do: capping Exposure deltas to ±0.7 stops, capping Shadows lift to +60, requiring Highlights to stay above -50. Lightroom defines the valid range for each slider — that is the only constraint. Within that range, trust the input data (preset values for legacy Mode B, model predictions for Mode A and model-based Mode B) and write whatever value is computed.

Style only comes from training data. If the data is Sonna-style, the model will be Sonna-style — without any thumb on the scale from the engineering layer.

Temperature uses log-space in model prediction space: the model predicts log(Kelvin), `WeightedSliderLoss` normalises using `lo=log(2000), hi=log(50000)`, and postprocess applies `exp()` to convert back.

**Epistemic clamps (allowed) vs stylistic clamps (banned, added 2026-05-13).** The "no defensive output caps" rule bans **stylistic clamps** — bounds based on assumptions about what's "reasonable" within Lightroom's valid range. **Epistemic clamps** are distinct and allowed: bounds based on what the model has training evidence for. The first one lives in `inference/pipeline.py` as `TEMPERATURE_LOG_CLAMP`: log(2037) to log(9400), the actual min/max of Temperature in v1.2.3 training data. It prevents catastrophic `exp()` amplification on out-of-distribution predictions without making assumptions about Sonna's stylistic preferences. Verified firing rate on test split: 0.06% (1/1694 predictions). Future epistemic clamps must cite their training-data evidence (min/max of the relevant target column on the training split); they are not a backdoor for stylistic intervention.

**Decision 7: One model file per profile, versioned not overwritten**

Each profile (e.g., "Sonna Events") has a single model checkpoint file (~50-200MB). Fine-tuning produces a new version (`v2.ckpt`) rather than overwriting the original. The profile registry tracks which version is current. Old versions are preserved indefinitely so rollback is trivial. Total disk usage after years of use is still trivial (single-digit GB).

### How the data actually flows

**Training:**
```
Lightroom catalog or folder of edited RAWs
  → Adobe DNG Converter (normalise to DNG)
  → Extract: {config.IMAGE_RESOLUTION}px thumbnail JPEG + metadata + 135 slider values from XMP/catalog (v1.2.x ships at 256)
  → Save to Parquet dataset (~200-500MB for 3,000 photos)
  → PyTorch training loop on M1 GPU (3-5 hours)
  → Output: model checkpoint file (~50-200MB)
```

The original RAW files are never modified. The training thumbnails can be archived after training — the model file is self-contained.

**Inference (using the tool on new shoots):**
```
Folder of new RAW files
  → (optional) Convert to DNG
  → Extract thumbnail + metadata for each
  → Batch through model on M1 GPU
  → Get 135 predicted slider values per photo
  → Write XMP sidecar next to each RAW
  → User opens in Lightroom, edits are auto-detected and applied
```

The model loads once at app startup and stays in memory. Inference uses CUDA, Apple MPS, or CPU depending on host capability. A 1,000-photo shoot processes fastest on GPU-backed machines.

**Continuous learning (fine-tuning loop):**
```
User processes shoot with model → tweaks photos in Lightroom → saves XMPs
  → Capture script identifies tweaked photos and computes (predicted vs final) deltas
  → After ~200+ tweaked photos accumulated:
  → Fine-tuning script: combines original training data + new tweaked data (weighted higher)
  → Trains for 30 epochs at low learning rate (1-2 hours on M1)
  → Saves as new version (v2.ckpt)
  → Profile registry updates current version
  → If v2 metrics worse than v1, rollback is one command
```

### Model architecture details

```
Image (H×W RGB, where H=W=config.IMAGE_RESOLUTION; v1.2.x ships at 256)
  → ConvNeXt-Tiny (pretrained ImageNet) → 768-d image features

Metadata + histogram
  → MetadataEncoder (embeddings + MLP) → 64-d metadata features

Concat (832-d)
  → Tone head            →  8 outputs  (exposure, contrast, highlights, shadows, whites, blacks, clarity, dehaze)
  → Presence head        →  3 outputs  (texture, vibrance, saturation)
  → WB head              →  2 outputs  (log_temperature, tint)
  → HSL head             → 24 outputs  (8 colors × 3: hue, saturation, luminance)
  → Parametric head      →  7 outputs  (parametric curve + 3 split points)
  → Color Grading head   → 14 outputs  (shadow/midtone/highlight/global Hue+Sat+Lum + blending + balance)
  → Calibration head     →  6 outputs  (R/G/B hue + saturation)
  → Detail head          →  4 outputs  (sharpness, radius, detail, edge masking)
  → Noise head           →  4 outputs  (lum smoothing, lum detail, lum contrast, color NR)
  → Effects head         →  8 outputs  (vignette amount/midpoint/roundness/feather/highlight contrast + grain amount/size/frequency)
  → Lens head            →  2 outputs  (distortion, lens vignette)
  → Transform head       →  5 outputs  (vertical, horizontal, rotate, scale, aspect)
  → Tone Curve head      → 48 outputs  (4 channels × 6 control points × X/Y)

Total output: 135 floats → clamped to Lightroom valid ranges → written to XMP
Parameter count: ~29.4M (ConvNeXt-Tiny 27.8M + metadata encoder 0.04M + 13 heads ~1.6M)
Temperature (index 11) predicted in log-space; postprocess applies exp() before XMP write.
```

**v2 extension (added 2026-05-13).** SonnaEditor instances default to
`slider_set_version="v2"`, which builds 5 additional extension heads after
the 13 v1 heads above:

```
v2 extends the v1 layout with 5 extension heads appended to the forward concat:
  → Noise ext head       →  2 outputs  (ColorNoiseReductionDetail, ColorNoiseReductionSmoothness)
  → Defringe head        →  6 outputs  (Defringe Purple/Green × Amount/HueLo/HueHi)
  → Lens profile head    →  2 outputs  (LensProfileDistortionScale, LensProfileVignettingScale)
  → Calibration ext head →  1 output   (ShadowTint)
  → Curve ext head       →  1 output   (CurveRefineSaturation)

v2 total output: 147 floats (135 v1 + 12 v2 extension fields at idx 135-146)
v2 parameter overhead: ~0.2M additional (5 heads × ~50K each).
```

Loading v1 checkpoints (e.g. v1.2.3 production) into v2 uses
`SonnaEditor.from_checkpoint(path, target_slider_set_version="v2")` with
automatic `strict=False`. The 5 extension heads start random-init and learn
from scratch on the next retrain. v2→v1 loads raise to avoid silent
information loss from dropping the extension heads.

**Loss:** all SLIDER_LOSS_WEIGHTS = 1.0 (Neutral Learner principle — see Decision 8). Range normalisation is applied inside WeightedSliderLoss before MSE so equal fractional-range errors contribute equally across all 135 sliders regardless of range width. `_count_unedited` in audit.py checks only the first 87 scalar slider fields — tone curve identity zeros must not inflate unedited counts.

**Augmentation:** input image gets random brightness/contrast/color jitter; output sliders are never touched. This teaches the model that the same photo at different exposures should still be edited toward the same final look.

**Train/val/test split:** by SHOOT, not by photo. Photos within ~12 hours from the same camera body are the same shoot and must stay grouped to prevent data leakage.

### Always-on XMP postprocess rules (added 2026-05-13)

Two `crs:` attributes are written to every output XMP regardless of model
version or prediction values. NOT in `SLIDER_FIELDS` — they're binary toggles,
not regression targets.

- `crs:LensProfileEnable="1"`: Sonna always wants lens profile correction
  applied. Lightroom uses camera+lens metadata to apply the embedded
  correction profile (distortion, vignetting, chromatic aberration).
- `crs:AutoLateralCA="1"`: automatic lateral chromatic aberration removal.
  Absent from real LR Classic 15.3 XMPs even with lens profile enabled
  (verified 2026-05-13 via Canon R6 + RF24-70mm export); the historical
  `LensProfileChromaticAberrationScale` slider was a separate manual
  control that was removed from LR's Lens Profile sub-panel.

Implementation: `inference/pipeline.py` defines `ALWAYS_ON_POSTPROCESS`
and passes it to `write_xmp(extra_attributes=...)`. Applies universally
to v1.2.3 and v2 inference paths.

### Resolution roadmap

This is staged deliberately to balance training speed against quality.

**v1 — 384px input (initial training)**
- Fast iteration: 3-5 hours per training run on the reference machine
- Validates architecture, dataset quality, and end-to-end pipeline
- Sufficient for global tone, white balance, and HSL prediction
- Weaker on local-detail operations: Clarity, Texture, Dehaze

**v2 — 512px input (production target)**
- Meaningful quality improvement on local-detail sliders
- Training time roughly doubles to 6-10 hours
- Memory headroom comfortable on the reference machine at batch size 16
- This is the realistic production resolution for Sonna's use

**v3 — 768px input (stretch goal)**
- Maximum practical resolution on the reference machine; marginal quality gains likely diminishing at this point
- Training time on the reference machine: 12-20 hours per run (overnight runs)
- Memory: 18-22GB during training at batch size 16; may need batch size 8 + gradient accumulation
- Worth pursuing only after v2 has been in production use long enough to identify specific quality gaps

**Why staged, not straight to 768px:**

Going straight to 768px adds training friction (longer runs, tighter memory, more iteration overhead) without guaranteed proportional quality gain. The diminishing returns curve for regression tasks like ours flattens somewhere between 512 and 768. Better to validate at 384px, lock in the architecture and data pipeline at 512px, then make an informed decision about 768px based on real production weaknesses you've observed.

**Architecture supports all three:** input resolution is a single config value. Switching from 384 to 512 to 768 is a config change, dataset re-extraction (thumbnails regenerated at new resolution), and retrain. No code refactor.

---

## Part 3: Build plan

### Phases overview

| Phase | Deliverable | Realistic time (with Claude Code) |
|---|---|---|
| 0 | Working dev environment, M1 GPU verified | 1-2 hrs |
| 1 | Data extraction pipeline + dataset auditor | 6-10 hrs |
| 2 | Mode B (Lite preset) end-to-end working | 4-6 hrs |
| 3 | Model architecture + first trained Sonna profile | 8-12 hrs (+ 3-5 hrs training time) |
| 4 | Inference engine processing real shoots | 3-5 hrs |
| 5 | Continuous learning / fine-tuning loop | 4-6 hrs |
| 6 | Profile management & versioning | 2-3 hrs |
| 7 | Electron desktop UI | 10-15 hrs |
| 8 | Team distribution (deferred) | TBD |

**Total: ~40-60 hours of focused work.** At 10 hrs/week, ~4-6 weeks to a fully working system. Mode B usable by week 2.

### Detailed phase specifications

**Note:** the full task-by-task spec with explicit Claude Code prompts lives in `SONNA_EDITOR_BUILD_SPEC.md`, which should be in the project root. This handover is the higher-level reference; the build spec is the working document.

### Project structure

```
sonna-editor/
├── pyproject.toml              # uv-managed dependencies
├── README.md
├── SONNA_EDITOR_BUILD_SPEC.md  # detailed build spec
├── HANDOVER.md                 # this document
├── .gitignore
├── .python-version             # Python 3.11
│
├── src/sonna_editor/
│   ├── config.py               # paths, slider definitions, constants
│   ├── data/                   # Phase 1: catalog, dng, xmp, extract, dataset, audit
│   ├── preset/                 # Phase 2: parser, adjuster, pipeline
│   ├── model/                  # Phase 3: architecture, losses, augmentation
│   ├── training/               # Phase 3: datamodule, training module, train script
│   ├── inference/              # Phase 4: engine, confidence, pipeline
│   ├── finetune/               # Phase 5: capture, delta, retrain
│   ├── profiles/               # Phase 6: registry, manager
│   └── ui/                     # Legacy PyQt6 scaffold; Electron app lives in saha-app/
│
├── scripts/                    # CLI entrypoints
├── tests/                      # pytest tests
├── notebooks/                  # exploration
│
├── data/                       # gitignored
│   ├── raw/                    # source RAW + XMP
│   ├── dng/                    # normalised DNGs
│   ├── parquet/                # training datasets
│   ├── thumbnails/             # cached thumbnails
│   ├── captures/               # user tweak data for fine-tuning
│   └── audits/                 # auditor reports
│
└── models/                     # gitignored (sync to Dropbox/iCloud)
    ├── manifest.json           # profile registry
    ├── sonna_events_v1.ckpt
    ├── sonna_events_v2.ckpt
    └── ...
```

### Stack (locked)

- Python 3.11 via `uv`
- PyTorch 2.x with MPS backend
- PyTorch Lightning for training loops
- `rawpy` for RAW handling
- `lxml` for XMP read/write
- Pillow for image processing
- Pandas + PyArrow for Parquet datasets
- Electron + React for desktop UI
- pytest + ruff + mypy for dev quality
- Adobe DNG Converter (free, external)
- All other dependencies free and open source

### Costs

**Build phase:** NZD $0 (all free tools)
**Operational:** NZD $0/month (local M1, no cloud, no subscriptions)
**Phase 8 (team distribution, deferred):** ~NZD $165/year for Apple Developer Program if code-signing for Gatekeeper

---

## Part 4: Known risks and mitigations

### Risk: model collapse on sparse-variance or metadata-bottlenecked targets is fundamental to the approach

Empirical finding (v1.2.3 audit + three-way comparison vs Imagen Personal): trained slider-regression models will produce **collapsed** predictions (std_ratio near 0, predictions clustering near the target mean regardless of input) on a subset of fields, even with well-curated training data. This is not a Sonna-specific bug — Imagen Personal's three-way comparison output shows 73 fixed-value sliders across 117 photos, including `ColorGradeMidtoneHue` collapsed in BOTH trained models from independent lineages.

Two distinct mechanisms produce collapse:
- **Sparse signal** — when the target is genuinely not predictable from available inputs (e.g. `ColorGradeMidtoneHue` — both Saha and Imagen converge to ~constant; data may not constrain this slider).
- **Metadata-bottlenecked routing** — when the signal IS in the inputs but the model's fusion architecture doesn't route it cleanly (e.g. v1.2.3 Tint: ρ=0.913 with `as_shot_tint` available as input, but predictions still collapse to mean +7.4). Loss-weight bumping doesn't compensate.

**Mitigation:**
- Treat collapsed sliders as **expected, not a regression to debug into oblivion.** Run the all-slider audit (`scripts/audit_all_sliders_v1.2.3.py`) after each retrain. Categorise into HEALTHY / HIGH ERROR / COLLAPSED / WRONG DIRECTION / SPARSE TARGET.
- For collapsed sliders with available metadata signal: **architectural fix is the only viable path** (e.g. skip-connection from metadata to output head — see Part 6 item 15). Phase 5 fine-tuning on similar data will not help — the bottleneck is upstream of the loss.
- For collapsed sliders without learnable signal: **use `default_skip_fields`** to suppress predictions in production XMPs. Lightroom uses its own defaults; better outcome than committing to a constant.
- Document each collapse with its root-cause category in Part 6 item 14 so future retrain decisions are informed.

### Risk: Lightroom catalog reader pain

The .lrcat schema is partially undocumented and changes between Lightroom versions. Risk: Phase 1 task 1.4 takes longer than estimated.

**Mitigation:** The plan already includes a fallback path. If catalog reader proves painful, use Lightroom's built-in "Save Metadata to File" feature to dump XMP sidecars next to RAWs, then read the XMPs directly (well-documented format). This sidesteps catalog reverse-engineering entirely. The catalog reader becomes a productivity tool, not a critical path.

### Risk: MPS-specific PyTorch issues

Apple's MPS backend has occasional gaps. Some ops fall back to CPU, some have memory quirks, fp16 isn't fully supported.

**Mitigation:** Use fp32 ("32-true" precision) by default. If a specific op fails, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for that op. Document any workarounds in the codebase as they're discovered. Most issues are well-known and have community workarounds.

### Risk: First trained model doesn't hit quality targets

Quality targets:
- Median exposure error < 0.20 stops
- Median temperature error < 250K
- Median tint error < 5
- Median HSL error < 6 units

v1.2.3 actual results (audit-confirmed):
- Median temperature error: ~731 K (well above 250K target — in-range mean-attraction bias, not extrapolation; clamp tested and confirmed not to reduce MAE)
- Tint: collapsed (std_ratio 0.04, MAE 6.15) — architectural failure, not data
- ~50 sliders HIGH ERROR, ~11 COLLAPSED, ~11 WRONG DIRECTION, ~52 SPARSE TARGET (out of 135) per the all-slider audit

The original generic-ML causes (overfit, underfit, wrong loss weights) don't apply to v1.2.3's actual failure modes — most issues are **architectural routing problems** documented in the audit and three-way comparison (see Part 7 "Diagnostic reports").

**Mitigation (refined post-v1.2.3):**
- The dataset auditor (Phase 1.6) catches data issues before training.
- After training, run the **all-slider audit** (`scripts/audit_all_sliders_v1.2.3.py`) to categorise each slider's failure mode. Don't conflate categories — collapse vs wrong-direction vs high-error each need different interventions.
- For metadata-correlated targets that collapse: try the **skip-connection experiment** (item 15) before committing to a full retrain. Cheaper diagnostic, three-outcome decision tree.
- For sliders that collapse with no available signal: accept and add to `default_skip_fields`. Some sliders are not learnable; that's not a model bug.
- Multiple training iterations are still expected — budget 2-3 runs to hit quality, plus 1-2 architectural-experiment iterations to address routing issues.

### Risk: New Lightroom slider releases

Adobe periodically adds new sliders. Existing models won't predict them.

**Mitigation:** Slider list is config-driven, so adding new sliders is a 1-line config change + dataset re-extraction + retrain. Won't be predicted accurately until Sonna has been using the new slider in editing for several months (need training data showing how Sonna uses it).

### Risk: Model file backup/loss

A trained profile is real intellectual asset. Losing the model file means weeks of training data work is lost.

**Mitigation:** The `models/` directory should be synced to Dropbox, iCloud Drive, or Google Drive. Files are 50-200MB each, sync is fast and free. Optionally also push to a private GitHub repo using Git LFS.

### Risk: Camera profile mismatch between training and inference

If training data was edited under "Adobe Color" baseline and inference photos use "Camera Standard," predictions will be off because the baseline rendering differs.

**Mitigation:** XMP output specifies the camera profile the model expects. App warns if input photos use a different baseline profile than the training set. Long-term: consider separate profiles per camera profile.

### Risk: Reproducibility

PyTorch on MPS is not bit-reproducible across runs even with seeded RNG.

**Mitigation:** Document this. For our use case (production tool, not research), it doesn't matter — only the final model quality matters, not exact bit-equivalence. If reproducibility becomes important, train on CUDA in the cloud as a one-off.

### Risk: Continuous learning corrupts the model

If the user feeds back inconsistent or low-quality tweaks, fine-tuning could degrade the model.

**Mitigation:** Fine-tuning is always a new version, never overwriting. Before adopting v2, the system reports val loss vs v1 on a held-out test set. If worse, prompt the user before promoting. Old versions kept indefinitely so rollback is one command.

### Risk: Test coverage gaps on production inference paths

The v2 slider expansion (commit `3d0d90c`, 2026-05-13) silently broke v1.2.3 production inference for ~20 hours despite a full passing pytest suite — no test exercised a v1 checkpoint end-to-end through `InferenceEngine.predict()`, so the shape mismatch in `postprocess_predictions()` was invisible until Mode B rebuild Step 2 surfaced it during manual verification (see Part 6 item 18). Hotfixed in commit `4133e8a` with a regression test in `tests/test_postprocess.py`, but a SECOND latent shape bug (`predictions_to_dict` IndexError) was then exposed at the next pipeline step — the same root cause (147-length SLIDER_FIELDS paired with 135-length v1 model output) had multiple sites that the audit-and-refactor track (commits `24756c1` → `28f3290`, 2026-05-14) has since systematically migrated.

**Mitigation — CLOSED for the immediate gap:** `tests/test_inference_pipeline_integration.py` (commit `28f3290`) loads a synthetic v1 SonnaEditor and runs it through the full production pipeline — `engine.predict` → `predictions_to_dict` → `WeightedSliderLoss.forward` + `direction_stats` + `per_field_mae` → `write_xmp` → `read_xmp` roundtrip — for both v1 and v2 in single tests that run in ~7s. Every site patched in the refactor is exercised on a v1 checkpoint by this file. Future slider_set_version additions (e.g. v3) must extend this integration test in the same commit that introduces the new variant.

**Mitigation — PARTIAL for the broader structural risk:** the integration test catches v1/v2 shape mismatches end-to-end, but doesn't replace per-component coverage. Some inference paths (the API route in `src/sonna_editor/api/routes/process.py`, the Saha Electron app's IPC flow) are still only exercised at integration-test level via `process_shoot_with_model`. End-to-end tests against the real Saha app stack remain a manual-QA gap; the integration test is the automated floor, not the ceiling.

### Risk: Slider list expansion v1/v2 compatibility

Adding fields to `config.SLIDER_FIELDS` requires using version-aware helpers (`sonna_editor/slider_set.py`) at every site that pairs the field list with model output. Direct iteration over `SLIDER_FIELDS` (or `range(len(SLIDER_FIELDS))`, `enumerate(SLIDER_FIELDS)`) is prohibited in code that handles model output or buffers paired with model output. The 2026-05-14 audit found 7 hard-crash sites and 3 silent-degradation sites of this pattern after the v2 expansion (commit `3d0d90c`); the refactor commits `24756c1` → `28f3290` migrated all of them to the version-aware helpers.

**The three helpers and when to use each:**

- `v1_fields()` — locked 135-field v1 slice. Use when the caller is explicitly targeting v1 (e.g. v1-only audit scripts, Mode B preset-to-checkpoint converter).
- `fields_for_version(slider_set_version)` — slice for the given version. Use when a model or training context exposes `slider_set_version` directly (loss objects, training/fine-tune pipelines that know the loaded model's version).
- `fields_matching_tensor(tensor)` — slice matching the tensor's last dim. Use when a function receives a prediction tensor without separate version info (postprocess, predictions_to_dict, generic diagnostics).

**Class-of-bug detection** is enforced by `tests/test_inference_pipeline_integration.py` (both v1 and v2 paths covered) plus per-site unit tests in `tests/test_slider_set.py`, `tests/test_postprocess.py`, `tests/test_losses.py` (v1 coverage section), `tests/test_finetune_*.py`.

**Mitigation:** Future SLIDER_FIELDS extensions (a hypothetical v3, or any reordering — which would also violate the locked-append-only rule from Decision 6) must:
1. Register the new length in `slider_set.py:_SUPPORTED_LENGTHS` and version mapping
2. Audit any new direct iterations of `SLIDER_FIELDS` in code review; reject in favour of helpers
3. Extend `test_inference_pipeline_integration.py` with the new version variant in the same PR

---

## Part 5: Claude Code workflow — phase by phase

This section tells you exactly how to drive Claude Code for each phase. Model selection, when to use `/plan`, when to invoke multi-agent review, and the recommended session structure. Follow this rather than improvising.

### Universal session opener

Every Claude Code session for this project starts with this prompt:

```
Project context: Sonna Editor build.

Please read these two documents in the project root before responding:
1. HANDOVER.md (project context, architectural reasoning, decisions)
2. SONNA_EDITOR_BUILD_SPEC.md (detailed task specs)

I'm working on [Phase X, Task Y] today. Before writing any code, confirm you've read both documents and summarise back the relevant context for this task in 3-5 bullets.
```

This forces Claude Code to load context properly before acting. Skip it and you'll get generic suggestions that don't fit the project.

### Phase-by-phase workflow

#### Phase 0 — Environment setup

**Model:** Sonnet (default)
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Routine project scaffolding. Well-defined task, low risk of getting wrong.

**Workflow:**
1. Universal session opener
2. Paste Task 0.2 prompt from SPEC
3. Let Claude Code execute
4. Run `scripts/verify_environment.py` and confirm output
5. Commit and push

**Estimated session count:** 1 session, 1-2 hours

---

#### Phase 1 — Data extraction pipeline

**Tasks 1.1, 1.2, 1.3 (XMP, DNG, Extract):**
**Model:** Sonnet
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Well-trodden patterns (file I/O, subprocess wrappers, library usage).

**Workflow per task:**
1. Universal session opener
2. Paste task prompt from SPEC
3. Implement
4. Write/run tests
5. Manual validation: write an XMP, open in Lightroom, confirm edits applied (Task 1.1) / convert a real RAW to DNG (1.2) / extract metadata from real Sonna photos (1.3)
6. Commit

**Estimated:** 1 session per task, 1-2 hours each

**Task 1.4 (Lightroom catalog reader) — HIGH RISK:**
**Model:** Opus
**Use `/plan`:** Yes
**Multi-agent:** Yes (architect + engineer + reviewer)
**Why:** Schema is partially undocumented, version-dependent, edge cases matter, opening user catalogs read-only is critical. Worst-case task in the project.

**Workflow:**
1. Universal session opener
2. Run `/plan` with the task prompt — get the architectural approach surfaced
3. Review the plan, push back on anything that looks fragile
4. Invoke multi-agent: ask Claude Code to act as architect first (propose schema query strategy), then engineer (implement), then reviewer (specifically check: read-only enforcement, missing photo handling, schema version compatibility, error messages)
5. Test against a *copy* of a real Lightroom catalog (never the live working catalog)
6. Manual validation: confirm extracted develop settings match what `xmp.read_xmp()` returns when Lightroom exports the same photo's metadata
7. Commit

**Fallback if this gets ugly:** abandon catalog reader, use Lightroom's "Save Metadata to File" feature to dump XMPs alongside RAWs, then read XMPs directly. Catalog reader becomes a productivity tool for v2, not a critical path.

**Estimated:** 2-3 sessions, 4-6 hours total

**Tasks 1.5, 1.6 (Dataset builder, Auditor):**
**Model:** Sonnet
**Use `/plan`:** No (1.5), Yes (1.6)
**Multi-agent:** No
**Why:** 1.5 is straightforward dataset construction. 1.6 benefits from `/plan` because the audit report design (what to check, how to present findings) is a thinking task before implementation.

**Workflow:**
- 1.5: standard implementation
- 1.6: `/plan` first to define the audit checks and report structure, then implement

**Estimated:** 1 session each, 1-2 hours each

---

#### Phase 2 — Mode B (Lite preset)

> **Note (added 2026-05-14):** This section describes the **legacy rule-based Mode B** workflow that shipped in Phase 2. The production direction for Mode B is now the model-based rebuild track (see Decision 4 + Part 6 item 17). The Phase 2 deliverable still exists as a CLI tool and remains a valid fallback, but new Mode B work follows the rebuild track, not this workflow.

**All tasks (2.1, 2.2, 2.3):**
**Model:** Sonnet
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Routine implementation. The architectural decisions are already made (preset format, content-aware adjustment logic, XMP output). This is wiring it together.

**Workflow:**
1. Standard session for each task
2. **Critical end-of-Phase validation:** apply Mode B to a real recent Sonna shoot, open in Lightroom, confirm edits look right across varied lighting

**MILESTONE:** End of Phase 2 = working tool you use day-to-day.

**Estimated:** 3 sessions total, 4-6 hours

---

#### Phase 3 — Model architecture & training

**Task 3.1 (Model architecture) — HIGH STAKES:**
**Model:** Opus
**Use `/plan`:** Yes
**Multi-agent:** Yes (architect + engineer + reviewer + QA)
**Why:** Architecture decisions are hard to reverse later. Wrong choices here propagate through the rest of the project. Multi-agent review catches issues early.

**Workflow:**
1. Universal session opener with explicit note: "This is high-stakes architectural work. I want multi-agent review."
2. `/plan` with the Task 3.1 prompt — get architectural reasoning surfaced
3. Architect agent: propose architecture, justify backbone choice, fusion strategy, output head structure
4. Engineer agent: implement based on architect's plan
5. Reviewer agent: specifically check — parameter count reasonableness, CUDA/MPS/CPU compatibility, batch dimension handling, save/load correctness, embedding registry update logic
6. QA agent: write comprehensive tests including edge cases (single-sample batch, missing metadata fields, embedding overflow)
7. Run tests, validate parameter count and forward pass on dummy data

**Estimated:** 2-3 sessions, 4-6 hours

**Task 3.2 (Loss function & augmentation) — HIGH STAKES:**
**Model:** Opus
**Use `/plan`:** Yes
**Multi-agent:** Yes (architect + engineer + reviewer)
**Why:** Loss weights critically affect training quality. Augmentation logic must augment input but preserve targets — easy to get wrong, hard to debug.

**Workflow:**
1. Session opener
2. `/plan` to surface loss weight reasoning and augmentation strategy
3. Architect: justify loss weights per parameter category, specify augmentation pipeline that augments input only
4. Engineer: implement
5. Reviewer: specifically check — gradient balance across parameters, target preservation through augmentation pipeline, deterministic seeding for reproducibility
6. Visual validation: save augmented examples to disk, confirm they look reasonable

**Estimated:** 1-2 sessions, 2-4 hours

**Task 3.3 (Training pipeline):**
**Model:** Opus for the Lightning module, Sonnet for the data module
**Use `/plan`:** Yes for module.py, no for datamodule.py
**Multi-agent:** Just architect + engineer for module.py
**Why:** Training pipeline correctness matters but most of it is well-known PyTorch Lightning patterns.

**Workflow:**
1. datamodule.py: standard Sonnet session
2. module.py: `/plan` with Opus, surface optimizer/scheduler choices, then implement
3. train.py CLI script: standard Sonnet session
4. Smoke test: train for 2 epochs on 100 photos, confirm loss decreases

**Estimated:** 1-2 sessions, 3-5 hours

**Task 3.4 (Real training run) — NOT A CODING TASK:**
This is operational. Run the training, monitor TensorBoard, evaluate results. If quality misses targets, come back with the audit data and ask Opus how to debug. Don't try to fix mid-training — let runs complete and analyse.

**Estimated:** Wall-clock 3-5 hours per training run, your active time 30-60 min

---

#### Phase 4 — Inference engine

**Task 4.1 (Inference engine):**
**Model:** Sonnet, escalate to Opus if performance is poor
**Use `/plan`:** No initially, yes if optimisation pass needed
**Multi-agent:** No
**Why:** Mostly wiring up trained model to the existing (legacy Mode B) preset pipeline. The performance optimisation might warrant `/plan` later. Note: the model-based Mode B rebuild track reuses this same Phase 4 inference pipeline directly — Mode A and Mode B share `inference/engine.py` and `inference/pipeline.py` (see Decision 4).

**Workflow:**
1. Standard implementation session with Sonnet
2. Real-data validation: process a real shoot end-to-end, time it, compare to the 5-minute-per-1000-photos target
3. If too slow: new session with Opus + `/plan` to surface bottlenecks and optimisation strategy

**Estimated:** 1-2 sessions, 2-4 hours

---

#### Phase 5 — Continuous learning

**Task 5.1 (Edit capture & delta tracking):**
**Model:** Sonnet
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Mechanical comparison logic.

**Task 5.2 (Fine-tuning script) — HIGH STAKES:**
**Model:** Opus
**Use `/plan`:** Yes
**Multi-agent:** Yes (architect + engineer + reviewer + QA)
**Why:** This is the most dangerous code in the project. A bug here can corrupt your trained profile or silently degrade quality over time. The version preservation logic is critical.

**Workflow:**
1. Session opener with explicit warning: "Fine-tuning logic must never overwrite the original model. New version on every run. Rollback must work."
2. `/plan` with full task prompt
3. Architect: design the version management, learning rate schedule, weighted sampling strategy
4. Engineer: implement
5. Reviewer: specifically check — original model preservation, weighted sampling correctness, validation set consistency between original training and fine-tuning, registry update atomicity
6. QA: write tests covering rollback scenarios, val-loss-worse scenarios, registry corruption scenarios

**Estimated:** 2 sessions, 3-5 hours

---

#### Phase 6 — Profile management

**All tasks:**
**Model:** Sonnet
**Use `/plan`:** No
**Multi-agent:** No
**Why:** JSON registry CRUD operations. Routine.

**Estimated:** 1 session, 2-3 hours

---

#### Phase 7 — Electron UI

**Task 7.1 (Scaffolding):**
**Model:** Sonnet
**Use `/plan`:** Yes (just for the layout decisions, brief)
**Multi-agent:** No
**Why:** UI structure benefits from upfront thinking but the implementation is well-trodden.

**Workflow:**
1. `/plan` to surface layout, navigation, view structure
2. Implement with Sonnet
3. Visual validation: open the app, click through every view

**Tasks 7.2-7.5 (Wire up functionality):**
**Model:** Sonnet
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Connecting UI to existing backend logic. Mechanical work.

**Workflow:** standard sessions, manual UI testing after each task

**Task 7.6 (Polish & packaging):**
**Model:** Sonnet, escalate to Opus for PyInstaller-specific issues if they arise
**Use `/plan`:** No
**Multi-agent:** No
**Why:** Polish is iterative. Packaging is well-documented.

**Estimated for Phase 7:** 5-6 sessions, 10-15 hours

---

### Quick reference: model selection matrix

| Task type | Model | `/plan` | Multi-agent |
|---|---|---|---|
| Project scaffolding (Phase 0) | Sonnet | No | No |
| File I/O wrappers, parsers | Sonnet | No | No |
| CLI scripts | Sonnet | No | No |
| Tests | Sonnet | No | No |
| Documentation | Sonnet/Haiku | No | No |
| UI scaffolding | Sonnet | Yes (light) | No |
| UI wiring | Sonnet | No | No |
| Lightroom catalog reader | **Opus** | **Yes** | **Yes (3 agents)** |
| Model architecture | **Opus** | **Yes** | **Yes (4 agents)** |
| Loss function & augmentation | **Opus** | **Yes** | **Yes (3 agents)** |
| Training pipeline (Lightning) | Opus | Yes | architect+engineer |
| Data module | Sonnet | No | No |
| Dataset auditor | Sonnet | Yes (light) | No |
| Inference engine | Sonnet | No (initially) | No |
| Inference optimisation | Opus | Yes | No |
| Fine-tuning logic | **Opus** | **Yes** | **Yes (4 agents)** |
| Profile registry | Sonnet | No | No |
| Debugging when stuck >30min | Opus | Sometimes | Sometimes |

### Multi-agent invocation pattern

When the table says "multi-agent," here's how to actually invoke it in Claude Code. Don't just say "use 4 agents" — that's vague. Use this template:

```
For this task, I want you to work through it as four separate roles in sequence:

1. ARCHITECT: Before any code, propose the design. Specifically address [task-specific concerns from spec]. Pause and let me review.

2. ENGINEER: After I approve the architecture, implement it. Stick to the architect's plan.

3. REVIEWER: After implementation, switch roles. Critically review the code for: [specific concerns]. Find at least 3 things to improve or flag, even if minor.

4. QA: Write comprehensive tests covering: [specific test cases including edge cases].

Pause between each role for my review.
```

The pause-between-roles is what makes this work. If you let Claude Code run all four roles in one go, it'll just rationalise its initial code through all of them. The pause forces real review.

### What NOT to do with Claude Code

- **Don't ask for a "complete implementation" of a whole phase.** Work task by task.
- **Don't skip the universal session opener.** Without it, Claude Code loses project context and proposes generic solutions.
- **Don't accept code without tests** for non-trivial logic. Especially for Phase 1, 3, and 5.
- **Don't merge code that hasn't been run end-to-end** at least once on real data.
- **Don't use multi-agent on every task.** It's friction that doesn't help on routine implementation.
- **Don't use Sonnet for the high-stakes tasks** flagged in the table. The Opus quality difference is real on hard architectural work.
- **Don't change scope without updating HANDOVER.md.** If a decision is revisited, document why.

### Critical practices for this project

1. **Always run tests after changes.** The spec includes pytest for every component.
2. **Commit small.** Each task = 1-3 commits, not 20.
3. **Validate against real data early.** Phase 1 dataset auditor runs against real catalog as soon as dataset builder works. Don't wait until Phase 4 to discover data issues.
4. **Don't skip the audit.** Phase 1.6 audit before Phase 3 training. Skipping costs you days of training a bad model.
5. **Test XMP roundtrip in Lightroom early.** First time you write an XMP, manually open in Lightroom and verify edits applied. Catches namespace bugs immediately.
6. **Sync `models/` to cloud storage.** Trained profiles are real IP. Dropbox/iCloud Drive sync is free and fast.
7. **Keep HANDOVER.md updated.** As decisions get revisited, update the doc. Future-you will thank present-you.
8. **Verify defaults flips with the full test suite.** When changing implicit defaults (function default args, class default attrs, config defaults consumed by callers who don't pass the value explicitly), run `uv run pytest tests/` before committing. Localised verification will miss callers that depend on the old default.

---

## Part 6: Deferred decisions

These are decisions explicitly kicked to later phases. Don't try to solve them now.

1. **Phase 8 team distribution mechanics** (code signing, installer, auto-update, shared profile sync) — wait until Phase 7 is solid for Darshil personally
2. **Whether to support multiple Sonna profiles** (commercial, portrait, lifestyle as separate profiles) — initially train one "Sonna Events" profile, evaluate if quality justifies splitting
3. **Whether to integrate AI Culling** — separate problem, separate model, may be Phase 9+
4. **Cloud training option** — only revisit if M1 training becomes a bottleneck
5. **Subject mask / local adjustment support** — significant additional complexity, evaluate after global edits work well
6. **Higher input resolution (768px)** — **[REVISED 2026-05-13]** original plan: 384px v1, 512px v2 (production target), 768px v3 (stretch goal). v1 production runs at 256px, not 384px as originally planned. The 384 → 512 → 768 staged approach was never executed — v1 trained at 256px for M1 training speed. The direction (resolution bump in v2 to improve local-detail slider predictions) remains valid. v2 target resolution to be decided based on the skip-connection experiment outcome and Phase 5 results — likely 512px as originally specified, but 384px is a reasonable intermediate if compute is constrained.
7. **Ensemble models** — start with single model, ensemble only if quality plateaus
8. **"Quick Profile" calibration mode** — preference-driven mid-tier between Mode B (preset only) and Mode A (full trained model). User supplies a base preset and edits 30-50 calibration photos that span their typical lighting scenarios. System trains a lightweight personal profile in 10-20 minutes using only those edits. Predicts per-photo variation across all sliders based on the calibration data. Useful for new contractors, new clients, or fast onboarding without thousands of training photos. No assumptions about style are encoded — the calibration photos are the sole style signal.
9. **Tone curves — v2 architectural fix (scope refined 2026-05-13 by v1.2.3 audit + three-way comparison vs Imagen Personal)**
   - **v1.0.1-era framing now disproven:** "all four channels converge to near-identity" is wrong for v1.2.3. The all-slider audit + three-way comparison show a more nuanced failure pattern.
   - **Actual v1.2.3 tone curve behavior** (mean L2 distance from identity, interior Pt2–Pt5 Y values vs matching X coordinates):
     - **Composite:** pred_dev 15.93, target_dev 19.05 → **partial learning, ~83% of target deviation** (slight undershoot, not collapse)
     - **Red:** pred_dev 25.89, target_dev 22.81 → **overshoot ~114%** (predicts non-zero deviation, but wrong direction or magnitude)
     - **Green:** pred_dev 23.25, target_dev 19.84 → **overshoot ~117%**
     - **Blue:** pred_dev 22.17, target_dev 18.80 → **overshoot ~118%**
     - For comparison, Imagen Personal's Composite pred_dev is 20.99 (closer to its target than Saha's Composite is to ours); Imagen R/G/B also overshoot.
   - **The actual collapse is per-control-point, not per-channel.** Audit-categorised COLLAPSED tone-curve fields:
     - Composite: `ToneCurve_Pt1_Y`, `ToneCurve_Pt2_X`, `ToneCurve_Pt2_Y`, `ToneCurve_Pt3_X`, `ToneCurve_Pt3_Y` (5 of 12)
     - Red: `ToneCurveRed_Pt2_Y` (1 of 12)
     - Blue: `ToneCurveBlue_Pt2_Y`, `ToneCurveBlue_Pt3_Y` (2 of 12)
     - Green: 0 individually collapsed (but the channel as a whole overshoots)
     - Pt4–Pt5 (upper-tone) and Pt6 (endpoint) are *not* collapsed across channels — model partially learns upper-tone shape.
   - **Refined v2 fix scope:** focus on **lower-tone control points (Pt1–Pt3 Y values across channels)**, not the whole curve head. Two distinct problems to address:
     - **Lower-tone collapse** (Composite + Red Pt2_Y + Blue Pt2/3_Y): model gives up on these specific predictions. Likely the same metadata-bottlenecked routing issue as Tint — Pt1–Pt3 Y values may need more direct routing of relevant features.
     - **R/G/B overshoot:** model predicts non-identity but in wrong directions. Different mechanism — possibly model is learning per-shoot curve mean offsets but not per-photo content-driven adjustments.
   - **v2 approach options (revised):**
     - **Loss formulation:** penalise predicted non-zero deviation that doesn't match target direction (addresses R/G/B overshoot specifically); flatness penalty for collapsed control points
     - **Representation change:** predict curve as deltas from identity (still applicable) — forces the model to explicitly justify non-zero departures
     - **Re-architect the curve head as residuals** rather than absolute coordinates (most aggressive fix; bundle with skip-connection experiment if that pattern generalises)
     - **Variable-length curves:** 6 fixed → 8–12 points for complex grades — separate question; bundle decision with curve work but don't conflate
   - **DO NOT** add a baseline template at inference time. Keep `inference/pipeline.py` writing whatever the model predicts — preserves measurability of v2's improvement against the v1.2.3 baseline numbers above.
   - **The composite tone curve is load-bearing for Sonna's grade.** Shipping permanently-collapsed Pt1–Pt3 is not acceptable. The R/G/B overshoot is also a real problem; ignoring it would let the v2 retrain accept arbitrary non-identity predictions.

10. **Perspective / lens correction fields**
    - **Status in v1.0.1:** `PerspectiveScale ≈ 99.15` (default 100) caused a visible black-border artifact — Lightroom zooms out to expose the canvas edge. Fixed by `_V1_SKIP_FIELDS` in `inference/pipeline.py`, which omits all `Perspective*`, `LensManualDistortionAmount`, and `VignetteAmount` from written XMPs.
    - **v2 update (2026-05-13):** `LensProfileDistortionScale` and `LensProfileVignettingScale` are now in SLIDER_FIELDS (idx 143-144) and predicted by v2's `lens_profile_head`. They are NOT in `_V1_SKIP_FIELDS` and will be written to XMPs from v2 models. The `Perspective*`, `LensManualDistortionAmount`, and `VignetteAmount` fields remain in the skip list (geometric corrections still deferred).
    - **v2 decision:** either remove these fields from `SLIDER_FIELDS` entirely (geometric corrections are per-photo, not style-learnable), or accumulate enough training data that predictions are reliably near-default and the artifact disappears.

11. **Additional v2 architectural items (bundle with curve work)**
    - Local adjustments: brush, gradient filter, radial filter masks
    - AI Masks: Subject, Sky, Background, People
    - Spot Healing / Clone Stamp
    - Crop and Transform (currently filtered; decide whether to model or drop)
    - Camera Profile categorical selection (currently embedding falls back to "unknown")
    - AI Denoise (separate model, LR-native for now)
    - 512px training resolution (current: 384px)

12. **`_count_unedited` semantic redesign:** current implementation conflates "value is zero" with "value is unedited." This worked accidentally for v1 because most sliders defaulted to 0. v2 extension fields have non-zero defaults (Defringe Hue Lo=30, LensProfile scales=100, CurveRefineSaturation=100), so the zero-count threshold becomes fragile. Redesign as "count fields at `LR_DEFAULTS[field]`" rather than "count fields equal to 0". Not urgent — v1.2.3 audit still works, v2 audit thresholds are conservative — but worth fixing before audit-driven decisions become load-bearing.

13. **WB skip semantics unification (v2 cleanup):** Mode A (inference) uses explicit AsShot substitution for skipped WB fields. Mode B (preset) uses Temperature=0 sentinel for "not specified", which xmp.py converts to attribute omission relying on Lightroom's AsShot fallback (undocumented behaviour). Mode B's mechanism has been working but relies on the same unverified assumption that Mode A was just fixed for. Unifying both modes on explicit AsShot substitution would remove the inconsistency and the unverified-LR-behaviour dependency. Not urgent — Mode B has been working in production — but architecturally cleaner. Estimated ~1 hour of careful work touching `preset/pipeline.py` and `adjuster.py` with focused regression testing on Mode B preset paths.

14. **v1.2.3 `default_skip_fields` rationale (current production config in `v1_learning/model-v1.2.3-prod256.json`):** the shipping profile skips three sliders by default — Tint, SplitToningShadowHue, ColorGradeMidtoneHue. Each represents a v1.2.3 model failure mode mitigated at runtime via the skip mechanism — a **reversible runtime filter** (see `inference/pipeline.py:_V1_SKIP_FIELDS` + WB skip semantics in commit `dac535e`), **NOT an architectural fix**. The underlying model issues persist; default_skip_fields just suppresses unreliable predictions for production use until a v2 retrain addresses each.

    - **`Tint`** — model collapses to ~mean (+7.4) despite ρ=0.91 correlation with `as_shot_tint` in training data (deep dive: `scripts/output/tint_deep_dive.md`). The signal is near-perfect and the input is available (`MetadataEncoder.as_shot_tint_fc`), but v1.2.3's fusion-MLP doesn't extract it. Loss weight 4.0 didn't help. Phase 5 fine-tuning on similar data won't fix — architecture issue. Skip triggers Mode A WB substitution (commit `dac535e`): `crs:Tint` is **written with the AsShot value, not omitted** — matches user intent on ~91% of photos. Planned arch fix: see item 15.

    - **`SplitToningShadowHue`** — COLLAPSED per the v1.2.3 audit (std ratio 0.03 on test, MAE 48° on the 360° hue wheel). Real spread exists in targets; model predicts ~constant regardless of input. Suspected to share Tint's fusion-MLP routing problem — signal exists somewhere in the inputs, model doesn't route it to the head. Skip omits `crs:SplitToningShadowHue` from XMP; Lightroom uses the panel default.

    - **`ColorGradeMidtoneHue`** — COLLAPSED in v1.2.3 (std ratio 0.05). Also COLLAPSED in Imagen Personal per the three-way comparison (`~/Desktop/saha_three_way_comparison.md`) — fundamental data limit, not Saha-specific. Both trained models give up on this hue wheel; the data may genuinely not constrain it. Skip omits the attribute.

    Temperature is **intentionally NOT in `default_skip_fields`** despite its 731 K MAE on test: the epistemic clamp (commit `55aa70b`) provides safety against `exp()` amplification at the tails, and the model does learn the log-space spread well (std ratio 0.93, corr 0.71). The 731 K MAE is accepted as a known v1.2.3 limit pending the skip-connection experiment (item 15). The same architectural hypothesis applies — `as_shot_temperature` has ρ=0.94 correlation with target Temperature in training data, and the same fusion-MLP bottleneck that loses Tint's signal may be limiting Temperature too. Skipping Temperature now would make Item 1's clamp pointless and force AsShot WB universally — discarding the model's real learning signal on the in-distribution cases where it works.

    Skipping is per-profile config — reversible, no checkpoint or code change. Users can override per-job via the `[x] Skip <field>` checkboxes in the Saha Output panel.

15. **WB head metadata-skip-connection (implemented 2026-05-28, pending training audit):** v1.2.3's WB head (`832 → 128 → 64 → 2`) receives metadata only via the fusion MLP's compressed output (image features 768-d + metadata 64-d → fusion → 832-d → head). The Tint deep dive (`scripts/output/tint_deep_dive.md` §7) found that despite `as_shot_tint` having ρ=0.91 with the Tint target and being wired as a metadata input (`MetadataEncoder.as_shot_tint_fc`), the model fails to extract this near-identity mapping — predictions collapse to ~mean regardless of as_shot_tint. Hypothesis: the fusion MLP's `(128 → 64)` bottleneck dilutes the direct `as_shot_tint → Tint` shortcut against image features competing for representation; raising Tint's loss weight to 4.0 didn't compensate.

    Implementation: new v2 models default to `use_wb_metadata_skip=True`. The WB output is `learned_residual + Linear([log(as_shot_temperature), as_shot_tint])`, and that linear layer is identity-initialised, so the starting prediction is AsShot WB plus a learnable residual. Legacy native and Lightning checkpoints load with the skip disabled unless their checkpoint metadata explicitly enables it, preserving v1.2.x production behaviour. Next action: train a v2 profile, then run the all-slider audit against v1.2.3 to confirm whether Temperature/Tint improved.

16. **Fundamental data-limit sliders (added 2026-05-13 from three-way comparison findings):** the three-way XMP comparison (`~/Desktop/saha_three_way_comparison.md`) revealed that some sliders collapse in BOTH Saha v1.2.3 AND Imagen Personal — two trained models from independent lineages, training data, and architectures. When this happens, the most parsimonious explanation is that **the slider value is genuinely not predictable from image content alone** (the data does not constrain it).

    Currently documented:
    - **`ColorGradeMidtoneHue`** — COLLAPSED in Saha (std_ratio 0.05) AND in Imagen Personal (per three-way comparison). Both models converge to ~constant. No amount of more training data on similar Sonna shoots will fix this; the signal isn't in the inputs.

    **Implication:** for sliders in this category, no architectural fix is warranted — `default_skip_fields` is the correct permanent answer. Distinguishing "fundamental data limit" (skip permanently) from "metadata-bottlenecked routing" (fixable via architecture, item 15) is essential before allocating retrain budget.

    **How to identify:** when running future audits, cross-reference the COLLAPSED list against any available competitor-model output. A slider that collapses across multiple independent models is almost certainly in this category.

17. **Mode B rebuild track — COMPLETE 2026-05-14 (supersedes Phase 2's rule-based Mode B as the production direction):** Mode B is now a model-based profile type that shares the `SonnaEditor` architecture and Phase 5 fine-tuning pipeline with Mode A — only the initial checkpoint differs (see Decision 4 for full rationale). All three substeps shipped:

    - **Step 1: Style survey. DONE 2026-05-14 (`71fcf2b`), restored to six-question Lite behavior 2026-06-03.** CLI questionnaire in `scripts/run_style_survey.py` maps user preferences onto Exposure2012, Temperature, Tint, Contrast2012, Saturation, and Shadows2012 via 5-point answers. Output: JSON payload `{questions: {key: {slider_field, answer, offset}}, summary}` consumed by Step 2. Module logic in `src/sonna_editor/mode_b/survey.py`. Initial Lite processing still dynamically adjusts only Exposure/WB while preset look sliders stay fixed; all six survey answers remain in the profile package for calibration metadata and future fine-tuning. The Saha Lite wizard exposes the same six-question survey.
    - **Step 2: Preset-to-checkpoint converter. DONE 2026-05-14 (`a8d7c04`), v2-base compatibility fixed 2026-06-01, adaptive Lite output fixed 2026-06-02.** `scripts/build_mode_b_checkpoint.py` ingests `(preset.xmp + survey.json)` and produces a `SonnaEditor` ckpt. Implementation in `src/sonna_editor/mode_b/checkpoint_builder.py`: loads the base checkpoint at its native slider set, preserves the base feature layers for future fine-tuning, zeroes each output head's final linear weights, sets final biases to absolute preset+survey targets in prediction space (log-K for Temperature, raw units elsewhere), zeroes v2 `wb_metadata_skip`, and saves a Mode B ckpt with sidecar JSON marking `profile_type: "mode_b_initial"` plus the inherited `slider_set_version`. v1 bases produce v1 Mode B ckpts; v2 bases produce v2 Mode B ckpts without down-converting. Built-in verification confirms final weights are zero and biases equal the preset+survey target; initial processing then uses sidecar preset/survey metadata for adaptive per-photo output. Focused tests live in `tests/test_checkpoint_builder.py`.
    - **Step 3: Mode B inference path. DONE 2026-05-14, revised 2026-06-02.** Initial validation run on `test_data/mode_b_test/` (19 RAW files; `DP Event.xmp` preset + all-zero survey) confirmed 16/16 photos processed end-to-end. On 2026-06-02, the first-pass Mode B path was corrected to match the expected Lite behavior: `process_shoot_with_model()` detects `profile_type: "mode_b_initial"`, bypasses `InferenceEngine`, reloads the copied preset and survey, computes per-photo auto Exposure/WB only through `preset.adjuster`, writes those adjusted values to XMP, and records them in `sonna_predictions.json`; all non-Exposure/WB look sliders stay preset-fixed. The exposure adjuster now protects upper tones using 85th/95th percentile luminance checks after real-folder testing showed mean-only exposure could over-lift shadow-heavy event frames. After a Mode B profile is fine-tuned, normal model inference remains the path. Two small fixes landed during the original validation:
        - `b6b2d1e` `feat(inference): record profile_type and slider_set_version in predictions sidecar` — extends `sonna_predictions.json` with `profile_type`, `profile_id`, `base_checkpoint`, `slider_set_version` propagated from the ckpt's sidecar JSON. Phase 5 capture branches on these fields to pick the right delta baseline (Mode A trained vs Mode B preset-derived). Closes item 19.
        - `7c93906` `fix(mode-b): Mode B ckpt sidecar inherits base ckpt's resolution` — bug surfaced during validation where Mode B sidecars recorded `config.IMAGE_RESOLUTION` (the global default, 512) instead of the base ckpt's actual resolution (v1.2.3 = 256). The resolution now also controls preview extraction for the initial adaptive Lite branch.
        - 2026-06-02 adaptive Lite fix — active v2-base Lite profiles were stacking base model output on top of preset/survey targets, then the first attempted fix made output too constant. Initial Mode B processing now uses preset+survey as the style baseline plus per-photo auto corrections, so Lite is image-adaptive before fine-tuning.

    Operational commands for producing a Mode B profile end-to-end live in Part 7 "Mode B usage". The integration test `tests/test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation` covers the full path from base ckpt through Mode B ckpt build to predictions-sidecar contents.

    **Why sequenced before Phase 5 redesign:** Phase 5 infrastructure (capture, finetune, registry update) is already shipped for Mode A. If Mode B rebuild lands first, Phase 5 can be re-validated to confirm it handles both profile types with no code changes (expected outcome — the only difference is the starting ckpt). If Phase 5 needs adjustment for Mode B (e.g. different correction-weight defaults for preset-initialised profiles, or a survey-update flow distinct from edit-capture), surfacing those needs during the rebuild is cheaper than after.

    **What remains of legacy Phase 2:** the preset parser, content-aware adjuster, and end-to-end CLI stay in `src/sonna_editor/preset/` as a fallback / diagnostic tool. They are not deleted. The WB skip semantics unification (item 13) still applies to the legacy CLI if it stays in use. The Saha app's existing Mode B switch may need a small UI adjustment when the new model-based Mode B lands (e.g. surface preset-init metadata in the profile card) — defer until rebuild Step 2 is concrete.

18. **v1/v2 SLIDER_FIELDS shape-mismatch bug class — RESOLVED 2026-05-14 across 6 commits.** Recorded here as the post-mortem for the bug class, not as outstanding work.

    **Root cause:** The v2 slider list expansion (commit `3d0d90c`, 2026-05-13 15:12 NZT) grew `config.SLIDER_FIELDS` from 135 → 147 while v1 checkpoints (including production `v1_learning/model-v1.2.3-prod256.ckpt`) still output `[B, 135]` tensors. Every downstream site that paired `SLIDER_FIELDS` (length 147) with a model output tensor (length 135 for v1) silently developed a shape mismatch. The full pytest suite passed because no test loaded a v1 checkpoint and ran it end-to-end through the production pipeline.

    **Discovery sequence:**
    - 2026-05-14: Mode B rebuild Step 2 implementation surfaced the first crash — `postprocess_predictions()` range tensor broadcast against v1 [B, 135] output.
    - Hotfix `4133e8a` patched that one site, then production v1.2.3 inference still crashed at the very next pipeline step — `predictions_to_dict` IndexError at `i=135`. Same root cause, second site.
    - A repo-wide audit then found **7 hard-crash sites + 3 silent-degradation sites** following the same pattern. Patching individually would keep producing surprises.

    **Resolution lineage (6 commits, 2026-05-14):**
    | Commit | Scope |
    |---|---|
    | `4133e8a` | `fix(inference): postprocess_predictions respects model's slider_set_version` — first hotfix, unblocked engine.predict but exposed the next site |
    | `24756c1` | `feat(slider-set): version-aware field list helpers` — new `sonna_editor.slider_set` module with `v1_fields()`, `fields_for_version(slider_set_version)`, `fields_matching_tensor(tensor)` |
    | `31e82ca` | `fix(losses): version-aware WeightedSliderLoss buffers and methods` — required `slider_set_version` arg; buffers + `direction_stats` + `per_field_mae` migrated; 40 test sites pinned to "v2", new v1 coverage section added |
    | `8350903` | `fix(slider-set): migrate predictions_to_dict to fields_matching_tensor` — Site 1 fixed; closes the second production crash |
    | `7c58139` | `fix(finetune): version-aware field handling in capture/delta/retrain` — Sites 8-10 (the 3 silent-degradation sites) migrated to dict-key-derived iteration |
    | `28f3290` | `test(integration): v1 production inference pipeline end-to-end` — production-pipeline integration test that would have caught both shape bugs immediately if it had existed |

    **What was actively broken in production:** The Saha app's main entry point (`src/sonna_editor/api/routes/process.py` → `inference/pipeline.py:process_shoot_with_model` → `engine.predict`) crashed on every v1 inference call between 2026-05-13 15:12 NZT and the hotfix landing on 2026-05-14. Four diagnostic scripts (`scripts/run_v1_pilot.py`, `scripts/audit_v1_predictions.py`, `scripts/audit_v1_reading.py`, `scripts/process_shoot_model.py`) routed through the same broken path. The semantic-degradation sites in `finetune/` would have produced mis-attributed Phase 5 capture data for any v1 profile fine-tune (latent — Phase 5 hasn't been run on v1.2.3 production data yet).

    **What this prevents going forward:** Direct iteration over `config.SLIDER_FIELDS` is now prohibited in any code path that pairs the field list with a model output tensor (see Part 4 "Slider list expansion v1/v2 compatibility" risk for the rule). `tests/test_inference_pipeline_integration.py` exercises v1 and v2 end-to-end on every commit.

19. **`sonna_predictions.json` schema extension — RESOLVED 2026-05-14 (commit `b6b2d1e`).** Originally deferred from commit `7c58139` (slider_set refactor Commit 3) because the schema change naturally belonged with Mode B Step 3 (item 17 sub-step c). Landed there as planned.

    **What shipped:** `inference/pipeline.py:373` now reads the loaded ckpt's sidecar JSON (same lookup pattern as `engine.py:123`) and writes four additional fields to `sonna_predictions.json`:
    - `profile_type` — `"mode_b_initial"` for Mode B ckpts, `None` for legacy Mode A ckpts (e.g. v1.2.3 production whose sidecar predates this field)
    - `profile_id` — slug + UTC timestamp from the ckpt's sidecar
    - `base_checkpoint` — for Mode B, the v1.2.3 backbone the Mode B ckpt warm-loaded from
    - `slider_set_version` — `"v1"` or `"v2"`, read from `engine._model._slider_set_version`

    **Phase 5 capture branches on these fields** to pick the right delta baseline (Mode A trained = model's per-image prediction; Mode B initial = preset+survey values from the Mode B sidecar). Test coverage in `tests/test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation` exercises the full sidecar propagation path.

    **Not addressed by this resolution:** threading `slider_set_version` through `finetune/{capture,delta,retrain}.py` to replace the dict-key-derived approach from commit `7c58139`. The dict-key-derived approach works correctly for v1 and v2 captures and handles mixed-version cases gracefully; the explicit-threading alternative would be cleaner but isn't strictly necessary. Revisit if Phase 5 redesign surfaces a concrete need.

---

## Part 6b: Continuous learning design principles

These are fixed decisions for the Phase 5 continuous learning loop. Do not reverse without discussion.

**Capture is lossless.** `capture_user_edits()` records all 135 slider deltas regardless of magnitude. No threshold filtering at capture time. Whether a delta is "significant" is a fine-tuning-time decision based on data. Destroying small deltas at capture time would prevent learning subtle systematic biases.

**All captured photos are weighted equally by default.** `prepare_finetune_dataset()` defaults to `weight_recent=1.0` — captured edits have the same sample weight as original training data. If the user wants to up-weight corrections, that's a CLI flag (`--correction-weight N`) at fine-tune time, not a default. Rationale: photos where the user changed nothing are also training signal (agreement = the model was right).

**Fine-tuning is always opt-in.** No automatic triggers, no "you have N captures, want to retrain?" prompts. The user explicitly runs `finetune_profile.py`. Unexpected model updates could silently degrade quality on untested inputs.

**Bias detection is correlation-based, not hardcoded bins.** `analyse_deltas()` computes Spearman rank correlations between per-field deltas and all available numeric metadata columns (ISO, aperture, shutter speed, focal length). Reports pairs where |r| ≥ 0.3 AND p < 0.05 AND n ≥ 10. No hardcoded "ISO > 3200" buckets — the data reveals which metadata correlates with which biases.

**Prediction sidecar (`sonna_predictions.json`) is written on every inference run** (unless `--no-save-predictions`). Contains the full 135-field model output including `_V1_SKIP_FIELDS` — all fields are captured for v2 analysis even if they weren't written to XMP. Includes provenance: model_path, model_version, run_timestamp, v1_skip_fields list.

---

## Part 7: Quick reference

### Diagnostic reports (auxiliary artifacts produced during the v1.2.3 audit track)

Future-Claude: these are the source-of-truth for the audit findings cited throughout HANDOVER. When in doubt about a specific failure-mode characterization, check the raw report rather than trusting the summary.

| Report | What's in it | Cited where |
|---|---|---|
| `scripts/output/all_slider_audit_v1.2.3.md` | Per-slider behavior categorisation (HEALTHY / HIGH ERROR / COLLAPSED / WRONG DIRECTION / SPARSE TARGET) across all 135 v1 sliders. Test split 1,694 photos. Includes worst-offender photos, tone-curve identity-distance metrics, Temperature dual-view (log-K + Kelvin). | Current shipping state, Part 4, Part 6 items 9 + 14 + 16 |
| `scripts/output/all_slider_audit_v1.2.3_stats.parquet` | Raw per-slider stats for the audit above (machine-readable). Used by `scripts/tint_deep_dive.py` and any future comparison runs. | Reference for audit re-runs |
| `scripts/output/tint_deep_dive.md` | Tint root-cause analysis. Distribution, Spearman correlations (`as_shot_tint`: ρ=0.913), per-shoot variability, side-by-side with Temperature. Concludes with the architectural-failure-not-data-limit finding. | Current shipping state, Part 6 items 14 + 15, Part 4 model-collapse risk |
| `~/Desktop/saha_comparison_report.md` | Original two-way comparison (Saha vs an unknown — turned out to be Imagen Personal). Surfaced the fixed-value-fingerprint distinction and Imagen's collapse-vs-Saha's-collapse asymmetry. | Part 6 item 14 (ColorGradeMidtoneHue framing) |
| `~/Desktop/saha_three_way_comparison.md` | Three-way Saha v1.2.3 + Imagen Personal + Imagen Lite. Field-presence audit, crop/Upright intelligence (Imagen straightens via small `CropAngle`, not `PerspectiveUpright`), per-slider three-way disagreement table. | Current shipping state, Part 4 model-collapse risk, Part 6 items 9 + 14 + 16 |

### Mode B usage (operational commands)

Mode B profiles are Lite profile packages produced from a Lightroom preset + style-survey JSON, warm-loaded from the configured foundation checkpoint. The Lite builder inherits the foundation checkpoint's `slider_set_version`: v1 bases produce v1 Lite checkpoints, v2 bases produce v2 Lite checkpoints. Before fine-tuning, `process_shoot_model.py` detects `profile_type: "mode_b_initial"` and uses the adaptive preset branch. After fine-tuning, the profile uses normal model inference. See Decision 4 + Part 6 item 17 for the architectural reasoning.

**Three-step workflow:**

1. **Generate a style survey JSON** (one-time per profile). Captures the user's Exposure, Temperature, Tint, Contrast, Saturation, and Shadows preferences as integer answers in `{-2, -1, 0, +1, +2}`. The preset owns initial look sliders such as Contrast, Saturation, Shadows, Highlights, Whites, Blacks, and Vibrance during the first Lite processing pass. Two modes:

   ```bash
   # Interactive (prompts through each question):
   uv run python scripts/run_style_survey.py \
       --output path/to/survey.json

   # Non-interactive (CLI / scripted):
   uv run python scripts/run_style_survey.py \
       --output path/to/survey.json \
       --non-interactive \
       --answers exposure=0,temperature=0,tint=0,contrast=0,saturation=0,shadows=0
   ```

   Neutral baseline = all answers 0 (preset values pass through unchanged). Each non-zero answer applies a slider-specific offset in the profile carrier (see `src/sonna_editor/mode_b/survey.py:OFFSET_MAGNITUDES`). Initial Lite runtime then applies only Exposure/WB dynamically; after fine-tuning, the same six-answer profile package can move through normal model inference.

2. **Build the Mode B initial checkpoint** from a Lightroom preset + the survey:

   ```bash
   uv run python scripts/build_mode_b_checkpoint.py \
       --preset path/to/preset.xmp \
       --survey path/to/survey.json \
       --base-ckpt v1_learning/model-v2.0.0.ckpt \
       --output path/to/mode_b.ckpt \
       --profile-name "Mode B - Wedding Lite"
   ```

   Produces `mode_b.ckpt` (the profile carrier, size depends on base architecture) + `mode_b.json` (sidecar with `profile_type: "mode_b_initial"`, `profile_id`, `base_checkpoint`, inherited `slider_set_version`, `default_skip_fields`, and `resolution` inherited from the base ckpt). Built-in verification confirms the saved ckpt's final output weights are zero and biases match the preset+survey targets; the sidecar/copy metadata drives the initial adaptive Lite output.

3. **Run the Lite profile on a shoot** using the same CLI entry point as Mode A:

   ```bash
   uv run python scripts/process_shoot_model.py \
       --input-dir path/to/shoot/ \
       --model-path path/to/mode_b.ckpt \
       --output-dir path/to/output/
   ```

   The Saha app's API route (`src/sonna_editor/api/routes/process.py`) routes through the same `process_shoot_with_model` function. For initial Lite profiles, that function keeps preset look sliders fixed, applies per-photo Exposure/WB corrections only, and writes those adjusted values into `sonna_predictions.json` with the Mode B identity fields (`profile_type`, `profile_id`, `base_checkpoint`, `slider_set_version`) so Phase 5 capture can attribute deltas correctly.

**Phase 5 (continuous learning) is Mode-agnostic.** Once a user has edited the Mode B output XMPs in Lightroom, the same `scripts/finetune_profile.py` flow that handles Mode A will capture deltas and produce a fine-tuned ckpt. The fine-tuned profile is structurally identical to a Mode A trained profile (Decision 4 — Mode A and Mode B converge after first fine-tune).

### Slider list (147 values, in order — indices match model output)

**Tone (8, idx 0-7):** Exposure2012, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012, Clarity2012, Dehaze

**Presence (3, idx 8-10):** Texture, Vibrance, Saturation

**White balance (2, idx 11-12):** Temperature (log-space in model), Tint

**HSL Hue (8, idx 13-20):** HueAdjustment{Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta}

**HSL Saturation (8, idx 21-28):** SaturationAdjustment{Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta}

**HSL Luminance (8, idx 29-36):** LuminanceAdjustment{Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta}

**Parametric Tone Curve (7, idx 37-43):** ParametricHighlights, ParametricLights, ParametricDarks, ParametricShadows, ParametricHighlightSplit, ParametricMidtoneSplit, ParametricShadowSplit

**Color Grading (14, idx 44-57):** SplitToningShadow{Hue,Saturation}, ColorGradeShadowLum, ColorGradeMidtone{Hue,Sat,Lum}, SplitToningHighlight{Hue,Saturation}, ColorGradeHighlightLum, ColorGradeBlending, ColorGradeGlobal{Hue,Sat,Lum}, SplitToningBalance — Shadow/Highlight Hue+Sat use legacy SplitToning XMP names; Lum channels and Midtone use modern ColorGrade names

**Camera Calibration (6, idx 58-63):** {Red,Green,Blue}{Hue,Saturation} — LR uses short names without CameraCalibration prefix

**Detail — Sharpening (4, idx 64-67):** Sharpness (0-150), SharpenRadius (0.5-3.0), SharpenDetail (0-100), SharpenEdgeMasking (0-100)

**Detail — Noise Reduction (4, idx 68-71):** LuminanceSmoothing, LuminanceNoiseReductionDetail, LuminanceNoiseReductionContrast, ColorNoiseReduction

**Effects (8, idx 72-79):** PostCropVignetteAmount, PostCropVignetteMidpoint, PostCropVignetteRoundness, PostCropVignetteFeather, PostCropVignetteHighlightContrast, GrainAmount, GrainSize, GrainFrequency

**Lens Corrections (2, idx 80-81):** LensManualDistortionAmount, VignetteAmount — LensProfileEnable deferred to v2 (binary flag, not continuous)

**Transform (5, idx 82-86):** PerspectiveVertical, PerspectiveHorizontal, PerspectiveRotate (-10 to 10), PerspectiveScale (50 to 150), PerspectiveAspect

**Tone Curves (48, idx 87-134):** 4 channels × 6 control points × (X, Y). Each channel maps to an XMP rdf:Seq element (not a crs: attribute). Variable-length LR curves (2-7 points) are normalised to 6 fixed points on extraction: n>6 → even-spaced index downsampling; n<6 → piecewise-linear interpolation; n=0 or 1 → identity defaults across 6 evenly-spaced control points. All X/Y values range 0-255.
- Composite (idx 87-98): ToneCurve_Pt{1-6}_{X,Y} → crs:ToneCurvePV2012
- Red (idx 99-110): ToneCurveRed_Pt{1-6}_{X,Y} → crs:ToneCurvePV2012Red
- Green (idx 111-122): ToneCurveGreen_Pt{1-6}_{X,Y} → crs:ToneCurvePV2012Green
- Blue (idx 123-134): ToneCurveBlue_Pt{1-6}_{X,Y} → crs:ToneCurvePV2012Blue
write_xmp always writes all 4 channels (identity if values absent). Tone curve fields always return float (never None), unlike scalar slider fields.

**v2 extension fields (12, idx 135-146):** added 2026-05-13 per the locked-append-only rule (Decision 6). Predicted by 5 extension heads built only when `slider_set_version="v2"` (the default for new SonnaEditor instances).
- Noise Reduction extension (2, idx 135-136): ColorNoiseReductionDetail (0-100, default 50), ColorNoiseReductionSmoothness (0-100, default 50)
- Manual Defringe (6, idx 137-142): DefringePurple{Amount (0-20, default 0), HueLo (0-100, default 30), HueHi (0-100, default 70)}, DefringeGreen{Amount (0-20, default 0), HueLo (0-100, default 40), HueHi (0-100, default 60)}
- Lens Profile scales (2, idx 143-144): LensProfileDistortionScale (0-200, default 100), LensProfileVignettingScale (0-200, default 100). NOTE: LensProfileChromaticAberrationScale excluded — absent from real LR Classic 15.3 XMPs even with lens profile enabled.
- Calibration extension (1, idx 145): ShadowTint (-100 to 100, default 0)
- Tone Curve extension (1, idx 146): CurveRefineSaturation (0-100, default 100)

### Quality targets for v1 model

- Median exposure error < 0.20 stops
- Median temperature error < 250K
- Median tint error < 5 units
- Median HSL parameter error < 6 units
- Visual spot check: 20 random test photos look plausibly Sonna-style

### Realistic timeline at 10 hrs/week

- Week 1-2: Phases 0, 1, 2 — Mode B working end-to-end
- Week 3-4: Phase 3 — first trained Sonna profile
- Week 5: Phase 4 — inference engine for production use
- Week 6: Phase 5, 6 — fine-tuning loop and profile management
- Week 7-8: Phase 7 — desktop UI

After week 8: continuous use with periodic fine-tuning. Phase 8 team rollout when comfortable.

### Key file locations

- **Spec:** `SONNA_EDITOR_BUILD_SPEC.md` (in project root)
- **This handover:** `HANDOVER.md` (in project root)
- **Trained models:** `models/` (gitignored, sync to cloud storage)
- **Training data:** `data/parquet/` (gitignored, can archive after training)
- **Source code:** `src/sonna_editor/`
- **CLI scripts:** `scripts/`

### Hardware-specific notes

- M1 Pro 32GB remains the reference machine, but the project targets macOS, Windows, and Linux
- Use runtime device selection: CUDA first, Apple MPS second, CPU fallback
- Use fp32 precision by default
- Batch size 16 for training, 32 for inference
- Plug into power for training runs (battery drains fast)
- Hard surface for airflow during multi-hour training

---

## Part 8: For an AI assistant picking this up mid-stream

If you are Claude (or another AI) being given this document to continue the Sonna Editor project, here's what you need to know in priority order:

1. **Read this document fully before responding.** The reasoning matters. Don't skim.
2. **Read `SONNA_EDITOR_BUILD_SPEC.md` for task-level detail.** This handover is the high-level reference; the build spec is operational.
3. **Stay consistent with decisions already made** unless given explicit new context to revisit them. Don't propose switching from PyTorch to TensorFlow because you happen to know TensorFlow better, for example.
4. **Respect the deferred-decisions list.** Don't try to solve Phase 8 problems while we're still in Phase 1.
5. **Be honest about quality expectations.** Quality on day one will be limited. The fine-tuning loop is the long-term quality lever. Don't oversell.
6. **The user is Darshil, founder of Sonna Studios.** Based in Hamilton, NZ. Direct, action-oriented working style. Prefers immediately usable outputs and clean prerequisite thinking. No em dashes, no corporate filler, casual warm professional tone.
7. **Claude Code is the primary build tool.** Recommendations should assume Claude Code as the implementation partner unless told otherwise.
8. **Local hardware first.** Don't suggest cloud GPUs unless local CUDA/MPS/CPU workflows prove inadequate.
9. **The point of this build is learning + IP control + cost savings.** Not commercial sale. This is internal tooling.
10. **Ask before changing scope.** If you think a decision should be revisited, surface that thinking — don't just make the change.

---

## Final note

This build is achievable. The architecture is sound. The hardware is sufficient. The cost is zero. The main risks are time commitment, training data quality, and the inevitable edge cases that any ML build hits.

The fastest path to a working tool is: complete Phase 0 today, Phase 1 this week, Phase 2 next week. By week 2 you have something you actually use. Everything after that is making it better.

Don't over-think it. Don't try to make it perfect on the first pass. Build, use, iterate.

Good luck.

