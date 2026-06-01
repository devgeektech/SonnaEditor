# Session State - Sonna Editor

**Saved:** 2026-06-01 17:22 local time
**Current phase/task:** Training-console warning cleanup after Lite profile v2 compatibility fix.

## Current Workspace

- Repo path: `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`
- Branch: `main`, tracking `origin/main`
- Recent committed history in this checkout: `aac360a feat: Enhance dataset building and training scripts` on top of four earlier commits.
- Current intentional dirty files from this work: `HANDOVER.md`, `SESSION_STATE.md`, `TRAINING_COMMANDS.md`, `project_knowledge.md`, `scripts/train_profile.py`, `scripts/build_mode_b_checkpoint.py`, `scripts/train_v1_2_0_full_production.py`, `scripts/analyse_prediction_collapse.py`, `scripts/audit_dataset_diversity.py`, `src/sonna_editor/config.py`, `src/sonna_editor/data/catalog_dataset.py`, `src/sonna_editor/data/dataset.py`, `src/sonna_editor/data/extract.py`, `src/sonna_editor/finetune/capture.py`, `src/sonna_editor/finetune/retrain.py`, `src/sonna_editor/inference/engine.py`, `src/sonna_editor/mode_b/checkpoint_builder.py`, `src/sonna_editor/model/architecture.py`, `src/sonna_editor/training/__init__.py`, `src/sonna_editor/training/datamodule.py`, `src/sonna_editor/training/module.py`, `tests/test_architecture.py`, `tests/test_checkpoint_builder.py`, `tests/test_dataset.py`, `tests/test_extract.py`, `tests/test_training.py`.
- Generated local diagnostics/run artifacts from this pass are under gitignored `data/audits/` and `data/models/sonna-v2-scene-stats-run01/`.
- `TRAINING_COMMANDS.md` already had user edits before this pass; they were preserved and cleaned up around the resume-training note.

## Environment

- Python: 3.11.15
- uv: 0.11.17
- PyTorch: `2.11.0+cu128`
- Runtime device: `cuda`
- GPU: NVIDIA GeForce RTX 3050
- `scripts/verify_environment.py`: 11/11 checks passed
- Adobe DNG Converter discovered at the default Windows install path

The earlier `GPU available: False` training issue was caused by a CPU-only PyTorch install (`torch 2.11.0+cpu`). `pyproject.toml` and `uv.lock` now pin `torch` and `torchvision` to the PyTorch CUDA 12.8 wheel index on Windows/Linux x86_64, and `uv sync --extra dev` installs CUDA wheels.

## Data And Models

- Local dataset exists at `v1_learning/dataset/dataset.parquet` with 189 rows.
- Stratified by-shoot splits exist at `v1_learning/dataset/splits_v2_stratified/`:
  - train: 132 rows
  - val: 27 rows
  - test: 30 rows
- The previous split was imbalanced for Exposure2012: train mean ~0.212 while val/test were ~0.480/~0.504. The regenerated split reduces that gap: train mean ~0.264, val ~0.379, test ~0.318.
- Temperature labels in the current dataset mostly cool relative to AsShot WB, so a warmer model output points to training flow/initialisation/split issues rather than warmer target labels.
- Tint labels are consistently positive/magenta relative to AsShot, so some magenta tendency is present in the labels.
- `v1_learning/model-v2.0.0.ckpt` and `v1_learning/model-v2.0.0.json` are present locally and remain the active frontend profile (`~/.saha/active_profile.txt` set to `sonna-v2-run-01-v2.0.0`).
- Lite profile creation now supports using this active v2 checkpoint as its base; the builder preserves `slider_set_version="v2"` and v2 extension heads instead of forcing a v1 load.
- A fresh scene-stats candidate was trained at `data/models/sonna-v2-scene-stats-run01/`, but it was rejected for frontend use. It briefly published as `v1_learning/model-v2.0.1.*`, then those frontend-visible copies were removed after collapse analysis showed worse prediction spread than v2.0.0.

## What Changed This Session

- Fixed the Python environment so CUDA PyTorch is installed and preserved by `uv sync`.
- Updated `scripts/train_profile.py` logging:
  - default recipe values now log as `Training recipe ...`
  - explicit CLI flags log as `Override ...`
- Updated operational docs to reflect the current Windows CUDA state, small local dataset, absent local checkpoints, and updated training/resume guidance.
- Updated `AGENTS.md` and `CLAUDE.md` with a stronger required-reading rule and a documentation update rule after non-trivial work.
- Disabled photometric jitter by default in `src/sonna_editor/config.py`; training augmentation is geometry-only unless explicitly changed. This avoids corrupting Exposure/colour supervision because the XMP target sliders are for the original image, not a brightened/darkened synthetic image.
- Reworked `src/sonna_editor/data/dataset.py` split logic to balance shoot-grouped splits across Temperature correction, Exposure2012, and Tint correction, with tail coverage so rare edit styles do not concentrate in one split.
- Added fresh-model target-prior initialisation in `SonnaEditor.initialise_output_priors()` and wired it into `scripts/train_profile.py`. Fresh output heads now start at training-set slider medians, while WB residual heads start at zero when AsShot WB skip is enabled.
- Raised the default v2 Exposure2012 loss weight to 4.0. The current training-set priors are Exposure2012=0.22, Temperature=5191K, Tint=5.
- Fixed the pink/red Lightroom output issue in `src/sonna_editor/inference/pipeline.py`. Diagnosis on `0H5A3190A-2.xmp`: Temperature/Tint were close to the expected XMP, but the model-written RGB tone-curve highlight endpoints were not neutral (`Green Pt6=240/221`, `Blue Pt6=213/234`). Those endpoints colour-cast white highlights pink/red. The inference pipeline now forces RGB tone-curve black endpoints to `0/0` and white endpoints to `255/255`, while leaving mid-curve shape predictions intact.
- Added preview-derived luminance scene statistics (`mean_luminance`, `median_luminance`, `luminance_std`, `highlight_clip_pct`, `shadow_clip_pct`, `dynamic_range`) to extraction, RAW/XMP datasets, catalog datasets, fine-tune captures, training dataloaders, inference batches, and a new `arch_version=2` metadata encoder path.
- Updated default visual-priority loss weights to match the improvement plan: Exposure=5.0, Temperature/Tint=4.0, Contrast/Highlights/Shadows=3.0, Whites/Blacks/Saturation/Vibrance=2.0 minimums, while preserving existing targeted timid-field bumps above those floors.
- Added validation distribution logging for Exposure, Contrast, Highlights, Shadows, Temperature, and Tint (`val_dist_*_std_ratio`) so future training runs surface collapse during training.
- Added `scripts/analyse_prediction_collapse.py` and `scripts/audit_dataset_diversity.py`.
- Trained `data/models/sonna-v2-scene-stats-run01` on CUDA. Best checkpoint was epoch 0 with `best_val_loss=0.001177`, `test_loss=0.000994`, `test_mae_exposure=0.229`, `test_mae_temperature=115K`, `test_mae_hsl_avg=2.03`. Despite low MAE, collapse audit worsened from 14 collapsed sliders on `model-v2.0.0` to 29 collapsed sliders on this candidate, so it was not kept frontend-visible.
- Fixed frontend Lite profile creation when the active Personal AI base is v2. `src/sonna_editor/mode_b/checkpoint_builder.py` now loads the base checkpoint at its native slider set, computes preset/survey bias deltas for `v1` or `v2`, writes the matching `slider_set_version` into the Lite sidecar, and preserves v2 extension-head weights instead of calling `from_checkpoint(target_slider_set_version="v1")`.
- Removed stale v2-rejection behavior from the Mode B tests and removed an unused test import. Added focused v2 coverage proving Lite creation from a v2 base succeeds and keeps extension heads intact.
- Cleaned up noisy training console warnings. `scripts/train_profile.py` now chooses `log_every_n_steps` from the actual train-batch count so the 132-row local split uses 9 instead of tripping Lightning's default-10 warning. The training package, current trainer, legacy production trainer, and fine-tune path suppress the upstream Lightning `LeafSpec` deprecation and optional Torch Triton FLOP-counter warning.

## Verification

- `uv run python -c "import torch; ..."` confirmed:
  - `torch 2.11.0+cu128`
  - `torch.cuda.is_available() == True`
  - device: NVIDIA GeForce RTX 3050
  - CUDA tensor matmul succeeded
- `uv run python scripts\verify_environment.py` passed 11/11 checks.
- `uv run ruff check scripts\train_profile.py pyproject.toml` passed.
- `uv run ruff check src\sonna_editor\data\dataset.py src\sonna_editor\model\architecture.py src\sonna_editor\config.py scripts\train_profile.py tests\test_dataset.py tests\test_training.py` passed.
- `uv run python -m py_compile scripts\train_profile.py` passed.
- Targeted backend tests passed: `tests/api/test_profiles.py`, `tests/api/test_health.py`, `test_training_step_returns_scalar`, `test_training_step_loss_is_non_negative` (22 passed, 2 Lightning logging warnings).
- Focused training/data tests passed: `uv run pytest tests\test_dataset.py tests\test_training.py::test_output_prior_initialisation_sets_exposure_and_zero_wb_residual tests\test_training.py::test_training_step_returns_scalar tests\test_config.py` -> 59 passed, 1 skipped.
- Inference endpoint fix tests passed: `uv run pytest tests\test_inference_v2.py` -> 10 passed.
- Real sample verification: applying the new endpoint stabiliser to the bad Sonna `0H5A3190A-2.xmp` changes RGB Pt6 endpoints to `255/255` for red, green, and blue. The temporary verification file was removed after the check.
- Regenerated the affected local shoot output with `scripts\process_shoot_model.py` using `v1_learning\model-v2.0.0.ckpt`:
  - input/output: `C:\Users\vikas.DESKTOP-61LEE8B\OneDrive\Pictures\Testing_Sonna`
  - processed: 100 RAWs, failed: 0
  - wrote fresh XMPs plus `sonna_predictions.json`
  - verified `0H5A3190A-2.xmp` now has Red/Green/Blue Pt6 endpoints all at `255,255`
- `npm run build:vite` passed in `saha-app`.
- `uv run ruff check ...` over changed source/tests/scripts passed.
- `uv run python -m py_compile scripts\train_profile.py scripts\analyse_prediction_collapse.py scripts\audit_dataset_diversity.py` passed.
- `uv run pytest tests\test_training.py tests\test_dataset.py tests\test_catalog_dataset.py tests\test_architecture.py -q` passed: 134 passed, 7 skipped.
- `uv run pytest tests\test_extract.py::TestComputeSceneStatistics -q` passed: 3 passed.
- `scripts\audit_dataset_diversity.py` ran on `v1_learning\dataset\dataset.parquet`: 189 photos / 35 shoots; brightness split dark=92, balanced=81, bright=16; WB split warm=66, daylight=114, cool=9.
- `scripts\analyse_prediction_collapse.py` ran on existing `model-v2.0.0`: 27 val photos, 14 collapsed sliders; Exposure2012 std_ratio=0.115, Temperature/Tint std_ratio ~1.0.
- `scripts\analyse_prediction_collapse.py` ran on rejected scene-stats candidate: 27 val photos, 29 collapsed sliders; Exposure2012 std_ratio near zero, so the candidate was rejected despite lower test MAE.
- Dark low-light output diagnosis on `0H5A4599`: the reference/training XMP has `Exposure2012=+1.11`, while active `model-v2.0.0` writes about `+0.105`, roughly one stop too dark. Other key tone/WB sliders and tone curves are close to the reference, so this is not an XMP writer or tone-curve endpoint issue. Root cause is Exposure2012 prediction collapse: across the 189-row dataset, target Exposure std is ~0.454 but model output std is ~0.061; in the darkest luminance quartile, targets average `+0.695` while predictions average only `+0.090`.
- Full `uv run pytest tests` result: 691 passed, 12 skipped, 28 failed. Failures are concentrated in `tests/test_extract.py` and `tests/test_xmp.py` because the gitignored RAW/XMP fixtures are absent from `tests/fixtures/` (`sample.cr3`, `sample.xmp`, `sample_edit.xmp`) and two extract tests require Windows symlink privileges.
- Lite compatibility verification passed: `uv run ruff check src\sonna_editor\mode_b\checkpoint_builder.py tests\test_checkpoint_builder.py tests\api\test_profiles.py`; `uv run pytest tests\test_checkpoint_builder.py tests\api\test_profiles.py -q` -> 53 passed; `uv run python -m py_compile src\sonna_editor\mode_b\checkpoint_builder.py scripts\build_mode_b_checkpoint.py`; real smoke build from `v1_learning\model-v2.0.0.ckpt` to `%TEMP%\sonna-lite-v2-smoke\mode-b-v2-smoke.ckpt` succeeded and wrote sidecar `slider_set_version: v2`.
- Training warning cleanup verification passed: `uv run ruff check scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\training\__init__.py src\sonna_editor\finetune\retrain.py tests\test_training.py`; `uv run pytest tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset tests\test_training.py::test_training_step_returns_scalar tests\test_training.py::test_training_step_loss_is_non_negative -q` -> 3 passed; `uv run python -m py_compile scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\finetune\retrain.py src\sonna_editor\training\__init__.py`; one-epoch smoke training with `--num-workers 2 --no-publish` completed without the pasted `LeafSpec`, Triton FLOP-counter, or `log_every_n_steps` warnings.

## Current Code Behavior Notes

- `scripts/train_profile.py` default v2 recipe now uses:
  - `image_resolution=512`
  - `lr=1e-4`
  - `max_epochs=50`
  - `freeze_backbone_epochs=3`
  - `arch_version=2` for fresh models, adding six luminance scene-stat metadata inputs
  - Temperature loss weight=4.0
  - Tint loss weight=4.0
  - Exposure2012 loss weight=5.0
  - Temperature bucket loss weight=0.15
  - Tint bucket loss weight=2.0
  - Sign-wrong penalty weight=0.2
  - WB metadata skip enabled
- Fresh training defaults to target-prior output initialisation. On the current 189-row dataset, this can win validation loss while still producing collapsed predictions; always run collapse analysis before promoting a small-data candidate.
- Active `model-v2.0.0` under-brightens dark/low-light photos because its Exposure2012 head is nearly averaged. Example: `0H5A4599` mean luminance `0.126`, target `+1.11`, prediction about `+0.10`.
- Default training augmentation is now geometry-only; photometric jitter remains configurable but disabled by default.
- Training on tiny splits now logs once that it adjusted `log_every_n_steps` instead of letting Lightning warn. This is expected for the current 132-row train split, which has 9 batches at batch size 16.
- `Profile.profile_type` is already implemented in backend profile responses and frontend profile classification. `None` means legacy/current Mode A trained profile; `"mode_b_initial"` means Lite/Mode B preset-derived profile.
- Mode B/Lite checkpoints now inherit the active base checkpoint's slider set: v1 bases produce v1 Lite checkpoints, v2 bases produce v2 Lite checkpoints with the 12 extension fields retained.

## Next Suggested Step

Do not promote the scene-stats candidate from `data/models/sonna-v2-scene-stats-run01`; it is a useful diagnostic run, not a better model. The next real model-quality step is more data: rebuild from a substantially larger edited dataset (at least 1,000 pairs, preferably 2,000-5,000+) and rerun training plus collapse analysis. On this 189-photo dataset, the median-prior baseline is too competitive and encourages parameter averaging.

Before relying on full-suite green status, restore the local fixture files under `tests/fixtures/` or mark those fixture-dependent tests as integration/local-data tests, then rerun `uv run pytest tests`.
