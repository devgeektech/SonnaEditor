# Session State - Sonna Editor

**Saved:** 2026-06-03 local time
**Current phase/task:** Foundation model and Lite profile architecture cleanup.

## Current Workspace

- Repo path: `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`
- Branch: `main`, tracking `origin/main`
- Recent committed history in this checkout: `aac360a feat: Enhance dataset building and training scripts` on top of four earlier commits.
- Training/profile caches were intentionally cleared for a fresh dataset reset.
- Cleared repo-local generated artifacts: `v1_learning\dataset`, `data\models`, `data\parquet`, `data\captures`, `data\thumbnails`, `data\audits`, `data\dbg`, `data\raw\sonna_training`, `.pytest_cache`, `.ruff_cache`.
- Cleared frontend active-profile pointer: `.saha\active_profile.txt`.
- `v1_learning\` is currently empty; there is no guaranteed frontend-visible profile until a fresh Personal AI profile is trained or a checkpoint is intentionally published.

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

- Local training dataset was cleared. Add a fresh RAW+XMP dataset before training.
- No current train/val/test split exists in `v1_learning\dataset`.
- The previous split was imbalanced for Exposure2012: train mean ~0.212 while val/test were ~0.480/~0.504. The regenerated split reduces that gap: train mean ~0.264, val ~0.379, test ~0.318.
- Temperature labels in the current dataset mostly cool relative to AsShot WB, so a warmer model output points to training flow/initialisation/split issues rather than warmer target labels.
- Tint labels are consistently positive/magenta relative to AsShot, so some magenta tendency is present in the labels.
- No `v1_learning/model-v*.ckpt` profile is present after the reset.
- Lite profile creation now uses the configured foundation checkpoint as its base; the builder preserves the foundation checkpoint's native slider set and writes a profile sidecar that lets initial Mode B processing use preset+survey style plus adaptive per-photo Exposure/WB correction.
- A fresh scene-stats candidate was trained at `data/models/sonna-v2-scene-stats-run01/`, but it was rejected for frontend use. It briefly published as `v1_learning/model-v2.0.1.*`, then those frontend-visible copies were removed after collapse analysis showed worse prediction spread than v2.0.0.

## What Changed This Session

- Moved transient app state to repo-local `.saha\` instead of `~\.saha\`. Active profile, recent folders, queued job snapshots, Personal AI training workspaces, and fine-tune scratch runs now resolve from the project root by default.
- Added `config.ensure_runtime_directories()` and wired it into backend/server and CLI entrypoints so a fresh clone auto-creates `data\training_sources\`, `data\raw\`, `data\raw\sonna_training\`, `data\datasets\`, `data\dng\`, `data\parquet\`, `data\captures\`, `data\audits\`, `data\dbg\`, `v1_learning\`, and `.saha\` before use.
- Split local learning inputs from generated outputs. Source photos now belong under separate gitignored child folders such as `data\training_sources\sonna_personal_001\raw_xmp\`; future FiveK image-pair inputs should use folders such as `data\training_sources\fivek_expert_c\raw_dng\` and `data\training_sources\fivek_expert_c\expert_tiff\`. Generated Parquet/checkpoint outputs remain under `data\training_workspace\` and `data\foundation_repo\`.
- Moved the default training workspace and hidden foundation repo into the project tree as well. `SONNA_TRAINING_WORKSPACE` now defaults to `data\training_workspace\`, and `SONNA_FOUNDATION_REPO` defaults to `data\foundation_repo\`. Both are auto-created on startup unless the operator overrides them.
- Updated `scripts\process_shoot_model.py` so the default model path resolves to the newest published `v1_learning\model-v*.ckpt` instead of a stale hardcoded legacy checkpoint path. If no published profile exists yet, the CLI now fails with a clear instruction.

- Decoupled Lite profile creation from active Personal AI profiles. `POST /api/profiles/lite` now resolves the configured foundation checkpoint and passes that to the Lite checkpoint builder.
- Added `src/sonna_editor/foundation.py` helpers for creating the separate foundation repo layout, resolving the active foundation checkpoint from env/manifest/fallback, and promoting trained checkpoints into that repo.
- Added `scripts/train_foundation_model.py`, the canonical foundation-training command. It can build a RAW+XMP dataset or use existing splits, trains with the current recipe without publishing to the frontend, then promotes the final checkpoint into the configured foundation repo.
- Updated API tests so Lite creation proves it uses the foundation checkpoint even when no Personal AI profile is active.
- Added `tests/test_foundation.py` for foundation repo layout, manifest writing, promotion, and checkpoint resolution.
- Updated runbooks to clarify that RAW+XMP data preparation, Personal AI training, foundation training, and Lite profile creation are separate workflows. Raw training photos should live outside the app repo; the foundation checkpoint lives in a separate Git/LFS-ready foundation repo.
- Cleaned operator-facing docs to avoid making v1/v2 a user decision. Internal legacy checkpoint compatibility remains in code.
- Added `MAC_SETUP.md`, a Mac-specific setup and operating guide covering clean machine setup, uv/Python dependency sync, MPS verification, backend/frontend startup, profile discovery, RAW+XMP and catalog dataset preparation, Personal AI training, hidden foundation training, Lite profile creation, shoot processing, fine-tuning, diagnostics, and update steps.
- Added a README pointer to `MAC_SETUP.md`.
- Updated `project_knowledge.md` so the Mac runbook is part of the documented source map and current notes.
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
- Added preview-derived luminance scene statistics (`mean_luminance`, `median_luminance`, `luminance_std`, `highlight_clip_pct`, `shadow_clip_pct`, `dynamic_range`) to extraction, RAW/XMP datasets, catalog datasets, fine-tune captures, training dataloaders, inference batches, and the scene-stat metadata encoder path.
- Updated default visual-priority loss weights to match the improvement plan: Exposure=5.0, Temperature/Tint=4.0, Contrast/Highlights/Shadows=3.0, Whites/Blacks/Saturation/Vibrance=2.0 minimums, while preserving existing targeted timid-field bumps above those floors.
- Added validation distribution logging for Exposure, Contrast, Highlights, Shadows, Temperature, and Tint (`val_dist_*_std_ratio`) so future training runs surface collapse during training.
- Added `scripts/analyse_prediction_collapse.py` and `scripts/audit_dataset_diversity.py`.
- Trained `data/models/sonna-v2-scene-stats-run01` on CUDA. Best checkpoint was epoch 0 with `best_val_loss=0.001177`, `test_loss=0.000994`, `test_mae_exposure=0.229`, `test_mae_temperature=115K`, `test_mae_hsl_avg=2.03`. Despite low MAE, collapse audit worsened from 14 collapsed sliders on `model-v2.0.0` to 29 collapsed sliders on this candidate, so it was not kept frontend-visible.
- Fixed frontend Lite profile creation compatibility with current checkpoints. `src/sonna_editor/mode_b/checkpoint_builder.py` now loads the base checkpoint at its native slider set, writes the matching internal `slider_set_version` into the Lite sidecar, and preserves extension-head weights instead of down-converting.
- Removed stale v2-rejection behavior from the Mode B tests and removed an unused test import. Added focused v2 coverage proving Lite creation from a v2 base succeeds and keeps extension heads intact.
- Cleaned up noisy training console warnings. `scripts/train_profile.py` now chooses `log_every_n_steps` from the actual train-batch count so the 132-row local split uses 9 instead of tripping Lightning's default-10 warning. The training package, current trainer, legacy production trainer, and fine-tune path suppress the upstream Lightning `LeafSpec` deprecation and optional Torch Triton FLOP-counter warning.
- Fixed `scripts/quick_diagnostic.py` so old training summaries that do not embed row counts still print dataset split row counts. It now checks nested summary fields first, then summary parquet path fields if present, then falls back to the canonical `v1_learning/dataset/splits_v2_stratified/*.parquet` metadata. It also replaced emoji status markers with ASCII `OK`/`BAD` labels so the script finishes cleanly in Windows PowerShell.
- Fixed the Mode B/Lite overexposure source in the profile builder. `src/sonna_editor/mode_b/checkpoint_builder.py` now zeroes each output head's final linear weights and copies absolute preset+survey targets into final biases, so the profile carrier does not add the trained base model's own Exposure/colour predictions on top of the preset. For v2 bases, the direct `wb_metadata_skip` route is zeroed in Mode B initial checkpoints so AsShot WB is not secretly stacked on top of preset Temperature/Tint.
- Fixed the actual UI/CLI Mode B processing flow. `src/sonna_editor/inference/pipeline.py` now detects checkpoint sidecars with `profile_type: "mode_b_initial"`, bypasses `InferenceEngine` for the initial Lite run, reloads the copied preset and survey, computes per-photo Exposure/WB corrections only through `sonna_editor.preset.adjuster`, writes those adjusted values to XMP, and records the adjusted baseline in `sonna_predictions.json`.
- Fixed the second root cause in `src/sonna_editor/preset/adjuster.py`: the old auto-exposure heuristic used mean luminance only. On real event frames with black suits/dark rooms plus bright faces/signage, it could add too much positive Exposure. The new heuristic still uses mean luminance but guards it with 85th/95th percentile luminance targets so already-bright upper tones prevent over-lifting.
- Published a corrected Lite profile at `v1_learning/model-v0.2.0.ckpt` / `.json` from the same preset and survey as stale `model-v0.1.0`. Its profile id is `k-fixed-lite-20260602-2120`. Copied support files to `model-v0.2.0-preset.xmp` and `model-v0.2.0-survey.json`, updated the v0.2 sidecar to point to them, and removed stale `model-v0.1.0.*` artifacts.
- Updated Mode B tests to pin both the checkpoint builder behavior and the adaptive initial processing path. Removed stale assertions/comments that expected inherited final-layer head weights and an unused test local.
- Added frontend Personal AI profile creation: the profile page now opens a RAW+XMP training wizard, starts `POST /api/profiles/personal`, streams epoch progress through the existing job hook, supports cancel, and refreshes profiles after completion.
- Lite profile creation now presents the six-question style survey in the UI. All six answers are stored in the survey JSON and profile package; initial Lite processing still dynamically adjusts only Exposure, Temperature, and Tint so preset look sliders stay fixed until fine-tuning.
- Added a fresh-model staged-head learning path. `SonnaEditor` now defaults to `arch_version=3`, where WB/presence heads condition on the tone block output and later heads condition on tone + presence + WB outputs. Existing `arch_version=1`/`2` checkpoints load unchanged.
- Added profile-delete confirmation in the frontend before removing any local checkpoint/sidecar files. Active profile deletion remains blocked by the backend.
- Moved the training callable into `src/sonna_editor/training/profile_runner.py`; `scripts/train_profile.py` is now a thin CLI wrapper. The API imports the packaged runner rather than a script module.
- Added cancellation handling to frontend Personal AI training so cancelled runs do not test/save/publish a new checkpoint.
- Updated docs so Personal AI and Lite training/execution are frontend flows, while foundation training is CLI-only through `scripts/train_foundation_model.py`.
- Renamed `TRAINING_COMMANDS.md` to `CLI_COMMANDS.md` and created `FOUNDATION_TRAINING.md` for foundation train, resume, retrain, promotion, and FiveK guidance.
- Updated frontend Personal AI backend flow so RAW+XMP profile training resolves the configured hidden foundation checkpoint and warm-starts model weights from it before publishing a user-facing checkpoint into `v1_learning/`. Warm-start now uses `base_model_checkpoint` rather than Lightning resume state, keeps the new training registry, and skips categorical embedding-table copies.
- Verified MIT-Adobe FiveK is suitable foundation material, with the caveat that it is 5,000 DNG inputs plus five expert renditions/catalog edits, not 25,000 independent RAW inputs. Use one expert target style first.
- Implemented the TIFF/image foundation path. `scripts\train_foundation_model.py` now accepts `--raw-image-dir` plus `--target-tiff-dir` for paired `RAW/DNG/image -> edited TIFF` training, saving an `image_to_image_v1` checkpoint in the foundation repo. Personal AI warm-start and Lite profile creation can copy the image-foundation ConvNeXt backbone into a fresh `SonnaEditor`; Mode A remains RAW+XMP slider regression.
- Updated foundation training semantics so each new foundation run warm-starts from the active foundation checkpoint by default, writes a new versioned checkpoint, promotes it as the active default, and keeps previous checkpoints untouched. If the active checkpoint file is removed after a bad run, resolution falls back to the newest remaining checkpoint in the foundation repo.
- Adjusted Lite/preset auto-exposure for low-light images. Dark scenes with no near-clipped highlights now receive a stronger positive exposure floor, so event frames like the sofa/table example lift closer to the Imagen-style reference instead of staying slightly underexposed. WB remains conservative by default.
- Guarded Lightning metric logging in `src/sonna_editor/training/module.py` so standalone `training_step()` unit tests no longer emit `self.log()` warnings without a Trainer.

## Verification

- `uv run ruff check src\sonna_editor\foundation.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\api\models.py scripts\train_foundation_model.py scripts\build_mode_b_checkpoint.py tests\test_foundation.py tests\api\test_profiles.py tests\api\conftest.py` passed.
- `uv run python -m py_compile src\sonna_editor\foundation.py src\sonna_editor\api\routes\profiles.py scripts\train_foundation_model.py scripts\build_mode_b_checkpoint.py` passed.
- `uv run pytest tests\test_foundation.py tests\api\test_profiles.py -q` passed: 22 passed.
- `uv run ruff check src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\api\test_callback_bridge.py tests\test_inference_pipeline_integration.py` passed.
- `uv run python -m py_compile src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py` passed.
- `uv run pytest tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\test_architecture.py tests\api\test_callback_bridge.py::test_mode_b_initial_uses_per_photo_preset_adjuster tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` passed: 143 passed, 6 skipped.
- `npm run build:vite` passed in `saha-app/`.
- Cleanup verification after foundation/Mode A/Mode B updates:
  - `uv run ruff check .` passed.
  - `uv run python -m py_compile src\sonna_editor\model\architecture.py src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\training\module.py scripts\train_foundation_model.py` passed.
  - `npm run build:vite` passed in `saha-app/`.
  - `uv run pytest tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\test_architecture.py tests\api\test_callback_bridge.py::test_mode_b_initial_uses_per_photo_preset_adjuster tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation tests\test_training.py::test_foundation_warm_start_keeps_training_registry tests\test_training.py::test_training_step_returns_scalar tests\test_training.py::test_training_step_loss_is_non_negative tests\test_training.py::test_loss_gradient_flow_to_predictions tests\test_training.py::test_output_prior_initialisation_sets_exposure_and_zero_wb_residual -q` passed: 148 passed, 6 skipped, no warnings.
  - `uv run mypy src scripts` is still not clean in this workspace; it reports broad pre-existing strict-typing debt and missing third-party stubs across many files. Treat that as a dedicated type-hardening task, separate from lint/compile/test cleanup.
- Reviewed existing `HANDOVER.md`, `SESSION_STATE.md`, `project_knowledge.md`, `SONNA_EDITOR_BUILD_SPEC.md`, `CLI_COMMANDS.md`, `RUN.md`, `README.md`, `pyproject.toml`, and `saha-app/package.json` before writing the Mac guide.
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
- Quick diagnostic row-count verification passed: `uv run ruff check scripts\quick_diagnostic.py`; `uv run python -m py_compile scripts\quick_diagnostic.py`; `uv run python scripts\quick_diagnostic.py --summary-path data\models\sonna-v2-run01\training_summary.json` now prints train=132, val=27, test=30 and completes successfully.
- Mode B preset-faithful verification passed:
  - `uv run ruff check src\sonna_editor\mode_b\checkpoint_builder.py tests\test_checkpoint_builder.py`
  - `uv run pytest tests\test_checkpoint_builder.py -q` -> 34 passed
  - `uv run pytest tests\api\test_profiles.py tests\api\test_process_route.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 31 passed
  - `uv run python -m py_compile src\sonna_editor\mode_b\checkpoint_builder.py scripts\build_mode_b_checkpoint.py`
  - Active-v2 smoke build from `v1_learning\model-v2.0.0.ckpt` using `tests\fixtures\preset_sonna_v1.xmp` now outputs the preset values exactly on synthetic dark and bright inputs: `Exposure2012=+0.35`, `Contrast2012=15`, `Highlights2012=-45`, `Shadows2012=30`, `Temperature≈5200`, `Tint=-3`, `Saturation=5`, `Vibrance=12`.
  - `npm run build:vite` passed in `saha-app`.
- Mode B adaptive processing verification passed:
  - `uv run ruff check src\sonna_editor\inference\pipeline.py tests\api\test_callback_bridge.py`
  - `uv run python -m py_compile src\sonna_editor\inference\pipeline.py`
  - `uv run pytest tests\api\test_callback_bridge.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 13 passed
  - New coverage proves the initial Mode B UI processing path does not load `InferenceEngine`, produces different Exposure values for dark vs bright previews, applies grey-world WB from the photo baseline when the preset omits Temperature/Tint, preserves fixed preset style fields, and records the adjusted per-photo values in `sonna_predictions.json`.
- Mode B root-cause verification on the real UI folder passed:
  - Stale active profile `v1_learning/model-v0.1.0.json` had old notes: final-layer weights inherited from base checkpoint and biases shifted by preset+survey deltas. This is the base-model + preset double-apply path.
  - Existing bad XMP example `OneDrive\Pictures\Test_Sonna - Copy\0H5A9030-3.xmp` had model-stacked values: `Exposure2012=+1.409`, `Contrast2012=-9.73`, `Highlights2012=-83.14`, `Shadows2012=86.48`, `Whites2012=-71.68`, `Blacks2012=11`.
  - Current code with corrected `v1_learning/model-v0.2.0.ckpt` processed all 260 RAWs from `OneDrive\Pictures\Test_Sonna - Copy` to `%TEMP%\sonna-modeb-v020-real-output`: 260 processed, 0 failed.
  - Corrected real output for screenshot file `0H5A0100-6.xmp`: `Exposure2012=-0.329`, `Contrast2012=-2`, `Highlights2012=-53`, `Shadows2012=60`, `Whites2012=-52`, `Blacks2012=-2`, `Temperature=3633`, `Tint=5.7`, `Saturation=7`, `Vibrance=-7`.
  - Corrected real output for prior high-exposure sample `0H5A9030-3.xmp`: `Exposure2012=+0.780` with fixed preset style values, not base-model tone values.
- Focused suite after the adjuster fix: `uv run pytest tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\api\test_process_route.py tests\api\test_callback_bridge.py tests\test_pipeline.py tests\test_adjuster.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 121 passed, 1 skipped.
- Production profile-flow verification passed:
  - `uv run ruff check .` -> passed
  - `npm run build:vite` in `saha-app` -> passed
  - `uv run python -m py_compile scripts\train_profile.py src\sonna_editor\training\profile_runner.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\api\jobs.py src\sonna_editor\api\models.py` -> passed
  - `uv run pytest tests\api\test_profiles.py tests\api\test_callback_bridge.py tests\test_adjuster.py tests\test_checkpoint_builder.py -q` -> 101 passed
  - `uv run pytest tests\test_config.py::TestSliderLossWeights::test_c3k_tuned_weights tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset -q` -> 2 passed
  - Full `uv run pytest tests -q` -> 705 passed, 12 skipped, 28 failed. Remaining failures are fixture/environment issues: `tests/fixtures/sample_edit.xmp` is missing, `tests/fixtures/sample.cr3` is unreadable by rawpy on this machine, and two tests require Windows symlink privileges.

## Current Code Behavior Notes

- `scripts/train_profile.py` default v2 recipe now uses:
  - `image_resolution=512`
  - `lr=1e-4`
  - `max_epochs=50`
  - `freeze_backbone_epochs=3`
  - `arch_version=3` for fresh models, adding staged output-head conditioning on top of the six luminance scene-stat metadata inputs
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
- `Profile.profile_type` is already implemented in backend profile responses and frontend profile classification. `None` means a legacy trained profile; `"mode_b_initial"` means a Lite preset-derived profile.
- Mode B/Lite checkpoints now inherit the configured foundation checkpoint's native slider set and field count. Before fine-tuning, the UI/CLI processing path treats `mode_b_initial` as an Imagen-aligned Lite profile: preset look fixed, per-photo Exposure/WB corrections only. After fine-tuning, the same profile can move back to normal model inference.

## Next Suggested Step

Add the fresh RAW+XMP dataset, start the backend/Electron app, and create a Personal AI profile from the frontend. Use Lite profile creation from the frontend after a foundation checkpoint is configured in `SONNA_FOUNDATION_CHECKPOINT`, `SONNA_FOUNDATION_REPO/foundation_manifest.json`, or `SONNA_FOUNDATION_REPO/foundation.ckpt`.

For current parameter-supervised foundation model work, use `scripts\train_foundation_model.py --raw-xmp-dir ...` or `--splits-dir ...` so the checkpoint is promoted to the separate foundation repo and stays out of the frontend profile list. It will warm-start from the active foundation checkpoint unless `--no-warm-start` is supplied. For MIT-Adobe FiveK, use `scripts\train_foundation_model.py --raw-image-dir ... --target-tiff-dir ...`; do not push FiveK TIFF targets through the existing RAW+XMP pipeline.

Before relying on full-suite green status, restore the local fixture files under `tests/fixtures/` or mark those fixture-dependent tests as integration/local-data tests, then rerun `uv run pytest tests`.
