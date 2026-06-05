# Session State - Sonna Editor

**Saved:** 2026-06-05 local time
**Current phase/task:** FiveK catalog foundation dataset support and runbook cleanup.

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

- Reviewed the local MIT-Adobe FiveK download at `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset`.
- Confirmed the extracted FiveK tree contains 5,000 `.dng` source files under `raw_photos\HQa*`, `raw_photos\fivek.lrcat`, Lightroom preview/helper/catalog-data folders, and text/license/category files.
- Inspected `raw_photos\fivek.lrcat` read-only through SQLite. It has 60,000 `Adobe_images` rows, 5,000 unique source DNGs, 96,458 develop-settings rows, and 60,000 active non-empty develop-setting blobs. Each DNG has 12 virtual-copy/recipe variants.
- Confirmed expert collections `A`, `B`, `C`, `D`, and `E` each have 5,000 linked rows. Collection `C` has 5,000 rows, all with develop settings and all pointing to existing DNGs.
- Added exact Lightroom collection filtering to `src\sonna_editor\data\catalog.py` and surfaced it through `scripts\build_dataset_from_catalog.py --collection-name`.
- Added `--include-unedited-looking` to `scripts\build_dataset_from_catalog.py` because FiveK develop blobs are sparse; many default sliders are absent, so the generic unedited-row filter would skip valid expert edits.
- Verified the project catalog reader can parse Collection `C`. A 250-row sample had about 55 populated slider fields per row; absent sliders remain `NaN` and are masked by the existing loss.
- Ran a 20-row real FiveK Collection C catalog smoke into `data\training_workspace\fivek_expert_c_catalog_smoke_nosplit`; dataset creation succeeded. A separate 20-row smoke with `--split` wrote the dataset but failed splitting because the tiny sample had too few shoot groups, so use full 5,000-row builds for real split generation.
- Updated `FOUNDATION_TRAINING.md`, `CLI_COMMANDS.md`, and `RUN.md` with the inspected FiveK path, catalog row analysis, Expert C catalog commands, cumulative foundation training order, and checkpoint naming instructions.
- Updated `project_knowledge.md` and `HANDOVER.md` with the FiveK catalog route and source-map changes.
- Upgraded foundation lineage management to schema-v2 manifests. `foundation_manifest.json` now records `active_version`, `versions[]`, checkpoint SHA256, foundation type, capabilities, and training-source tags while retaining compatibility fields such as `active_checkpoint` and `history`.
- Foundation promotion now auto-allocates `foundation-vN` when no explicit version stem is supplied, still refuses to overwrite existing checkpoints, and keeps older versions available for rollback.
- Added `scripts\rollback_foundation.py` with `--list` and explicit version activation so bad foundation runs can be rolled back by changing the manifest pointer instead of deleting checkpoint files.
- Personal AI training sidecars and summaries now record foundation provenance for foundation warm starts: version, checkpoint path, SHA256, foundation type, capabilities, and training-source tags.
- Foundation warm starts now use native SonnaEditor checkpoints only. The previous paired-image warm-start path has been removed.
- Added progressive backbone warm-start training. Frontend Personal AI and RAW+XMP foundation runs now freeze the full ConvNeXt backbone first, then unfreeze later stages in phases before full fine-tuning. The legacy partial strategy remains available as `--backbone-unfreeze-strategy partial`.
- Added `scripts\analyse_backbone_drift.py` to compare ConvNeXt backbone tensor drift between a foundation checkpoint and a final Personal AI checkpoint, reporting per-stage relative deltas, cosine similarity, and the largest drifting tensors.
- Removed the paired-image foundation training path. Foundation checkpoints are now native `SonnaEditor` slider-regression checkpoints trained from catalog-derived splits or RAW+XMP sidecars.
- Moved transient app state to repo-local `.saha\` instead of `~\.saha\`. Active profile, recent folders, queued job snapshots, Personal AI training workspaces, and fine-tune scratch runs now resolve from the project root by default.
- Added `config.ensure_runtime_directories()` and wired it into backend/server and CLI entrypoints so a fresh clone auto-creates `data\training_sources\`, `data\raw\`, `data\raw\sonna_training\`, `data\datasets\`, `data\dng\`, `data\parquet\`, `data\captures\`, `data\audits\`, `data\dbg\`, `v1_learning\`, and `.saha\` before use.
- Split local learning inputs from generated outputs. Source photos now belong under separate gitignored child folders such as `data\training_sources\sonna_personal_001\raw_xmp\`. Generated Parquet/checkpoint run outputs, including FiveK catalog datasets, remain under `data\training_workspace\`.
- `SONNA_TRAINING_WORKSPACE` defaults to `data\training_workspace\`. `SONNA_FOUNDATION_REPO` defaults to the repo-local `SonnaEditorFoundation\` folder so promoted foundation checkpoints stay inside the SonnaEditor workspace while remaining outside gitignored `data\`. Runtime directories are auto-created on startup unless the operator overrides them.
- Updated `scripts\process_shoot_model.py` so the default model path resolves to the newest published `v1_learning\model-v*.ckpt` instead of a stale hardcoded legacy checkpoint path. If no published profile exists yet, the CLI now fails with a clear instruction.

- Decoupled Lite profile creation from active Personal AI profiles. `POST /api/profiles/lite` now resolves the configured foundation checkpoint and passes that to the Lite checkpoint builder.
- Added `src/sonna_editor/foundation.py` helpers for creating the hidden foundation folder layout, resolving the active foundation checkpoint from env/manifest/fallback, and promoting trained checkpoints into that folder.
- Added `scripts/train_foundation_model.py`, the canonical foundation-training command. It can build a RAW+XMP dataset or use existing splits, trains with the current recipe without publishing to the frontend, then promotes the final checkpoint into the configured foundation folder.
- Updated API tests so Lite creation proves it uses the foundation checkpoint even when no Personal AI profile is active.
- Added `tests/test_foundation.py` for foundation folder layout, manifest writing, promotion, and checkpoint resolution.
- Updated runbooks to clarify that RAW+XMP data preparation, Personal AI training, foundation training, and Lite profile creation are separate workflows. Raw training photos should live in repo-local gitignored source folders by default, or in explicitly chosen external folders; the foundation checkpoint lives in `SonnaEditorFoundation/`.
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
- Verified MIT-Adobe FiveK is suitable foundation material through its Lightroom catalog, with the caveat that it is 5,000 DNG inputs plus expert catalog edit variants, not 25,000 independent RAW inputs. Use one expert collection first.
- Removed the previous paired-image foundation implementation and commands. `scripts\train_foundation_model.py` now accepts only `--raw-xmp-dir` or `--splits-dir`.
- Updated foundation training semantics so each new foundation run warm-starts from the active foundation checkpoint by default, writes a new versioned checkpoint, promotes it as the active default, and keeps previous checkpoints untouched. If the active checkpoint file is removed after a bad run, resolution falls back to the newest remaining checkpoint in the foundation folder.
- Updated the foundation and operator docs so RAW+XMP foundation training uses direct script commands only: Lightroom metadata export/source-folder expectation, inspectable dataset/split build, audits, training from prepared splits, and the direct `--raw-xmp-dir` shortcut. Operator-facing stale references to repo-local `data\foundation_repo\`, old local dataset presence, and old Lite profile artifacts were cleaned up.
- Added `matplotlib` to the base project dependencies and refreshed `uv.lock` so dataset audit plots generate without optional-import warnings. `scripts\audit_catalog.py` now prints ASCII `OK`/`WARN`/`STOP` status labels so Windows PowerShell does not fail on emoji encoding.
- Fixed foundation training CUDA OOM handling. `scripts\train_foundation_model.py` now defaults to `--batch-size 8` for foundation runs and the RAW+XMP slider-regression path automatically catches CUDA memory failures, clears the CUDA cache, and retries with halved batch sizes. Foundation docs now recommend batch 8 on the Windows RTX 3050 workstation.
- Adjusted Lite/preset auto-exposure for low-light images. Dark scenes with no near-clipped highlights now receive a stronger positive exposure floor, so event frames like the sofa/table example lift closer to the Imagen-style reference instead of staying slightly underexposed. WB remains conservative by default.
- Guarded Lightning metric logging in `src/sonna_editor/training/module.py` so standalone `training_step()` unit tests no longer emit `self.log()` warnings without a Trainer.
- Corrected the foundation workspace boundary so `SonnaEditor` is the parent folder. The default foundation folder is now `SonnaEditor\SonnaEditorFoundation\` instead of the sibling `Projects\SonnaEditorFoundation\`. The folder is tracked by the parent repo, with checkpoint binaries routed through Git LFS.
- Removed the nested `SonnaEditorFoundation\.git` metadata so the parent `SonnaEditor` repo can track the foundation folder directly. Added parent `.gitattributes` rules for `SonnaEditorFoundation/checkpoints/*.ckpt`, `v1_learning/*.ckpt`, and `models/**/*.ckpt` through Git LFS.
- Removed the obsolete `--init-git` foundation CLI option so future foundation runs cannot recreate a nested Git repository inside `SonnaEditorFoundation\`.
- Cleaned fixture-dependent RAW/XMP tests so missing/unreadable private fixtures skip cleanly instead of failing the full suite on this Windows workspace.

## Verification

- `uv run ruff check src\sonna_editor\foundation.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\api\models.py scripts\train_foundation_model.py scripts\build_mode_b_checkpoint.py tests\test_foundation.py tests\api\test_profiles.py tests\api\conftest.py` passed.
- Current repo-local foundation/Git LFS verification:
  - `git check-attr filter diff merge -- SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-001.ckpt` reports `filter: lfs`, `diff: lfs`, and `merge: lfs`.
  - `Test-Path SonnaEditorFoundation\.git` returned `False`.
  - `git add --dry-run .gitattributes SonnaEditorFoundation` lists the foundation manifest, sidecar, README, and active checkpoint as addable by the parent repo.
- Current cleanup verification:
  - `uv run ruff check .` passed.
  - `uv run python -m compileall -q src scripts tests` passed.
  - `uv run mypy src\sonna_editor\foundation.py scripts\train_foundation_model.py` passed.
  - `npm run build:vite` passed in `saha-app\`.
  - `uv run pytest tests\test_foundation.py tests\test_train_foundation_model.py tests\test_config.py tests\api\test_profiles.py tests\test_checkpoint_builder.py -q` passed: 88 passed.
  - `uv run pytest tests\test_extract.py tests\test_xmp.py -q` passed with local fixture skips: 71 passed, 34 skipped.
  - `uv run pytest tests -q --ignore=tests/test_extract.py --ignore=tests/test_xmp.py` passed: 653 passed, 11 skipped, 1 warning.
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
- Documentation-only RAW+XMP foundation runbook pass completed on 2026-06-03. Reviewed `FOUNDATION_TRAINING.md`, `CLI_COMMANDS.md`, `RUN.md`, `README.md`, `MAC_SETUP.md`, `HANDOVER.md`, `SESSION_STATE.md`, and `project_knowledge.md`; no code tests were required for this docs-only update.
- Dataset audit warning fix verification:
  - `uv lock` added `matplotlib` and plotting dependencies.
  - `uv sync --extra dev` installed `matplotlib==3.10.9`.
  - `uv run python scripts\audit_catalog.py --parquet-path data\training_workspace\sonna_foundation_001_dataset\dataset.parquet --output-dir data\training_workspace\sonna_foundation_001_dataset\audit` completed with no missing-matplotlib warnings and no PowerShell encoding error; status remained `WARN` because the 189-photo dataset is small and has 40 outlier sliders.
  - `uv run ruff check scripts\audit_catalog.py pyproject.toml` passed.
  - `uv run pytest tests\test_audit.py -q` passed: 26 passed.
- Foundation CUDA OOM fix verification:
  - `uv run ruff check scripts\train_foundation_model.py tests\test_train_foundation_model.py` passed.
  - `uv run pytest tests\test_train_foundation_model.py -q` passed: 5 passed.
  - `uv run python -m py_compile scripts\train_foundation_model.py` passed.
  - One-epoch smoke completed on CUDA with `--batch-size 8 --workers 0 --no-warm-start`, using `data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified` and disposable foundation folder `data\tmp_foundation_oom_smoke_repo`. It promoted `data\tmp_foundation_oom_smoke_repo\checkpoints\foundation-oom-smoke.ckpt`; the real foundation folder was not touched.
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
- Fixture-dependent RAW/XMP tests now skip cleanly when local private fixtures are absent or unreadable. `tests\test_extract.py tests\test_xmp.py` passes with local fixture skips instead of failing on missing `sample_edit.xmp`, unreadable `sample.cr3`, or Windows symlink privileges.
- Lite compatibility verification passed: `uv run ruff check src\sonna_editor\mode_b\checkpoint_builder.py tests\test_checkpoint_builder.py tests\api\test_profiles.py`; `uv run pytest tests\test_checkpoint_builder.py tests\api\test_profiles.py -q` -> 53 passed; `uv run python -m py_compile src\sonna_editor\mode_b\checkpoint_builder.py scripts\build_mode_b_checkpoint.py`; real smoke build from `v1_learning\model-v2.0.0.ckpt` to `%TEMP%\sonna-lite-v2-smoke\mode-b-v2-smoke.ckpt` succeeded and wrote sidecar `slider_set_version: v2`.
- Training warning cleanup verification passed: `uv run ruff check scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\training\__init__.py src\sonna_editor\finetune\retrain.py tests\test_training.py`; `uv run pytest tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset tests\test_training.py::test_training_step_returns_scalar tests\test_training.py::test_training_step_loss_is_non_negative -q` -> 3 passed; `uv run python -m py_compile scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\finetune\retrain.py src\sonna_editor\training\__init__.py`; one-epoch smoke training with `--num-workers 2 --no-publish` completed without the pasted `LeafSpec`, Triton FLOP-counter, or `log_every_n_steps` warnings.
- Quick diagnostic row-count verification passed: `uv run ruff check scripts\quick_diagnostic.py`; `uv run python -m py_compile scripts\quick_diagnostic.py`; `uv run python scripts\quick_diagnostic.py --summary-path data\models\sonna-v2-run01\training_summary.json` now prints train=132, val=27, test=30 and completes successfully.
- Quick diagnostic output clarity was improved. `scripts\quick_diagnostic.py` now prints a recommended-score column, uses green-circle `OK` pass statuses, treats missing `published_model` as normal for foundation runs, and prints next steps that distinguish hidden foundation validation from frontend Personal AI publishing.
- Quick diagnostic clarity verification passed: `uv run ruff check scripts\quick_diagnostic.py`; `uv run python scripts\quick_diagnostic.py --summary-path data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-003\training\training_summary.json`.
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
  - Historical note: full `uv run pytest tests -q` used to fail when local RAW/XMP fixtures were absent or unreadable. The fixture-dependent tests now skip cleanly in that state.

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
- Fresh training defaults to target-prior output initialisation. On the previous 189-row diagnostic dataset, this could win validation loss while still producing collapsed predictions; always run collapse analysis before promoting a small-data candidate.
- The previous `model-v2.0.0` diagnostic profile under-brightened dark/low-light photos because its Exposure2012 head was nearly averaged. Example: `0H5A4599` mean luminance `0.126`, target `+1.11`, prediction about `+0.10`.
- Default training augmentation is now geometry-only; photometric jitter remains configurable but disabled by default.
- Training on tiny splits now logs once that it adjusted `log_every_n_steps` instead of letting Lightning warn. This is expected for the current 132-row train split, which has 9 batches at batch size 16.
- `Profile.profile_type` is already implemented in backend profile responses and frontend profile classification. `None` means a legacy trained profile; `"mode_b_initial"` means a Lite preset-derived profile.
- Mode B/Lite checkpoints now inherit the configured foundation checkpoint's native slider set and field count. Before fine-tuning, the UI/CLI processing path treats `mode_b_initial` as an Imagen-aligned Lite profile: preset look fixed, per-photo Exposure/WB corrections only. After fine-tuning, the same profile can move back to normal model inference.

## Next Suggested Step

For immediate FiveK foundation training, build the Expert C catalog dataset with `scripts\build_dataset_from_catalog.py --collection-name "C" --include-unedited-looking`, audit it, then train from prepared splits with `scripts\train_foundation_model.py --splits-dir ...`. Do not mix all 60,000 FiveK virtual-copy rows in one unconditioned model.

Add the fresh RAW+XMP dataset, start the backend/Electron app, and create a Personal AI profile from the frontend. Use Lite profile creation from the frontend after a foundation checkpoint is configured in `SONNA_FOUNDATION_CHECKPOINT`, `SONNA_FOUNDATION_REPO/foundation_manifest.json`, or `SONNA_FOUNDATION_REPO/foundation.ckpt`.

For current foundation model work, use `scripts\train_foundation_model.py --raw-xmp-dir ...` or `--splits-dir ...` so the checkpoint is promoted to `SonnaEditorFoundation\` and stays out of the frontend profile list. It will warm-start from the active foundation checkpoint unless `--no-warm-start` is supplied. For MIT-Adobe FiveK, use catalog-derived splits from `build_dataset_from_catalog.py`.

Before relying on live RAW/XMP extraction coverage, restore the local fixture files under `tests/fixtures/`. The normal full suite now skips those local-data checks when the fixtures are missing or unreadable.
