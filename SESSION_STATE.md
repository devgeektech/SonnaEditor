# Session State - Sonna Editor

**Saved:** 2026-06-01 15:35 local time
**Current phase/task:** Training quality refresh plus inference fix for pink/red output cast.

## Current Workspace

- Repo path: `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`
- Branch: `main`, tracking `origin/main`
- Recent committed history in this checkout: `aac360a feat: Enhance dataset building and training scripts` on top of four earlier commits.
- Current intentional dirty files from this work: `AGENTS.md`, `CLAUDE.md`, `HANDOVER.md`, `README.md`, `RUN.md`, `SESSION_STATE.md`, `SONNA_EDITOR_BUILD_SPEC.md`, `TRAINING_COMMANDS.md`, `project_knowledge.md`, `pyproject.toml`, `scripts/train_profile.py`, `src/sonna_editor/config.py`, `src/sonna_editor/data/dataset.py`, `src/sonna_editor/inference/pipeline.py`, `src/sonna_editor/model/architecture.py`, `tests/test_dataset.py`, `tests/test_inference_v2.py`, `tests/test_training.py`, `uv.lock`.
- Also present in the working tree and left untouched as unrelated/pre-existing local edits: `scripts/audit_all_sliders_v1.2.3.py`, `scripts/quick_diagnostic.py`, `scripts/output/all_slider_audit_v1.2.3.md`, `scripts/output/all_slider_audit_v1.2.3_stats.parquet`.
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
- `v1_learning/model-v2.0.0.ckpt` and `v1_learning/model-v2.0.0.json` are present locally.
- No checkpoint files were found under `data/models/` during this pass.
- The frontend should discover the local v2 profile. Rerun inference after the RGB tone-curve endpoint fix to regenerate XMPs without the pink/red white-point cast.

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
- Full `uv run pytest tests` result: 691 passed, 12 skipped, 28 failed. Failures are concentrated in `tests/test_extract.py` and `tests/test_xmp.py` because the gitignored RAW/XMP fixtures are absent from `tests/fixtures/` (`sample.cr3`, `sample.xmp`, `sample_edit.xmp`) and two extract tests require Windows symlink privileges.

## Current Code Behavior Notes

- `scripts/train_profile.py` default v2 recipe remains:
  - `image_resolution=512`
  - `lr=1e-4`
  - `max_epochs=50`
  - `freeze_backbone_epochs=3`
  - Temperature loss weight=6.0
  - Tint loss weight=6.0
  - Exposure2012 loss weight=4.0
  - Temperature bucket loss weight=0.15
  - Tint bucket loss weight=2.0
  - Sign-wrong penalty weight=0.2
  - WB metadata skip enabled
- Fresh training defaults to target-prior output initialisation. Use `--no-target-prior-init` only for an ablation.
- Default training augmentation is now geometry-only; photometric jitter remains configurable but disabled by default.
- `Profile.profile_type` is already implemented in backend profile responses and frontend profile classification. `None` means legacy/current Mode A trained profile; `"mode_b_initial"` means Lite/Mode B preset-derived profile.

## Next Suggested Step

First rerun inference on the affected shoot so the new RGB tone-curve endpoint stabilisation rewrites fresh XMPs. For further model-quality evaluation, run a fresh v2 training command from `TRAINING_COMMANDS.md` now that CUDA is active and the split/training defaults are fixed. Do not resume the earlier unsatisfactory checkpoint for that evaluation; start from scratch so the target-prior initialisation, geometry-only augmentation, and regenerated splits all take effect.

Expect a new training process to show `GPU available: True, used: True`.

Before relying on full-suite green status, restore the local fixture files under `tests/fixtures/` or mark those fixture-dependent tests as integration/local-data tests, then rerun `uv run pytest tests`.
