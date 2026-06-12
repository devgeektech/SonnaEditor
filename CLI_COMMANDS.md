# Sonna Editor CLI Commands

This is the practical runbook for command-line data prep, Personal AI profiles,
Lite profiles, processing, diagnostics, and fine-tuning. Foundation-only
training, resume, and retrain commands live in `FOUNDATION_TRAINING.md`.

## Current Local State (2026-06-12)

- Python 3.11.15 via uv 0.11.17. `pyproject.toml` and `uv.lock` now require
  Python `3.11.*`; use `uv python pin 3.11` on Mac if uv tries a newer Python.
- PyTorch is `2.11.0+cu128`; CUDA is verified on the local NVIDIA GeForce RTX 3050.
- Direct runtime/dev dependencies are exact-pinned in `pyproject.toml` to reduce
  Mac resolver drift. `uv sync --extra dev` preserves CUDA PyTorch on
  Windows/Linux x86_64 through the pinned PyTorch CUDA 12.8 index, while macOS
  resolves the public `torch==2.11.0` / `torchvision==0.26.0` wheels.
- Training/profile caches were intentionally cleared so a fresh dataset can be added.
- `data\training_workspace\sonna_personal_001_dataset\`, `data\models\`, `data\parquet\`, `data\captures\`, `data\thumbnails\`, `data\audits\`, `data\dbg\`, `data\raw\sonna_training\`, `.pytest_cache`, `.ruff_cache`, and `.saha\active_profile.txt` were removed or emptied.
- There is currently no guaranteed local frontend-visible checkpoint in `v1_learning\`. Add fresh RAW+XMP data and train a Personal AI profile from the UI, or configure the hidden foundation checkpoint using `FOUNDATION_TRAINING.md`.
- Runtime directories are now created automatically from the project root. A fresh clone will bootstrap repo-local `data\training_sources\`, `data\raw\`, `data\raw\sonna_training\`, `v1_learning\`, and `.saha\` on backend or CLI startup.
- App startup is now wrapped by a one-command launcher. Use `.\run_saha.cmd`
  on Windows and `bash run_saha.sh` on macOS/Linux. The old two-terminal
  backend/frontend commands remain below as a debugging reference.
- Latest full local verification on 2026-06-12 passed: environment `11/11`,
  `uv run ruff check .`, `uv run python -m compileall -q src scripts tests`,
  `npm run build:vite`, and full pytest (`753 passed, 45 skipped, 1 warning`).
  The foundation CLI help now exposes the documented `--tone-presence-retry`
  and repeatable `--field-loss-weight FIELD=WEIGHT` flags.
- Lite profile creation now uses the configured foundation checkpoint. It does not depend on whichever Personal AI profile is active in the frontend.
- Raw training photos should live in separate repo-local child folders under `data/training_sources/` by default. Generated datasets and foundation runs default to `data/training_workspace/` unless you override `SONNA_TRAINING_WORKSPACE`.
- `scripts\train_profile.py` now logs default recipe values as `Training recipe ...`; only values explicitly supplied as CLI flags are logged as `Override ...`.
- Training now calibrates output-head biases from the training-set target medians. Fresh models zero the final head weights and set median biases; warm-started models keep learned final weights and only recenter the biases. With the current split those priors are Exposure2012=0.22, Temperature=5191K, Tint=5. WB residual heads start at zero when AsShot WB skip is enabled.
- Training startup now prints parameter counts, trainable percentage, dataset row counts, batches per epoch, estimated optimizer steps, effective learning rates, sampler/cap status, and a backbone freeze/unfreeze summary. The same payload is saved in `training_summary.json` as `startup_diagnostics`.
- Foundation runs use adaptive capacity. Catalog-scale splits default to `--backbone-trainable-layers stage:7`, so the final ConvNeXt stage plus feature fusion/heads train from epoch 0. Splits below 500 train rows automatically use `--backbone-unfreeze-strategy custom --backbone-trainable-layers none` unless explicit backbone flags are supplied. Use `--backbone-trainable-layers block:7:2,stage:6` for an ~8M trainable ablation, `block:7:1-2,stage:6` for ~12M, or `--backbone-unfreeze-strategy custom` to keep a spec fixed for the whole run.
- Default image augmentation is geometry-only. Photometric jitter is disabled by default because changing input brightness/colour without changing XMP labels adds noise to Exposure and white-balance learning.
- Fresh current-recipe models use the scene-stat architecture, adding six preview-derived luminance scene stats to the metadata path. Existing legacy checkpoints still load with their saved architecture version.
- Validation logs key-slider distribution ratios (`val_dist_*_std_ratio`) so prediction collapse is visible during training.

## Training Paths: What Is The Difference?

RAW+XMP data preparation and foundation training are not the same thing.

- **RAW+XMP dataset build:** reads edited photos, extracts previews/metadata/slider labels, and writes Parquet splits. This is just data preparation.
- **Personal AI profile training:** trains from those splits, warm-starting from the configured hidden foundation checkpoint in the frontend flow, and publishes a frontend-visible profile into `v1_learning/`.
- **Foundation training:** trains from real Lightroom
  slider labels (`RAW+XMP` or catalog-derived settings), but promotes the final
  checkpoint into the repo-local hidden foundation folder. It does not publish
  to `v1_learning/`.
- **Lite profile creation:** does not train from photos. It combines the configured foundation checkpoint with a preset and survey answers.

Foundation runs are versioned. By default a new foundation run warm-starts from
the active foundation checkpoint, trains on the new dataset, saves a new
checkpoint under `SonnaEditorFoundation\checkpoints\`, and makes that new
checkpoint active in `foundation_manifest.json`. The active checkpoint is the
cumulative foundation file: catalog and RAW+XMP runs update the same native
`SonnaEditor` slider-regression checkpoint. The default foundation folder is the
repo-local `SonnaEditorFoundation/` child folder, not gitignored `data/`. That
folder is tracked by the parent repo; checkpoint binaries are handled by Git
LFS. Existing checkpoints are not overwritten. New runs auto-promote as
`foundation-vN` unless a version stem is supplied. If a new run is bad, roll
back the active manifest pointer instead of deleting checkpoints:
`uv run python scripts\rollback_foundation.py --list`, then
`uv run python scripts\rollback_foundation.py foundation-vN`. Use
`--no-warm-start` only for a deliberate scratch foundation run. Frontend
Personal AI warm-starts use the progressive backbone schedule by default.
Foundation training starts with the final ConvNeXt stage trainable on larger
splits and then uses the progressive schedule to expand later. Small splits
below 500 train rows automatically use a heads/fusion-only custom schedule
unless a fixed custom strategy is requested.

Foundation promotion is now guarded. Normal foundation runs need at least 75
train rows and select the exported checkpoint by `val_visual_score`, a balanced
visual composite over key sliders plus collapse penalty, instead of plain
`val_loss`. Tiny RAW+XMP sets can still be used for smoke tests or reviewed
ablations with `--allow-small-foundation-dataset`.

The quality gate is tiered. Hard failures still block promotion and can be
overridden only with `--allow-quality-gate-failure` after visual review.
Moderate misses become `foundation_quality_warnings`: they are printed and
stored in the summary but do not block promotion by themselves. Do not use hard
failure overrides for routine active foundation updates.

Foundation summaries now persist `quality_gate_passed`,
`foundation_quality_failures`, `foundation_quality_warnings`,
`checkpoint_monitor`, and `best_checkpoint_score`. `quick_diagnostic.py` also
prints the active backbone capacity, field-loss overrides, quality-gate result,
and a train-median baseline comparison for failed gate fields when the split
Parquets are available.

Tone/presence-focused retries should use the same audited splits first, not a
new dataset. Pass `--tone-presence-retry` to use the reviewed retry recipe for
failed tone/presence gates. It raises loss pressure on `Exposure2012`,
`Whites2012`, `Blacks2012`, `Highlights2012`, `Shadows2012`, `Vibrance`, and
`Saturation`; explicit repeatable `--field-loss-weight FIELD=WEIGHT` values can
still override the preset per field. These are fresh runs from the prepared
Parquet splits with a foundation weight warm start by default, not
`--resume-from-checkpoint`.

Current triage note: `foundation-sonna-raw-xmp-001` was rejected as the active
foundation because it trained from only 132 train rows, overfit, and collapsed
Highlights/Shadows. The active foundation manifest was rolled back to
`foundation-fivek-catalog-expert-c-001`.

Output-head prior initialisation is bias-only for warm-started runs and
bias-plus-zero-final-weights for fresh runs. It does not freeze the head and it
does not hardcode the final prediction after training starts. The priors are
derived from the current training parquet's target medians, with missing fields
falling back to Lightroom defaults. When the direct AsShot WB skip is enabled,
Temperature/Tint head residual biases are set to zero so the initial WB output
starts at AsShot rather than the dataset median. Keep this enabled for
foundation and Personal AI training unless you are running a deliberate ablation with
`--no-target-prior-init`.

Foundation checkpoint naming rule for copy-paste commands:

```text
For every new foundation run, change BOTH --run-name and --version-stem.
Keep them identical, for example:
  foundation-fivek-catalog-expert-c-001
  foundation-fivek-catalog-expert-c-002

If you omit --version-stem, the system auto-allocates foundation-vN.
Do not reuse a previous --version-stem; old checkpoints are never overwritten.
```

## Project Flow

1. Choose the training data source.
   - **RAW + XMP sidecars:** edited RAW files with matching Lightroom `.xmp` files.
   - **Lightroom catalog:** edited photos in a `.lrcat`; no exported XMP required because slider targets are read from the catalog.
   - **Fine-tune captures:** previous Saha predictions plus final user-edited XMPs.
   - **Lite preset:** a foundation checkpoint + Lightroom preset + style survey can create an initial checkpoint, but this is not supervised model training from photos.

   Keep each source in its own folder so FiveK catalog builds, Sonna Personal
   AI runs, and future learning sets do not get mixed:

   ```text
   data/training_sources/sonna_personal_001/raw_xmp/
   data/training_sources/sonna_personal_002/raw_xmp/
   ```

   The shared RAW scanner currently recognises `.cr2`, `.cr3`, `.nef`,
   `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and
   `.srw` across dataset building, folder/API scans, preset processing,
   model inference, and fine-tune capture. Extension recognition means the app
   will attempt to process the file; successful preview/metadata extraction
   still depends on `rawpy`/LibRaw support for that camera file. Optional DNG
   normalisation still depends on Adobe DNG Converter support.

2. Build the training dataset.
   - RAW + XMP path: run `scripts/build_dataset.py`.
   - Catalog path: run `scripts/build_dataset_from_catalog.py`.
   - Both write a Parquet dataset, thumbnails, and train/val/test splits.
   - Splits are grouped by shoot to avoid data leakage.

3. Train the model.
   - Run `scripts/train_profile.py`.
   - The script supports `--resume-from-checkpoint PATH` to continue training from an existing checkpoint; omit it to start fresh from scratch.
   - This is now the only supported training entry point and includes the improved WB/Temperature/Tint recipe by default.
   - The training script saves the full run under your `--output-dir` and publishes a versioned checkpoint into `v1_learning/` by default so `/api/profiles` and the Electron frontend can see it.

4. Run the app.
   - Preferred one-command launcher: `.\run_saha.cmd` on Windows or
     `bash run_saha.sh` on macOS/Linux.
   - Backend scans `v1_learning/model-v*.ckpt`.
   - Frontend calls `GET /api/profiles` and shows those profiles.

5. Process new shoots.
   - Run through the Electron UI or `scripts/process_shoot_model.py`.
   - Output XMP files are written next to RAWs by default, plus `sonna_predictions.json` for later fine-tuning.

6. Fine-tune later.
   - Capture final Lightroom tweaks.
   - Run `scripts/finetune_profile.py` or the frontend fine-tune route.
   - Fine-tuned checkpoints also save into `v1_learning/` so they become visible in the frontend.

## Production Profile Paths

- **Personal AI profile:** built from RAW files plus matching Lightroom XMP sidecars. This is the normal profile-training path for operators and is now started from the Saha frontend. The backend resolves the configured foundation checkpoint, uses the same dataset builder and `sonna_editor.training.profile_runner.train_profile()` recipe as the CLI, warm-starts from that foundation, then publishes a versioned profile into `v1_learning/`.
- **Lite profile:** built from the configured foundation checkpoint plus a Lightroom preset and the six-question Lite survey. It does not depend on an active Personal AI profile. The first Lite processing pass dynamically adjusts Exposure, Temperature, and Tint because the preset owns the look sliders; all six survey answers are still stored in the profile package for calibration metadata and future fine-tuning.
- **Foundation model:** CLI-only and hidden from the UI. Use `FOUNDATION_TRAINING.md` for the complete train, resume, retrain, promotion, and FiveK guidance. The foundation CLI supports RAW+XMP folders and prepared Lightroom-parameter splits, including catalog-derived FiveK splits.

## Important Paths

| Purpose | Path | Notes |
|---|---|---|
| Source training RAW + XMP folder | Separate gitignored source folders, for example `data/training_sources/sonna_personal_001/raw_xmp/` | Edited RAW/DNG files plus same-stem `.xmp` sidecars |
| Source Lightroom catalog | Any `.lrcat` path, opened read-only | Lightroom should be closed for catalog reads |
| Personal AI dataset output root | `data/training_workspace/sonna_personal_001_dataset/` or frontend job workspace | Generated Parquet/splits/thumbnails; not scanned by the frontend |
| Foundation training workspace | `SONNA_TRAINING_WORKSPACE` or `data/training_workspace/` | Generated foundation datasets and run folders |
| Foundation repo | `SONNA_FOUNDATION_REPO` or repo-local `SonnaEditorFoundation/` | Promoted hidden checkpoints, outside gitignored `data/` |
| Foundation manifest | `<foundation_repo>/foundation_manifest.json` | Active foundation version, checkpoint pointer, version list, hashes, and capabilities |
| Foundation checkpoint path | `<foundation_repo>/checkpoints/<version>.ckpt` | Never overwrite old checkpoints |
| Personal AI training run outputs | `data/models/<run_name>/` or frontend job workspace | Non-foundation profile training artifacts |
| Best native checkpoint for a run | `<run_output>/model.ckpt` | Best validation checkpoint exported by trainer |
| Training summary | `<run_output>/training_summary.json` | Metrics and recipe record |
| Frontend-visible profile directory | `v1_learning/` | Only model checkpoints/sidecars belong here; do not store generated datasets here |
| Frontend-visible checkpoint pattern | `v1_learning/model-vX.Y.Z.ckpt` | Personal AI or fine-tuned profile |
| Frontend-visible sidecar pattern | `v1_learning/model-vX.Y.Z.json` | Profile metadata for the UI |
| Training source root | `data/training_sources/` | Local learning inputs only, one child folder per dataset/run |
| Generated data/artifacts | `data/` | Created by dataset/train/audit scripts, gitignored |
| Inference output XMP path | Next to RAWs when `write_xmp_in_place=True` | Or in the chosen output folder |
| Prediction capture sidecar | `<output_dir>/sonna_predictions.json` or shoot folder output | Needed for later fine-tuning |
| Fine-tune captures | `data/captures/` or frontend-selected captures folder | User correction data |

## What Can Train The Model

| Source | Can train `SonnaEditor`? | What provides target slider values? | Main command |
|---|---:|---|---|
| RAW files only, never edited | No | Nothing. RAW pixels/metadata are inputs only, not labels. | Not supported |
| RAW files + matching `.xmp` sidecars | Yes | Lightroom sidecar slider values | `scripts/build_dataset.py` then `scripts/train_profile.py` |
| Lightroom Classic catalog `.lrcat` + accessible RAW files | Yes | Catalog develop-settings blobs | `scripts/build_dataset_from_catalog.py` then `scripts/train_profile.py` |
| FiveK Lightroom catalog collection | Yes | Catalog develop-settings blobs from one expert collection, normally `C` first | `scripts/build_dataset_from_catalog.py` then `scripts/train_foundation_model.py --splits-dir ...` |
| Lightroom preset only | Not supervised training | Preset supplies fixed baseline values | `scripts/process_shoot_preset.py` |
| Foundation checkpoint + preset + style survey | Creates Lite initial profile | Foundation checkpoint supplies reusable model shell; preset + survey supply style baseline; initial processing adds per-photo auto corrections | `scripts/build_mode_b_checkpoint.py` |
| New shoot processed by Saha, then user edits XMPs | Yes, fine-tuning only | Final user-edited XMP compared against `sonna_predictions.json` | `scripts/finetune_profile.py` |

RAW-only training is not valid for the current model because this is supervised slider regression. The model learns `RAW preview + metadata -> Lightroom slider values`; without XMP, catalog develop settings, or captured final edits, there is no ground truth.

## Dataset Preparation File Map

| File | Used for | Notes |
|---|---|---|
| `scripts/build_dataset.py` | Command-line dataset build from RAW + XMP sidecars | Calls the RAW/XMP dataset builder and skips images without matching XMP labels. |
| `scripts/build_dataset_from_catalog.py` | Command-line dataset build from Lightroom catalog | Calls the catalog dataset builder and does not require exported XMP sidecars. |
| `src/sonna_editor/data/dataset.py` | RAW + XMP dataset implementation | Finds pairs, extracts previews/metadata/histograms, reads XMP slider labels, writes parquet and split files. |
| `src/sonna_editor/data/catalog_dataset.py` | Lightroom catalog dataset implementation | Prepares training rows from `.lrcat` develop settings, while reading the RAW files only for image input features and AsShot WB. |
| `src/sonna_editor/data/catalog.py` | Read-only Lightroom catalog access | Opens `.lrcat` safely, finds edited photos, and extracts develop-setting blobs. |
| `src/sonna_editor/data/extract.py` | Shared RAW feature extraction | Extracts embedded previews, metadata, histogram, and AsShot WB for both dataset paths and inference. |
| `src/sonna_editor/data/xmp.py` | XMP slider read/write | Reads Lightroom slider labels for RAW + XMP training and writes predicted sidecars during inference. |

So `catalog_dataset.py` is a dataset preparation file, but only for the catalog route. It is not the general RAW + XMP builder; that path lives in `dataset.py`.

## Lite Profile Flow

Preset-based profiles are called **Lite** profiles in the UI. This is not full supervised training from photos. It creates an initial checkpoint/sidecar package from the configured foundation checkpoint plus a Lightroom preset and Lite survey answers. During initial Lite processing, the pipeline detects `profile_type: mode_b_initial`, uses the preset as the fixed style baseline, and computes per-photo Exposure, Temperature, and Tint corrections before writing XMPs. Preset look sliders stay fixed. The foundation checkpoint's feature layers remain available for later fine-tuning.

Two preset execution paths exist:

| Path | What it does | Command |
|---|---|---|
| Lite checkpoint flow | Builds a frontend-visible Lite profile from foundation checkpoint + preset + survey. Initial processing uses fixed preset style plus per-photo Exposure/WB corrections, and later fine-tuned checkpoints use normal model inference. | `run_style_survey.py` -> `build_mode_b_checkpoint.py` -> `process_shoot_model.py` |
| Direct preset execution | Applies a preset with the same heuristic per-photo corrections and writes XMPs directly. No selectable profile checkpoint is created. | `process_shoot_preset.py` |

Recommended Lite flow:

1. Train or configure the foundation checkpoint in `SonnaEditorFoundation/`.
2. Export or choose a Lightroom preset `.xmp`.
3. Run the style survey to create a survey JSON.
4. Build a Lite checkpoint. If `--output` is omitted, the CLI publishes to the next frontend-visible path: `v1_learning/model-v0.N.0.ckpt`.
5. Refresh profiles in the frontend or call `/api/profiles`; the new `model-v0.N.0.ckpt` appears as `profile_type: mode_b_initial`.
6. Execute it via the Electron UI or `scripts/process_shoot_model.py`; initial Lite profiles use the adaptive preset branch automatically.
7. If the user edits the results in Lightroom, capture those edits and fine-tune later through the normal continuous-learning path.

Create survey JSON interactively:

```powershell
uv run python scripts\run_style_survey.py `
  --output v1_learning\wedding-lite-survey.json
```

Create survey JSON non-interactively:

```powershell
uv run python scripts\run_style_survey.py `
  --output v1_learning\wedding-lite-survey.json `
  --non-interactive `
  --answers exposure=0,temperature=1,tint=0,contrast=0,saturation=0,shadows=0
```

Build and publish a Lite checkpoint to the frontend-visible folder. By default
this reads the configured foundation checkpoint from `SONNA_FOUNDATION_CHECKPOINT`,
`SONNA_FOUNDATION_REPO/foundation_manifest.json`, or
`SONNA_FOUNDATION_REPO/foundation.ckpt`:

```powershell
uv run python scripts\build_mode_b_checkpoint.py `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --survey v1_learning\wedding-lite-survey.json `
  --profile-name "Wedding Lite"
```

Expected outputs:

```text
v1_learning\model-v0.2.0.ckpt
v1_learning\model-v0.2.0.json
```

If the first Lite version already exists, the CLI picks the next open version, for example `model-v0.2.0.ckpt`. It refuses to overwrite an existing output.

Important: Lite profiles created before the 2026-06-02 Mode B fixes can over-apply the preset because they added the base model's predicted Exposure/colour values on top. Rebuild those old `v1_learning\model-v0.*.ckpt` Lite profiles from the UI or CLI before judging current Mode B output.

After updating code, restart the backend/Electron app before processing from the
UI. A live backend process will otherwise keep the old Mode B processing code in
memory. Rebuild any old `model-v0.*.ckpt` Lite profile before judging current
Lite output.

Run the Lite checkpoint on a shoot:

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v0.2.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput
```

Lite checkpoints are visible in the UI when they are published into
`v1_learning/`. The foundation checkpoint stays in the repo-local hidden
foundation folder and is not listed as a frontend profile.

`process_shoot_preset.py` is the direct preset-only path. It does not build a
profile checkpoint and does not publish anything to the UI. Instead it:

- parses a Lightroom preset
- applies content-aware per-photo adjustments
- writes XMP sidecars directly next to the source RAWs (or into `--output-dir`)
- supports per-photo auto exposure, auto white balance, shadow recovery, and
  highlight recovery options

Use `process_shoot_preset.py` when you want quick preset-derived XMPs without
creating a selectable profile. Use the Lite checkpoint flow when you want the
preset to become a frontend profile that can later be fine-tuned from Lightroom
corrections.

## 1. Install And Verify

```powershell
cd C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor
uv sync --extra dev
uv run python scripts\verify_environment.py
```

Expected on the current Windows workstation:

```text
PyTorch import: v2.11.0+cu128
Preferred torch device: cuda
CUDA available: True
CUDA matmul: OK
```

## 2. Prepare Lightroom Training Data

In Lightroom Classic:

1. Select edited photos.
2. Use `Metadata -> Save Metadata to File`.
3. Confirm every RAW has a matching `.xmp` sidecar next to it.
4. Place or point to that folder as the dataset input.

Recommended repo-local training source folders:

```powershell
data\training_sources\sonna_personal_001\raw_xmp\
```

Mac example:

```bash
data/training_sources/sonna_personal_001/raw_xmp/
```

## 3A. Build Dataset And Splits From RAW + XMP Sidecars

```powershell
uv run python scripts\build_dataset.py `
  --input-dir data\training_sources\sonna_personal_001\raw_xmp `
  --output-dir data\training_workspace\sonna_personal_001_dataset `
  --profile-name "sonna_v2" `
  --workers 4 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Expected outputs:

```text
data\training_workspace\sonna_personal_001_dataset\dataset.parquet
data\training_workspace\sonna_personal_001_dataset\thumbnails\
data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet
data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet
data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\test.parquet
```

## 3B. Build Dataset And Splits From Lightroom Catalog

Use this when edits are still inside Lightroom and you do not want to export XMP sidecars first. Lightroom Classic must be closed, and the RAW files referenced by the catalog must be accessible at the paths stored in the catalog.

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "D:\Lightroom\Sonna Catalog.lrcat" `
  --output-dir data\training_workspace\sonna_personal_001_dataset `
  --profile-name "sonna_v2" `
  --limit 30000 `
  --workers 4 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Expected outputs are the same as the RAW + XMP path, plus:

```text
data\training_workspace\sonna_personal_001_dataset\catalog_build_stats.json
```

Important catalog behavior:

```text
Catalog is opened read-only.
Lightroom lock/journal files stop the run.
Only photos with develop settings are selected.
Virtual copies pointing to the same RAW are deduplicated.
Likely unedited photos are skipped.
Missing RAW files are skipped.
AsShot WB is still extracted from the RAW metadata for Temperature/Tint learning.
```

## 4. Train With Current Recipe

This is the recommended command for a Personal AI training run from prepared splits:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --test-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-personal-run01 `
  --profile-name "Sonna Personal Run 01" `
  --max-epochs 50 `
  --batch-size 16 `
  --num-workers 4
```

To resume an interrupted training run, add `--resume-from-checkpoint` and point it at one of the Lightning checkpoints under the run's `checkpoints\` directory:

```powershell
  --resume-from-checkpoint "data\models\sonna-personal-run01\checkpoints\epoch=...-val_loss=....ckpt" `
```

Omit `--resume-from-checkpoint` for a scratch CLI experiment. Pass
`--base-model-checkpoint <foundation.ckpt>` when you want a CLI warm start from
the hidden foundation checkpoint without carrying over optimizer or epoch state.
The frontend Personal AI route supplies that configured hidden foundation
checkpoint automatically, so normal operator-created Personal AI profiles start
from the foundation model.

Note on `--resume-from-checkpoint`:

- Purpose: intended to restart an interrupted training run and continue training with the same training module state (optimizer, scheduler, epoch counters, callbacks) preserved by PyTorch Lightning.
- Do not use it as the primary mechanism for one-off fine-tuning from a published frontend checkpoint or for foundation warm-starts. For foundation warm-starts use `--base-model-checkpoint <ckpt>`. For intentional fine-tuning from an existing published checkpoint use `scripts/finetune_profile.py --base-model <ckpt>` which implements the capture-combined dataset, evaluation, and promotion workflow.
- Technically you can resume and then change some hyperparameters, but this risks inconsistent optimizer/scheduler state. Prefer `finetune_profile.py` for controlled fine-tuning and `train_profile.py --resume-from-checkpoint` only for true restarts.

Current `scripts/train_profile.py` defaults:

```text
image_resolution=512
lr=1e-4
max_epochs=50
freeze_backbone_epochs=3
current scene-stat architecture for fresh models
Exposure loss weight=5.0
Temperature loss weight=4.0
Tint loss weight=4.0
Contrast/Highlights/Shadows minimum loss weight=3.0
Whites/Blacks/Saturation/Vibrance minimum loss weight=2.0
Temperature bucket loss weight=0.15
Tint bucket loss weight=2.0
Sign-wrong penalty weight=0.2
WB metadata skip=enabled
target prior init=enabled
photometric augmentation=disabled
optional repeatable named slider overrides via --field-loss-weight FIELD=WEIGHT
```

Use `--no-target-prior-init` only for an ablation. For the next quality evaluation, start fresh and do not resume the old unsatisfactory checkpoint, otherwise target-prior calibration, adaptive capacity, and regenerated splits will not be evaluated cleanly.

Important small-data caution:

- On the earlier 189-photo diagnostic dataset, the scene-stat experiment reached low MAE but collapsed harder than the previous published profile (`29` collapsed sliders versus `14` on the same 27-photo validation split).
- Treat collapse analysis as a promotion gate, not just validation loss or MAE.
- Do not promote old rejected diagnostic runs; keep them as diagnostics only.

What this saves:

```text
data\models\sonna-personal-run01\model.ckpt
data\models\sonna-personal-run01\model.json
data\models\sonna-personal-run01\training_summary.json
data\models\sonna-personal-run01\checkpoints\
data\models\sonna-personal-run01\tensorboard\
```

What gets published for the frontend:

```text
v1_learning\model-v2.0.0.ckpt
v1_learning\model-v2.0.0.json
```

If `model-v2.0.0.ckpt` already exists, the next run publishes as the next available version. Treat this as an internal file version, not a model-family decision.

### Diagnostics: quick checks and audits

Before or after a training run you can run lightweight diagnostics to inspect the run summary and deeper slider audits.

- Quick diagnostic (reads the saved training summary JSON and prints recommendations):

```powershell
uv run python scripts\quick_diagnostic.py
```

For a specific foundation run:

```powershell
uv run python scripts\quick_diagnostic.py `
  --summary-path data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-002-tone-presence\training\training_summary.json
```

- Prediction collapse audit (runs a checkpoint on a parquet split and reports target/predicted spread):

```powershell
uv run python scripts\analyse_prediction_collapse.py `
  --model-path v1_learning\model-v2.0.0.ckpt `
  --parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --output data\audits\prediction_collapse.md `
  --limit 50 `
  --batch-size 16
```

- Dataset diversity audit (scene brightness, contrast, WB, and edit target buckets):

```powershell
uv run python scripts\audit_dataset_diversity.py `
  --parquet data\training_workspace\sonna_personal_001_dataset\dataset.parquet `
  --output data\audits\dataset_diversity.md
```

- Full all-slider audit (v1.2.3 audit example; this is read-only and produces markdown + parquet outputs in `scripts/output/`):

```powershell
uv run python scripts\audit_all_sliders_v1.2.3.py
```

Notes:
- `quick_diagnostic.py` now discovers training summary JSON files automatically under the project tree and can also list published checkpoints in `v1_learning/` for selection. If an older summary does not include row counts, it reads train/val/test counts from split parquet metadata, falling back to `data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/`.
- `audit_all_sliders_v1.2.3.py` now discovers published checkpoints under `v1_learning/model-v*.ckpt` and will prompt you to choose one if multiple are found.
- Audit scripts (like `audit_all_sliders_v1.2.3.py`) are read-only analyses that load a published checkpoint and a test split; they may take longer and require the test parquet to exist. Output goes to `scripts/output/` — keep those reports under version control only when intended.

### Foundation model commands

Foundation training, resume, retrain, promotion, and FiveK-specific guidance
live in `FOUNDATION_TRAINING.md`. Keep the foundation checkpoint in the separate
foundation folder, not in `v1_learning/`, so it stays hidden from the frontend.

The current FiveK download was inspected at:

```text
C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset
```

It currently contains 5,000 DNG source files, `raw_photos\fivek.lrcat`,
Lightroom preview-cache files, and text/license metadata. The catalog is now
the supported FiveK training source; separate rendered target folders are not
used.

Verified on 2026-06-05:

```text
raw_photos:                 5,000 .dng files
fivek.lrcat:                present, about 655 MB
blocking lock files:        none
expert collections A-E:     5,000 rows each
Collection C smoke build:   20 rows succeeded, 0 missing files, 0 parse errors
```

The Lightroom catalog itself contains 60,000 catalog image rows over the same
5,000 DNGs: 12 virtual-copy/recipe variants per source file. Expert collections
`A`, `B`, `C`, `D`, and `E` each contain 5,000 rows. Use one collection first,
normally `C`, rather than mixing all variants in one plain slider-regression
model.

For RAW+XMP foundation training, first export Lightroom metadata to sidecars,
keep the source files in a dedicated folder, build/audit inspectable splits, and
then train from those splits:

```powershell
uv run python scripts\build_dataset.py `
  --input-dir data\training_sources\sonna_foundation_001\raw_xmp `
  --output-dir data\training_workspace\sonna_foundation_001_dataset `
  --profile-name "sonna_foundation_001" `
  --workers 8 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

```powershell
uv run python scripts\audit_catalog.py `
  --parquet-path data\training_workspace\sonna_foundation_001_dataset\dataset.parquet `
  --output-dir data\training_workspace\sonna_foundation_001_dataset\audit
```

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified `
  --workspace-dir data\training_workspace `
  --foundation-repo SonnaEditorFoundation `
  --profile-name "Sonna RAW XMP Foundation" `
  --run-name foundation-sonna-raw-xmp-001 `
  --version-stem foundation-sonna-raw-xmp-001 `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

Tone/presence retry from the same splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified `
  --workspace-dir data\training_workspace `
  --foundation-repo SonnaEditorFoundation `
  --profile-name "Sonna RAW XMP Foundation Tone Presence 002" `
  --run-name foundation-sonna-raw-xmp-002-tone-presence `
  --version-stem foundation-sonna-raw-xmp-002-tone-presence `
  --max-epochs 150 `
  --batch-size 8 `
  --workers 8 `
  --tone-presence-retry
```

FiveK catalog foundation training first builds Expert C splits:

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset\raw_photos\fivek.lrcat" `
  --output-dir data\training_workspace\fivek_expert_c_catalog_dataset `
  --profile-name "fivek_expert_c_catalog" `
  --collection-name "C" `
  --include-unedited-looking `
  --limit 5000 `
  --workers 8 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Then train and promote a native foundation checkpoint from the prepared splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir data\training_workspace\fivek_expert_c_catalog_dataset\splits_v2_stratified `
  --workspace-dir data\training_workspace `
  --foundation-repo SonnaEditorFoundation `
  --profile-name "Sonna FiveK Catalog Foundation Expert C" `
  --run-name foundation-fivek-catalog-expert-c-001 `
  --version-stem foundation-fivek-catalog-expert-c-001 `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

For the next copy-paste run, change both names together:

```text
--run-name foundation-fivek-catalog-expert-c-002
--version-stem foundation-fivek-catalog-expert-c-002
```

Do not reuse `--version-stem`; the checkpoint promoter refuses overwrites.
The promoted checkpoint becomes the default base model automatically because
`SonnaEditorFoundation\foundation_manifest.json` is updated to point at it.
Mode A Personal AI training and Mode B Lite creation both resolve that manifest
unless `SONNA_FOUNDATION_CHECKPOINT` is explicitly set.

Use `--include-unedited-looking` for FiveK because its catalog develop blobs are
sparse: many default sliders are absent, not proof that the expert edit is
unedited. Do not use that flag for ordinary Sonna catalogs unless the dataset
has been audited.

FiveK teaches the model through DNG preview features plus catalog develop
settings. It is not rendered-image training. Missing catalog slider values are
masked out of the loss, while fresh/foundation output priors fall back to
Lightroom defaults for fields with no labels.

For an ordinary Sonna catalog, use the same catalog builder without the FiveK
collection flag unless you intentionally want one Lightroom collection:

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "D:\Lightroom\Sonna Catalog.lrcat" `
  --output-dir data\training_workspace\sonna_catalog_dataset `
  --profile-name "sonna_catalog" `
  --workers 8 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Then train from the prepared Sonna catalog splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir data\training_workspace\sonna_catalog_dataset\splits_v2_stratified `
  --workspace-dir data\training_workspace `
  --foundation-repo SonnaEditorFoundation `
  --profile-name "Sonna Catalog Foundation 001" `
  --run-name foundation-sonna-catalog-001 `
  --version-stem foundation-sonna-catalog-001 `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

After FiveK training, run RAW+XMP foundation training without `--no-warm-start`
so it starts from the active FiveK checkpoint and writes a new cumulative
foundation checkpoint. The same works in reverse later: another FiveK catalog
run starts from the active RAW+XMP-updated checkpoint.

After training and auditing a foundation checkpoint, push the default base model
to GitHub through Git LFS:

```powershell
git status --short
git lfs status
git add SonnaEditorFoundation\foundation_manifest.json SonnaEditorFoundation\checkpoints\<your-version>.ckpt SonnaEditorFoundation\checkpoints\<your-version>.json
git lfs ls-files
git commit -m "train foundation checkpoint <your-version>"
git push origin main
```

Replace `<your-version>` with the exact `--version-stem` used during training.
Do not commit `data\training_workspace\`; that folder contains local generated
datasets and training runs.

You only need `git lfs install` once per machine. After that, normal `git push`
uploads checkpoint binaries through LFS automatically. On a new Mac or Windows
checkout, run `git lfs pull` after clone/pull to download the real `.ckpt`
contents.

On the Windows RTX 3050 workstation, start foundation runs at `--batch-size 8`.
The RAW+XMP foundation CLI will automatically retry with smaller batch sizes if
CUDA runs out of memory.

## 5. Train With Explicit Published Version

Use this when you want a specific visible checkpoint name:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --test-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-personal-run02 `
  --profile-name "Sonna Personal Candidate 02" `
  --publish-version v2.0.2 `
  --max-epochs 50 `
  --batch-size 16
```

The script refuses to overwrite an existing published checkpoint.

## 6. Train Without Publishing To Frontend

Use this for experiments you do not want visible in Saha:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --test-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\scratch-run `
  --image-resolution 512 `
  --no-publish
```

## 7. Start The App

Preferred client command from the repo root:

```powershell
.\run_saha.cmd
```

macOS/Linux:

```bash
bash run_saha.sh
```

Equivalent explicit command:

```powershell
uv run python scripts\run_app.py
```

The launcher creates runtime folders, runs `npm install` only when
`saha-app\node_modules\` is missing, then starts `npm run dev` in `saha-app`.
The Electron main process starts or reuses the backend on port `8765`.
PowerShell users can also run `.\run_saha.ps1`; `.\run_saha.cmd` is usually
smoother on client machines because it avoids execution-policy prompts.
Both machines need `uv` and Node.js LTS on `PATH`; missing tools are reported
with a clear setup message.

Manual two-terminal startup remains useful for debugging.

Terminal 1:

```powershell
uv run python scripts\serve.py --port 8765
```

Terminal 2:

```powershell
cd saha-app
npm install
npm run dev
```

The frontend profile list comes from:

```text
GET http://127.0.0.1:8765/api/profiles
```

It will show checkpoints matching:

```text
v1_learning\model-v*.ckpt
```

## 8. Confirm Published Profiles From Terminal

```powershell
Get-ChildItem v1_learning\model-v*.ckpt
Get-ChildItem v1_learning\model-v*.json
```

Or through the API:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/profiles
```

## 9. Process A Shoot With A Trained Model

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v2.0.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput
```

The Electron app uses the same backend inference path.

## 10. Fine-Tune Existing Profile

Dry run:

```powershell
uv run python scripts\finetune_profile.py `
  --base-model v1_learning\model-v2.0.0.ckpt `
  --captures-dir data\captures `
  --original-train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --dry-run
```

Actual fine-tune:

```powershell
uv run python scripts\finetune_profile.py `
  --base-model v1_learning\model-v2.0.0.ckpt `
  --captures-dir data\captures `
  --original-train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --output-dir v1_learning `
  --max-epochs 30 `
  --batch-size 16
```

The fine-tuned checkpoint is versioned under `v1_learning/`, so it becomes visible in the frontend after profile refresh.

