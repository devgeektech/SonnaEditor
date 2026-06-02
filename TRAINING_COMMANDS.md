# Sonna Editor Training Commands

This is the practical runbook for preparing a dataset, training a model, and making the trained checkpoint visible in the Saha frontend.

## Current Local State (2026-06-01)

- Python 3.11.15 via uv 0.11.17.
- PyTorch is `2.11.0+cu128`; CUDA is verified on the local NVIDIA GeForce RTX 3050.
- `uv sync --extra dev` now preserves CUDA PyTorch on Windows/Linux x86_64 through the pinned PyTorch CUDA 12.8 index in `pyproject.toml` / `uv.lock`.
- Current local dataset: `v1_learning\dataset\dataset.parquet` has 189 rows.
- Current stratified splits: train=132, val=27, test=30.
- Split generation is shoot-grouped and now balances Temperature correction, Exposure2012, and Tint correction. The previous local split had train Exposure mean ~0.212 versus val/test ~0.480/~0.504, which could make brightness look poor even when training loss improved.
- `v1_learning\model-v2.0.0.ckpt` and `v1_learning\model-v2.0.0.json` are currently present in this workspace, so the frontend can discover one local v2 profile.
- `v1_learning\model-v2.0.0.ckpt` remains the active local frontend profile. A fresh scene-stats candidate was trained at `data\models\sonna-v2-scene-stats-run01`, but it failed collapse analysis and was not kept frontend-visible.
- `scripts\train_profile.py` now logs default recipe values as `Training recipe ...`; only values explicitly supplied as CLI flags are logged as `Override ...`.
- Fresh training now initialises output-head biases from the training-set target medians. With the current split those priors are Exposure2012=0.22, Temperature=5191K, Tint=5. WB residual heads start at zero when AsShot WB skip is enabled.
- Default image augmentation is geometry-only. Photometric jitter is disabled by default because changing input brightness/colour without changing XMP labels adds noise to Exposure and white-balance learning.
- Fresh v2 models now use `arch_version=2`, adding six preview-derived luminance scene stats to the metadata path. Existing checkpoints load with their saved architecture version.
- Validation logs key-slider distribution ratios (`val_dist_*_std_ratio`) so prediction collapse is visible during training.

## Project Flow

1. Choose the training data source.
   - **RAW + XMP sidecars:** edited RAW files with matching Lightroom `.xmp` files.
   - **Lightroom catalog:** edited photos in a `.lrcat`; no exported XMP required because slider targets are read from the catalog.
   - **Fine-tune captures:** previous Saha predictions plus final user-edited XMPs.
   - **Mode B preset:** a Lightroom preset + style survey can create an initial checkpoint, but this is not supervised model training from photos.

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

4. Run the backend and frontend.
   - Backend scans `v1_learning/model-v*.ckpt`.
   - Frontend calls `GET /api/profiles` and shows those profiles.

5. Process new shoots.
   - Run through the Electron UI or `scripts/process_shoot_model.py`.
   - Output XMP files are written next to RAWs by default, plus `sonna_predictions.json` for later fine-tuning.

6. Fine-tune later.
   - Capture final Lightroom tweaks.
   - Run `scripts/finetune_profile.py` or the frontend fine-tune route.
   - Fine-tuned checkpoints also save into `v1_learning/` so they become visible in the frontend.

## Important Paths

| Purpose | Path |
|---|---|
| Source training RAW + XMP folder | `data/raw/<training_set_name>/` or any absolute folder |
| Source Lightroom catalog | Any `.lrcat` path, opened read-only |
| Dataset output root | `v1_learning/dataset/` |
| Full dataset Parquet | `v1_learning/dataset/dataset.parquet` |
| Training thumbnails | `v1_learning/dataset/thumbnails/` |
| Canonical train/val/test splits | `v1_learning/dataset/splits_v2_stratified/` |
| Training run outputs | `data/models/<run_name>/` (use a consistent folder per run) |
| Best native checkpoint for that run | `data/models/<run_name>/model.ckpt` |
| Training summary | `data/models/<run_name>/training_summary.json` |
| Frontend-visible profile directory | `v1_learning/` |
| Frontend-visible checkpoint pattern | `v1_learning/model-vX.Y.Z.ckpt` |
| Frontend-visible sidecar pattern | `v1_learning/model-vX.Y.Z.json` |
| Generated data/artifacts | `data/` | Created by dataset/train/audit scripts; keep this directory gitignored. |
| Inference output XMP path | Next to RAWs when `write_xmp_in_place=True` |
| Prediction capture sidecar | `<output_dir>/sonna_predictions.json` or shoot folder output |
| Fine-tune captures | `data/captures/` or frontend-selected captures folder |

## What Can Train The Model

| Source | Can train `SonnaEditor`? | What provides target slider values? | Main command |
|---|---:|---|---|
| RAW files only, never edited | No | Nothing. RAW pixels/metadata are inputs only, not labels. | Not supported |
| RAW files + matching `.xmp` sidecars | Yes | Lightroom sidecar slider values | `scripts/build_dataset.py` then `scripts/train_profile.py` |
| Lightroom Classic catalog `.lrcat` + accessible RAW files | Yes | Catalog develop-settings blobs | `scripts/build_dataset_from_catalog.py` then `scripts/train_profile.py` |
| Lightroom preset only | Not supervised training | Preset supplies fixed baseline values | `scripts/build_mode_b_checkpoint.py` |
| Preset + style survey | Creates Mode B initial checkpoint | Preset + survey calibrate output-head biases | `scripts/build_mode_b_checkpoint.py` |
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

## Preset / Mode B Flow

Preset-based profiles are called **Mode B** or **Lite** profiles in this project. This is not full supervised training from photos. It creates an initial checkpoint by taking an existing trained Mode A checkpoint and shifting its output-head biases from a Lightroom preset plus six survey answers.

Two preset execution paths exist:

| Path | What it does | Command |
|---|---|---|
| Mode B checkpoint flow | Builds a frontend-visible model checkpoint from preset + survey + base trained checkpoint. This uses the normal model inference path afterwards. | `run_style_survey.py` -> `build_mode_b_checkpoint.py` -> `process_shoot_model.py` |
| Direct preset execution | Applies a preset with heuristic per-photo corrections and writes XMPs directly. No model checkpoint is created. | `process_shoot_preset.py` |

Recommended Mode B flow:

1. Start from an active trained Mode A checkpoint in `v1_learning/`, for example `v1_learning/model-v1.2.3-prod256.ckpt` or a newer trained profile.
2. Export or choose a Lightroom preset `.xmp`.
3. Run the style survey to create a survey JSON.
4. Build a Mode B checkpoint. If `--output` is omitted, the CLI now publishes to the next frontend-visible path: `v1_learning/model-v0.N.0.ckpt`.
5. Refresh profiles in the frontend or call `/api/profiles`; the new `model-v0.N.0.ckpt` appears as `profile_type: mode_b_initial`.
6. Execute it like any other model via the Electron UI or `scripts/process_shoot_model.py`.
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
  --answers exposure=0,temperature=1,tint=0,contrast=1,saturation=-1,shadows=1
```

Build and publish a Mode B checkpoint to the frontend-visible folder:

```powershell
uv run python scripts\build_mode_b_checkpoint.py `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --survey v1_learning\wedding-lite-survey.json `
  --base-ckpt v1_learning\model-v1.2.3-prod256.ckpt `
  --profile-name "Wedding Lite"
```

Expected outputs:

```text
v1_learning\model-v0.1.0.ckpt
v1_learning\model-v0.1.0.json
```

If `model-v0.1.0.ckpt` already exists, the CLI picks the next open version, for example `model-v0.2.0.ckpt`. It refuses to overwrite an existing output.

Run the Mode B checkpoint on a shoot:

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v0.1.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput
```

Direct preset-to-XMP execution, without model checkpoint creation:

```powershell
uv run python scripts\process_shoot_preset.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --output-dir D:\Shoots\ClientShoot01\PresetOutput `
  --auto-exposure `
  --no-auto-white-balance `
  --auto-shadow-recovery `
  --auto-highlight-recovery
```

Use direct preset execution only when you want quick preset-derived XMPs. Use Mode B checkpoint flow when you want the preset to become a selectable frontend profile and later fine-tune from corrections.

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

Recommended local folder:

```powershell
data\raw\sonna_training\
```

The folder can be external too, for example:

```powershell
D:\SonnaTraining\EditedRawWithXmp\
```

## 3A. Build Dataset And Splits From RAW + XMP Sidecars

```powershell
uv run python scripts\build_dataset.py `
  --input-dir data\raw\sonna_training `
  --output-dir v1_learning\dataset `
  --profile-name "sonna_v2" `
  --workers 4 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Expected outputs:

```text
v1_learning\dataset\dataset.parquet
v1_learning\dataset\thumbnails\
v1_learning\dataset\splits_v2_stratified\train.parquet
v1_learning\dataset\splits_v2_stratified\val.parquet
v1_learning\dataset\splits_v2_stratified\test.parquet
```

## 3B. Build Dataset And Splits From Lightroom Catalog

Use this when edits are still inside Lightroom and you do not want to export XMP sidecars first. Lightroom Classic must be closed, and the RAW files referenced by the catalog must be accessible at the paths stored in the catalog.

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "D:\Lightroom\Sonna Catalog.lrcat" `
  --output-dir v1_learning\dataset `
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
v1_learning\dataset\catalog_build_stats.json
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

This is the recommended command for the next v2 training run:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-v2-run01 `
  --profile-name "Sonna v2 Run 01" `
  --max-epochs 50 `
  --batch-size 16 `
  --num-workers 4
```

To resume an interrupted training run, add `--resume-from-checkpoint` and point it at one of the Lightning checkpoints under the run's `checkpoints\` directory:

```powershell
  --resume-from-checkpoint "data\models\sonna-v2-run01\checkpoints\epoch=...-val_loss=....ckpt" `
```

Omit `--resume-from-checkpoint` for a fresh model.

Note on `--resume-from-checkpoint`:

- Purpose: intended to restart an interrupted training run and continue training with the same training module state (optimizer, scheduler, epoch counters, callbacks) preserved by PyTorch Lightning.
- Do not use it as the primary mechanism for one-off fine-tuning from a published frontend checkpoint. For intentional fine-tuning from an existing published checkpoint use `scripts/finetune_profile.py --base-model <ckpt>` which implements the capture-combined dataset, evaluation, and promotion workflow.
- Technically you can resume and then change some hyperparameters, but this risks inconsistent optimizer/scheduler state. Prefer `finetune_profile.py` for controlled fine-tuning and `train_profile.py --resume-from-checkpoint` only for true restarts.

Current `scripts/train_profile.py` defaults:

```text
image_resolution=512
lr=1e-4
max_epochs=50
freeze_backbone_epochs=3
arch_version=2 for fresh models
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
```

Use `--no-target-prior-init` only for an ablation. For the next quality evaluation, start fresh and do not resume the old unsatisfactory checkpoint, otherwise the new fresh-head initialisation and regenerated splits will not be evaluated cleanly.

Important small-data caution:

- On the current 189-photo dataset, `data\models\sonna-v2-scene-stats-run01` reached low MAE but collapsed harder than the existing `model-v2.0.0` (`29` collapsed sliders versus `14` on the same 27-photo validation split).
- Treat collapse analysis as a promotion gate, not just validation loss or MAE.
- Do not promote `data\models\sonna-v2-scene-stats-run01`; keep it as a diagnostic run.

What this saves:

```text
data\models\sonna-v2-run01\model.ckpt
data\models\sonna-v2-run01\model.json
data\models\sonna-v2-run01\training_summary.json
data\models\sonna-v2-run01\checkpoints\
data\models\sonna-v2-run01\tensorboard\
```

What gets published for the frontend:

```text
v1_learning\model-v2.0.0.ckpt
v1_learning\model-v2.0.0.json
```

If `model-v2.0.0.ckpt` already exists, the next run publishes as `model-v2.0.1.ckpt`, then `model-v2.0.2.ckpt`, and so on.

### Diagnostics: quick checks and audits

Before or after a training run you can run lightweight diagnostics to inspect the run summary and deeper slider audits.

- Quick diagnostic (reads the saved training summary JSON and prints recommendations):

```powershell
uv run python scripts\quick_diagnostic.py
```

- Prediction collapse audit (runs a checkpoint on a parquet split and reports target/predicted spread):

```powershell
uv run python scripts\analyse_prediction_collapse.py `
  --model-path v1_learning\model-v2.0.0.ckpt `
  --parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --output data\audits\prediction_collapse.md `
  --limit 50 `
  --batch-size 16
```

- Dataset diversity audit (scene brightness, contrast, WB, and edit target buckets):

```powershell
uv run python scripts\audit_dataset_diversity.py `
  --parquet v1_learning\dataset\dataset.parquet `
  --output data\audits\dataset_diversity.md
```

- Full all-slider audit (v1.2.3 audit example; this is read-only and produces markdown + parquet outputs in `scripts/output/`):

```powershell
uv run python scripts\audit_all_sliders_v1.2.3.py
```

Notes:
- `quick_diagnostic.py` now discovers training summary JSON files automatically under the project tree and can also list published checkpoints in `v1_learning/` for selection. If an older summary does not include row counts, it reads train/val/test counts from split parquet metadata, falling back to `v1_learning/dataset/splits_v2_stratified/`.
- `audit_all_sliders_v1.2.3.py` now discovers published checkpoints under `v1_learning/model-v*.ckpt` and will prompt you to choose one if multiple are found.
- Audit scripts (like `audit_all_sliders_v1.2.3.py`) are read-only analyses that load a published checkpoint and a test split; they may take longer and require the test parquet to exist. Output goes to `scripts/output/` — keep those reports under version control only when intended.

### Creating a foundation (base) model

When we say "foundation model" here we mean a stable, broadly-trained base checkpoint that you will later fine-tune for specific clients or Mode B presets. To create one:

1. Prepare a large, diverse dataset (see steps in Section 2/3) and ensure train/val/test splits are representative.
2. Run `scripts/train_profile.py` on that dataset to train a production-quality checkpoint. Example (same as above):

```powershell
uv run python scripts\train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-v2-foundation `
  --profile-name "Sonna v2 Foundation" `
  --max-epochs 100 `
  --batch-size 16
```

3. Verify training summary (`data/models/sonna-v2-foundation/training_summary.json`) and run `quick_diagnostic.py` or targeted audits.
4. Publish to the frontend-visible folder (the training CLI does this by default unless `--no-publish` is supplied). The published path under `v1_learning/` is then the base checkpoint you can pass to `scripts/finetune_profile.py --base-model v1_learning/model-vX.Y.Z.ckpt`.

Alternatives:
- To create a Mode B (preset-derived) initial checkpoint, use `scripts/build_mode_b_checkpoint.py --preset <preset.xmp> --survey <survey.json> --base-ckpt <published_base.ckpt>` — this preserves the base backbone and shifts output-head biases and is not a supervised photo-training run.

Summary guidance:
- Use `--resume-from-checkpoint` to recover interrupted training runs (same run continuation).
- Use `scripts/finetune_profile.py --base-model <ckpt>` for deliberate fine-tuning on captured edits.
- Use `scripts/train_profile.py` on a large dataset to create your foundation/base model, then promote/publish and use that published ckpt as the `--base-model` for fine-tuning.

## 5. Train With Explicit Published Version

Use this when you want a specific visible checkpoint name:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-v2-run02 `
  --profile-name "Sonna v2 Candidate 02" `
  --publish-version v2.0.2 `
  --max-epochs 50 `
  --batch-size 16
```

The script refuses to overwrite an existing published checkpoint.

## 6. Train Without Publishing To Frontend

Use this for experiments you do not want visible in Saha:

```powershell
uv run python scripts\train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\scratch-run `
  --slider-set-version v2 `
  --image-resolution 512 `
  --no-publish
```

## 7. Start Backend And Frontend

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
  --original-train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --dry-run
```

Actual fine-tune:

```powershell
uv run python scripts\finetune_profile.py `
  --base-model v1_learning\model-v2.0.0.ckpt `
  --captures-dir data\captures `
  --original-train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --output-dir v1_learning `
  --max-epochs 30 `
  --batch-size 16
```

The fine-tuned checkpoint is versioned under `v1_learning/`, so it becomes visible in the frontend after profile refresh.
