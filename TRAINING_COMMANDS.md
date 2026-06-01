# Sonna Editor Training Commands

This is the practical runbook for preparing a dataset, training a model, and making the trained checkpoint visible in the Saha frontend.

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

Current `scripts/train_profile.py` defaults:

```text
image_resolution=512
lr=1e-4
max_epochs=50
freeze_backbone_epochs=3
Temperature loss weight=6.0
Tint loss weight=6.0
Exposure loss weight=2.0
Temperature bucket loss weight=0.15
Tint bucket loss weight=2.0
Sign-wrong penalty weight=0.2
WB metadata skip=enabled
```

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
