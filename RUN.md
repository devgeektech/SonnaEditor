# Run Sonna Editor

This is the cross-platform local runbook for macOS, Windows, and Linux.

## 1. Python Environment

The repo is uv-managed and pinned to Python 3.11 through `.python-version`.
On Windows/Linux x86_64, `pyproject.toml` pins `torch` and `torchvision` to the
PyTorch CUDA 12.8 wheel index. This keeps NVIDIA GPU support intact after
`uv sync --extra dev`.

Windows PowerShell:

```powershell
cd C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor
uv sync --extra dev
uv run python --version
```

macOS/Linux shell:

```bash
cd /path/to/SonnaEditor
uv sync --extra dev
uv run python --version
```

If `uv` is not installed yet:

```bash
python -m pip install --user uv
```

Use `python3` instead of `python` on systems where that is the correct Python
launcher.

## 2. Verify

```bash
uv run python scripts/verify_environment.py
```

The verifier checks Python, imports, and the best available PyTorch device:
CUDA, Apple MPS, or CPU. Adobe DNG Converter is reported as optional unless you
need RAW-to-DNG normalisation.

Current verified Windows workstation:

```text
Python 3.11.15
uv 0.11.17
PyTorch 2.11.0+cu128
Preferred torch device: cuda
GPU: NVIDIA GeForce RTX 3050
```

## 3. Backend API

```bash
uv run python scripts/serve.py --port 8765
```

The API should respond at `http://127.0.0.1:8765/api/health`.

## 4. Electron UI

Open a second terminal:

```bash
cd saha-app
npm install
npm run dev
```

Electron starts the React UI and connects it to the backend. In development it
can also spawn the backend itself when `uv` is on `PATH`.

## 5. Optional DNG Converter

For DNG conversion workflows, install Adobe DNG Converter and either use the
default installer path or set:

```bash
SONNA_DNG_CONVERTER=/absolute/path/to/converter
```

PowerShell equivalent:

```powershell
$env:SONNA_DNG_CONVERTER = "C:\Path\To\Adobe DNG Converter.exe"
```

## 6. Train A Profile

Training needs target Lightroom slider values. RAW files alone are not enough
for supervised training. Use one of these dataset sources:

- RAW files with matching exported `.xmp` sidecars.
- A Lightroom Classic `.lrcat` with develop settings and accessible RAW files.
- Fine-tune captures from previous Saha runs.

Preset + survey creates a Mode B initial checkpoint, but it is not supervised
training from photos.

Dataset preparation code paths:

- `scripts/build_dataset.py` -> `src/sonna_editor/data/dataset.py` for RAW + XMP sidecar training.
- `scripts/build_dataset_from_catalog.py` -> `src/sonna_editor/data/catalog_dataset.py` for Lightroom catalog training.
- `src/sonna_editor/data/catalog.py` opens `.lrcat` files read-only and supplies catalog develop settings.
- `src/sonna_editor/data/extract.py` supplies shared RAW previews, metadata, histograms, and AsShot WB for both paths.

So `catalog_dataset.py` is the catalog-based dataset preparation module. It can train without exported XMP sidecars, but it still needs edited catalog develop settings plus accessible RAW files.

### Preset / Mode B profile flow

Preset-based profiles are Mode B/Lite profiles. They do not train from photo labels. They start with a trained Mode A checkpoint, read a Lightroom preset plus six survey answers, and create a new checkpoint whose output biases are calibrated to that preset.

Create a survey JSON:

```powershell
uv run python scripts\run_style_survey.py `
  --output v1_learning\wedding-lite-survey.json `
  --non-interactive `
  --answers exposure=0,temperature=1,tint=0,contrast=1,saturation=-1,shadows=1
```

Build and publish a frontend-visible Mode B checkpoint:

```powershell
uv run python scripts\build_mode_b_checkpoint.py `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --survey v1_learning\wedding-lite-survey.json `
  --base-ckpt v1_learning\model-v1.2.3-prod256.ckpt `
  --profile-name "Wedding Lite"
```

Without `--output`, the checkpoint is published as the next available `v1_learning\model-v0.N.0.ckpt`, with a matching `.json` sidecar. The frontend sees it through the same `/api/profiles` scan as trained profiles.

Run it like any trained model:

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v0.1.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput
```

For quick preset-only XMP output with no checkpoint/profile creation:

```powershell
uv run python scripts\process_shoot_preset.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --output-dir D:\Shoots\ClientShoot01\PresetOutput
```

### Build dataset from RAW + XMP sidecars

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

### Build dataset from Lightroom catalog

Lightroom Classic must be closed. The catalog is opened read-only.

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

### Train from prepared splits

Use the stratified by-shoot splits and train a fresh v2 profile. The current default recipe uses 512px input, direct AsShot WB metadata skip, stronger Temperature/Tint/Exposure loss weights, and frontend publishing into `v1_learning/`.

```bash
uv run python scripts/train_profile.py \
  --train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --test-parquet v1_learning/dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-v2-run01 \
  --profile-name "Sonna v2 Run 01" \
  --slider-set-version v2 \
  --batch-size 16 \
  --max-epochs 50
```

Windows PowerShell uses the same command with backticks for line continuation:

```powershell
uv run python scripts/train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-v2-run01 `
  --profile-name "Sonna v2 Run 01" `
  --slider-set-version v2 `
  --batch-size 16 `
  --max-epochs 50
```

The script writes `model.ckpt`, `model.json`, TensorBoard logs, and
`training_summary.json` into the output directory. The exported `model.ckpt`
contains the best validation checkpoint, not just the final epoch.
It also publishes a versioned UI-visible copy such as
`v1_learning/model-v2.0.0.ckpt` plus `v1_learning/model-v2.0.0.json`.
Use `--resume-from-checkpoint <path>` to continue from a saved training checkpoint when resuming a run, or omit it to start fresh. Use `--output-dir` for run-specific artifacts, and allow the script to publish the visible checkpoint into `v1_learning/` for frontend discovery.

The current small local dataset contains:

```text
v1_learning/dataset/dataset.parquet: 189 rows
splits_v2_stratified/train.parquet: 132 rows
splits_v2_stratified/val.parquet: 27 rows
splits_v2_stratified/test.parquet: 30 rows
```

The current split is grouped by shoot and balanced across Temperature correction, Exposure2012, and Tint correction. Fresh v2 training also starts output heads from the training-set target medians and uses geometry-only augmentation by default, which directly addresses the earlier brightness/WB drift in small-data runs.

Mode A inference also stabilises RGB tone-curve endpoints before writing XMP: per-channel black endpoints stay `0,0` and white endpoints stay `255,255`. This avoids pink/red casts in white highlights caused by model drift in the RGB curve endpoints.

`v1_learning/model-v2.0.0.ckpt` and `v1_learning/model-v2.0.0.json` are present in this workspace at the time of this update, so one local v2 profile should appear in the UI.

Monitor training:

```bash
uv run tensorboard --logdir data/models/sonna-v2-run01
```
