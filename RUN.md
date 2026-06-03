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

It also bootstraps the repo-local runtime layout used by the app and scripts:
`data/`, `data/training_sources/`, `data/raw/`, `data/raw/sonna_training/`,
`data/datasets/`, `data/training_workspace/`, `data/foundation_repo/`,
`v1_learning/`, and `.saha/`.
A fresh clone no longer needs those gitignored folders created by hand.

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

Preset + survey creates a Lite initial checkpoint from the configured foundation
checkpoint, but it is not supervised training from photos.

Foundation has two implemented CLI paths:

- `scripts/train_foundation_model.py` currently trains the existing
  slider-regression model from real Lightroom parameters, then promotes the
  checkpoint into the hidden foundation repo.
- MIT-Adobe FiveK-style training uses an image-to-image foundation trainer from
  `RAW/DNG -> expert TIFF`. Do not turn FiveK TIFF outputs into fake XMP labels.

Every foundation run is versioned. By default, it warm-starts from the active
foundation checkpoint, trains on the new dataset, writes a new checkpoint under
`data/foundation_repo/checkpoints/`, and makes that checkpoint active in
`foundation_manifest.json`. Older checkpoints are kept. If a bad run is
promoted, remove the bad new `.ckpt`; the resolver falls back to the newest
remaining checkpoint. Use `--no-warm-start` only for a deliberate scratch run.

Dataset preparation code paths:

- `scripts/build_dataset.py` -> `src/sonna_editor/data/dataset.py` for RAW + XMP sidecar training.
- `scripts/build_dataset_from_catalog.py` -> `src/sonna_editor/data/catalog_dataset.py` for Lightroom catalog training.
- `src/sonna_editor/data/catalog.py` opens `.lrcat` files read-only and supplies catalog develop settings.
- `src/sonna_editor/data/extract.py` supplies shared RAW previews, metadata, histograms, and AsShot WB for both paths.

So `catalog_dataset.py` is the catalog-based dataset preparation module. It can train without exported XMP sidecars, but it still needs edited catalog develop settings plus accessible RAW files.

### Frontend profile creation

The Saha frontend has two profile creation paths:

- **Personal AI profile:** choose a folder containing RAW files and matching Lightroom `.xmp` sidecars. The backend resolves the hidden foundation checkpoint, builds the dataset, warm-starts training from that foundation, publishes a versioned checkpoint into `v1_learning/`, and streams progress through the normal job API.
- **Lite profile:** choose a Lightroom preset and answer the six-question style survey. The backend derives a `mode_b_initial` checkpoint from the configured foundation checkpoint, the preset, and all six survey answers. The initial Lite run dynamically adjusts Exposure, Temperature, and Tint while preset look sliders stay fixed.

Profile deletion from the frontend asks for confirmation before removing the local checkpoint and sidecar files. Active profiles still cannot be deleted until another profile is activated.

Foundation model training is intentionally **not** exposed in the frontend. Train and promote it with:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-xmp-dir data\training_sources\sonna_personal_001\raw_xmp `
  --workspace-dir data\training_workspace `
  --foundation-repo data\foundation_repo `
  --profile-name "Sonna Parameter Foundation" `
  --version-stem foundation-sonna-parameter-001 `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 4 `
  --init-git
```

For TIFF/image foundation training, use paired folders matched by file stem:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-image-dir data\training_sources\fivek_expert_c\raw_dng `
  --target-tiff-dir data\training_sources\fivek_expert_c\expert_tiff `
  --workspace-dir data\training_workspace `
  --foundation-repo data\foundation_repo `
  --profile-name "Sonna FiveK Image Foundation Expert C" `
  --run-name foundation-fivek-image-expert-c-001 `
  --version-stem foundation-fivek-image-expert-c-001 `
  --image-resolution 512 `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 8
```

This produces an `image_to_image_v1` foundation checkpoint. Mode A still trains
from RAW+XMP; the TIFF checkpoint only warm-starts the visual backbone.

### Lite profile flow

Preset-based profiles are Lite profiles. They do not train from photo labels. They start with the configured foundation checkpoint, read a Lightroom preset plus Lite survey answers, and create a new checkpoint/sidecar package that the UI can select. During initial Lite processing, `process_shoot_model.py` detects `profile_type: mode_b_initial`, uses the preset as the fixed style baseline, and computes per-photo Exposure, Temperature, and Tint corrections before writing XMPs. Preset look sliders such as Contrast, Shadows, Highlights, Whites, Blacks, Saturation, and Vibrance stay fixed from the preset. The foundation checkpoint provides the profile shell and future fine-tuning starting point.

Create a survey JSON:

```powershell
uv run python scripts\run_style_survey.py `
  --output v1_learning\wedding-lite-survey.json `
  --non-interactive `
  --answers exposure=0,temperature=1,tint=0,contrast=0,saturation=0,shadows=0
```

Build and publish a frontend-visible Lite checkpoint. This reads the configured
foundation checkpoint by default:

```powershell
uv run python scripts\build_mode_b_checkpoint.py `
  --preset "D:\Lightroom\Presets\Sonna Wedding.xmp" `
  --survey v1_learning\wedding-lite-survey.json `
  --profile-name "Wedding Lite"
```

Without `--output`, the checkpoint is published as the next available `v1_learning\model-v0.N.0.ckpt`, with a matching `.json` sidecar. The frontend sees it through the same `/api/profiles` scan as trained profiles.

Lite checkpoints are visible in the UI when they are published into `v1_learning/`. The foundation checkpoint stays in the separate foundation repo and is not listed as a frontend profile.

Important: Lite profiles created before the 2026-06-02 Mode B fixes can over-apply the preset because they added the base model's predicted Exposure/colour values on top. Rebuild those old `v1_learning\model-v0.*.ckpt` Lite profiles from the UI or CLI before judging current Mode B output.

Also restart the backend/Electron app after pulling this fix. A running backend
keeps the old Python code loaded, so processing from the UI without a restart can
still write the old double-applied XMPs. In this workspace, the stale
`model-v0.1.0` Lite profile was removed; the corrected replacement is
`v1_learning\model-v0.2.0.ckpt` / `.json`.

Run the published Lite profile with the model-processing CLI:

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v0.2.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput
```

### UI-based Lite profile creation and processing

If you create the Lite profile in the UI, the backend uses the same Lite logic as the CLI. The UI calls:

- `POST /api/profiles/lite` to build the Lite checkpoint from the configured foundation checkpoint, selected preset, and six-question survey answers.
- `POST /api/process` to run a profile on a folder of RAWs and write XMPs.

The backend workflow is:

1. `POST /api/profiles/lite` copies the uploaded preset `.xmp` and survey JSON into `CHECKPOINTS_DIR`.
2. It resolves the configured foundation checkpoint and passes it to `sonna_editor.mode_b.checkpoint_builder.build_mode_b_checkpoint()`.
3. A new Lite checkpoint is written into `v1_learning/model-v0.N.0.ckpt`, and a sidecar JSON is created.
4. The UI discovers the new profile via `GET /api/profiles` and exposes it in the profile selector.

When you process images from the UI:

1. The UI submits `POST /api/process` with `profile_id` and the folder path.
2. The route resolves the profile ID to the checkpoint path and runs `sonna_editor.inference.pipeline.process_shoot_with_model()`.
3. XMPs are written next to the RAWs (or into an output folder), and progress/status is exposed through `GET /api/jobs/{job_id}` and `WS /api/jobs/{job_id}/stream`.

This is the Lite flow that keeps the preset look fixed and writes per-photo Exposure/WB first-pass XMPs. Later fine-tuning can replace the heuristic initial Lite path with a trained checkpoint that learns from Lightroom correction data.

`process_shoot_preset.py` is the direct preset-only path. It does not build a profile checkpoint or publish anything to the UI. It:

- parses a Lightroom preset,
- computes content-aware per-photo adjustments,
- writes XMP sidecars directly next to the source RAWs (or into `--output-dir`),
- supports `--auto-exposure`, `--auto-white-balance`, `--auto-shadow-recovery`, and `--auto-highlight-recovery`.

Use `process_shoot_preset.py` when you want quick preset-derived XMPs without creating a selectable profile. Use Lite checkpoint flow when you want the preset to become a frontend profile that can later be fine-tuned from Lightroom corrections.

### Build dataset from RAW + XMP sidecars

```powershell
uv run python scripts\build_dataset.py `
  --input-dir data\training_sources\sonna_personal_001\raw_xmp `
  --output-dir v1_learning\dataset `
  --profile-name "sonna_current" `
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
  --profile-name "sonna_current" `
  --limit 30000 `
  --workers 4 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

### Train from prepared splits

Use the stratified by-shoot splits and train a fresh Personal AI profile. The current frontend Personal AI path warm-starts from the configured foundation checkpoint. The direct CLI command below starts from scratch unless you pass `--base-model-checkpoint` for a warm start or `--resume-from-checkpoint` for an interrupted-run resume. The current default recipe uses 512px input, direct AsShot WB metadata skip, stronger Temperature/Tint/Exposure loss weights, and frontend publishing into `v1_learning/`.

```bash
uv run python scripts/train_profile.py \
  --train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --test-parquet v1_learning/dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-personal-run01 \
  --profile-name "Sonna Personal Run 01" \
  --batch-size 16 \
  --max-epochs 50
```

Windows PowerShell uses the same command with backticks for line continuation:

```powershell
uv run python scripts/train_profile.py `
  --train-parquet v1_learning\dataset\splits_v2_stratified\train.parquet `
  --val-parquet v1_learning\dataset\splits_v2_stratified\val.parquet `
  --test-parquet v1_learning\dataset\splits_v2_stratified\test.parquet `
  --output-dir data\models\sonna-personal-run01 `
  --profile-name "Sonna Personal Run 01" `
  --batch-size 16 `
  --max-epochs 50
```

The script writes `model.ckpt`, `model.json`, TensorBoard logs, and
`training_summary.json` into the output directory. The exported `model.ckpt`
contains the best validation checkpoint, not just the final epoch.
It also publishes a versioned UI-visible copy such as
`v1_learning/model-v2.0.0.ckpt` plus `v1_learning/model-v2.0.0.json`.
Treat this as an internal file version, not a model-family choice.
Use `--resume-from-checkpoint <path>` to continue from a saved training checkpoint when resuming a run, or omit it to start fresh. Use `--output-dir` for run-specific artifacts, and allow the script to publish the visible checkpoint into `v1_learning/` for frontend discovery.
Use `--base-model-checkpoint <path>` when you want to initialise from the hidden foundation checkpoint without carrying over optimizer or epoch state.

The current small local dataset contains:

```text
v1_learning/dataset/dataset.parquet: 189 rows
splits_v2_stratified/train.parquet: 132 rows
splits_v2_stratified/val.parquet: 27 rows
splits_v2_stratified/test.parquet: 30 rows
```

The current split is grouped by shoot and balanced across Temperature correction, Exposure2012, and Tint correction. Fresh current-recipe training also starts output heads from the training-set target medians and uses geometry-only augmentation by default, which directly addresses the earlier brightness/WB drift in small-data runs.

Fresh current-recipe models use the scene-stat architecture, which adds six preview-derived luminance scene statistics to the metadata path. The default loss recipe prioritises visually important sliders: Exposure=5.0, Temperature/Tint=4.0, Contrast/Highlights/Shadows=3.0, and Whites/Blacks/Saturation/Vibrance=2.0 minimums.

Promotion gate: run `scripts/analyse_prediction_collapse.py` after training. The earlier scene-stat experiment lowered some MAE metrics but collapsed harder than the prior published profile, so it was not kept in `v1_learning/` for frontend use.

Inference also stabilises RGB tone-curve endpoints before writing XMP: per-channel black endpoints stay `0,0` and white endpoints stay `255,255`. This avoids pink/red casts in white highlights caused by model drift in the RGB curve endpoints.

Monitor training:

```bash
uv run tensorboard --logdir data/models/sonna-personal-run01
```
