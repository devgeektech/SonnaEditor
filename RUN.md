# Run Sonna Editor

This is the cross-platform local runbook for macOS, Windows, and Linux.

## 1. Python Environment

The repo is uv-managed and pinned to Python `3.11.*` through `.python-version`,
`pyproject.toml`, and `uv.lock`. Direct runtime/dev dependencies are exact-pinned
in `pyproject.toml` to reduce Mac resolver drift. Auto straightening uses the
headless OpenCV runtime package (`opencv-python-headless`) for preview geometry.
On Windows/Linux x86_64, `torch` and `torchvision` resolve from the PyTorch CUDA
12.8 wheel index, which keeps NVIDIA GPU support intact after `uv sync --extra
dev`. macOS resolves the matching public PyTorch wheels.

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
launcher. If uv picks Python 3.12 or newer, force 3.11 with:

```bash
uv python pin 3.11
uv sync --extra dev
```

## 2. Verify

```bash
uv run python scripts/verify_environment.py
```

The verifier checks Python, imports, and the best available PyTorch device:
CUDA, Apple MPS, or CPU. Adobe DNG Converter is reported as optional unless you
need RAW-to-DNG normalisation.

It also bootstraps the repo-local runtime layout used by the app and scripts:
`data/`, `data/training_sources/`, `data/raw/`, `data/raw/sonna_training/`,
`data/datasets/`, `data/training_workspace/`,
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

## 3. Start The App

From the repo root, use the one-command launcher.

Windows PowerShell:

```powershell
.\run_saha.cmd
```

macOS/Linux shell:

```bash
bash run_saha.sh
```

The launcher creates the repo-local runtime folders, installs frontend npm
dependencies if `saha-app/node_modules/` is missing, then runs the Electron dev
app. Electron reuses an existing backend on port `8765` or starts
`scripts/serve.py` itself and shuts it down when the app quits. Backend startup
readiness waits up to 30s by default, which avoids false failures on cold
Windows/CUDA startup; set `SAHA_BACKEND_STARTUP_TIMEOUT_MS` only if a slower
machine needs a longer wait.

Prerequisites for both Windows and macOS: `uv` and Node.js LTS must be on
`PATH`. If either is missing, the launcher prints a setup message instead of
failing deep inside the app startup.

For a fully explicit command, the wrappers call:

```bash
uv run python scripts/run_app.py
```

Pass `--skip-install` if you want the launcher to fail instead of running
`npm install` when frontend dependencies are missing.

PowerShell users can also run `.\run_saha.ps1`; the `.cmd` wrapper is the
lowest-friction Windows option because it avoids script execution-policy
prompts.

## 4. Manual Backend API

```bash
uv run python scripts/serve.py --port 8765
```

The API should respond at `http://127.0.0.1:8765/api/health`.

## 5. Manual Electron UI

Open a second terminal:

```bash
cd saha-app
npm install
npm run dev
```

Electron starts the React UI and connects it to the backend. In development it
can also spawn the backend itself when `uv` is on `PATH`.

## 6. Optional DNG Converter

For DNG conversion workflows, install Adobe DNG Converter and either use the
default installer path or set:

```bash
SONNA_DNG_CONVERTER=/absolute/path/to/converter
```

PowerShell equivalent:

```powershell
$env:SONNA_DNG_CONVERTER = "C:\Path\To\Adobe DNG Converter.exe"
```

## 7. Train A Profile

Training needs target Lightroom slider values. RAW files alone are not enough
for supervised training. Use one of these dataset sources:

- RAW files with matching exported `.xmp` sidecars.
- A Lightroom Classic `.lrcat` with develop settings and accessible RAW files.
- Fine-tune captures from previous Saha runs.

The app scans the same RAW extension set everywhere: `.cr2`, `.cr3`, `.nef`,
`.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and `.srw`.
Actual extraction still depends on `rawpy`/LibRaw support for the specific
camera file; optional DNG conversion depends on Adobe DNG Converter support.

Preset + survey creates a Lite initial checkpoint from the configured foundation
checkpoint, but it is not supervised training from photos.

Foundation training uses `scripts/train_foundation_model.py` to train the
existing slider-regression model from real Lightroom parameters, then promotes
the checkpoint into the hidden repo-local foundation folder. Targets can come
from RAW+XMP sidecars or prepared catalog-derived splits.

Every foundation run is versioned. By default, it warm-starts from the active
foundation checkpoint, trains on the new dataset, writes a new checkpoint under
`SonnaEditorFoundation\checkpoints\`, and makes that checkpoint active in
`foundation_manifest.json`. The active foundation checkpoint is cumulative:
catalog and RAW+XMP runs update the same native SonnaEditor slider-regression
model. Older checkpoints are kept. New runs auto-promote as `foundation-vN` unless a
version stem is supplied. If a bad run is promoted, roll back the active
manifest pointer with `scripts\rollback_foundation.py` rather than deleting
checkpoints. Use `--no-warm-start` only for a deliberate scratch run.

Foundation training now uses adaptive capacity. Larger splits start with the
final ConvNeXt stage trainable by default (`--backbone-trainable-layers
stage:7`) plus the normal feature fusion/metadata/output-head layers. Small
splits below 500 train rows automatically use `--backbone-unfreeze-strategy
custom --backbone-trainable-layers none` unless explicit backbone flags are
supplied. Startup logs print total/trainable/frozen parameters, dataset size,
batches per epoch, estimated optimizer steps, learning rates, sampler/cap
status, and the backbone freeze summary. Use `--backbone-trainable-layers
block:7:2,stage:6` for an ~8M trainable ablation, `block:7:1-2,stage:6` for
~12M, or `--backbone-unfreeze-strategy custom` to keep a spec fixed for a full
run.

Foundation promotion has guardrails. The CLI refuses normal foundation
training/promotion from fewer than 75 train rows, and foundation runs select the
exported checkpoint by `val_visual_score` rather than plain `val_loss` so the
best epoch is balanced across key visible sliders. The quality gate is tiered:
hard failures block promotion, while moderate misses are stored as warnings and
require visual review. Overrides exist for deliberate reviewed experiments
only: `--allow-small-foundation-dataset` and `--allow-quality-gate-failure`.

Foundation summaries now record `quality_gate_passed`,
`foundation_quality_failures`, `foundation_quality_warnings`,
`checkpoint_monitor`, and `best_checkpoint_score`. Run
`scripts/quick_diagnostic.py --summary-path <training_summary.json>` after a
run; it prints backbone capacity, field-loss overrides, all-parameter MAE, the
gate result, and a train-median baseline comparison for failed gate fields when
the split Parquets are available.

If a run misses tone/presence metrics but the dataset audit looks healthy, retry
from the same prepared splits with repeatable `--field-loss-weight FIELD=WEIGHT`
overrides or the reviewed `--tone-presence-retry` shortcut before collecting
another dataset. This creates a new run and warm-starts model weights from the
active foundation by default. It is not `--resume-from-checkpoint`; resume is
only for continuing an interrupted run from that same run's Lightning
checkpoint.

Current active foundation note: `foundation-sonna-raw-xmp-001` was rolled back
after diagnostics showed only 132 train rows, overfitting, and collapsed
Highlights/Shadows. The active manifest now points to
`foundation-fivek-catalog-expert-c-001`.

When copying foundation commands, change both `--run-name` and `--version-stem`
for every new run. Keep them the same, for example
`foundation-fivek-catalog-expert-c-001` then
`foundation-fivek-catalog-expert-c-002`. Reusing an old version stem is expected
to fail because foundation checkpoints are never overwritten. Omit
`--version-stem` if you want the system to auto-allocate `foundation-vN`.

Dataset preparation code paths:

- `scripts/build_dataset.py` -> `src/sonna_editor/data/dataset.py` for RAW + XMP sidecar training.
- `scripts/build_dataset_from_catalog.py` -> `src/sonna_editor/data/catalog_dataset.py` for Lightroom catalog training.
- `src/sonna_editor/data/catalog.py` opens `.lrcat` files read-only and supplies catalog develop settings.
- `src/sonna_editor/data/extract.py` supplies shared RAW previews, metadata, histograms, and AsShot WB for both paths.

So `catalog_dataset.py` is the catalog-based dataset preparation module. It can train without exported XMP sidecars, but it still needs edited catalog develop settings plus accessible RAW files.

### Frontend profile creation

The Saha frontend has two profile creation paths:

- **Personal AI profile:** choose a folder containing RAW files and matching Lightroom `.xmp` sidecars. The backend resolves the hidden foundation checkpoint, builds the dataset, warm-starts training from that foundation, publishes a versioned checkpoint into `v1_learning/`, and streams progress through the normal job API.
- **Lite profile:** choose a Lightroom preset and answer the six-question style survey. The backend derives a `mode_b_initial` checkpoint from the configured foundation checkpoint, the preset, and all six answers. The initial Lite run dynamically adjusts Exposure, Temperature, and Tint while preset look sliders stay fixed. Tint calibration is deliberately gentle: the strongest survey answer maps to 10 Lightroom tint units, and per-photo Tint correction uses green-vs-magenta balance.

Profile deletion from the frontend asks for confirmation before removing the local checkpoint and sidecar files. Active profiles still cannot be deleted until another profile is activated.

Foundation model training is intentionally **not** exposed in the frontend. Train and promote it with:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-xmp-dir data\training_sources\sonna_personal_001\raw_xmp `
  --workspace-dir data\training_workspace `
  --foundation-repo SonnaEditorFoundation `
  --profile-name "Sonna Parameter Foundation" `
  --version-stem foundation-sonna-parameter-001 `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 4
```

Tone/presence focused retry from prepared RAW+XMP splits:

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

For FiveK, build Expert C catalog splits first:

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
  --splits-dir-name splits_v2_stratified
```

Then train from those prepared splits:

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

For the next run, change both `--run-name` and `--version-stem`, for example
`foundation-fivek-catalog-expert-c-002`. Do not reuse an old version stem.
The promoted checkpoint becomes the default base model through
`SonnaEditorFoundation\foundation_manifest.json`; Mode A and Mode B use that
active pointer unless an environment override is set.

The inspected FiveK download at
`C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset`
contains the 5,000 DNG source files and `raw_photos\fivek.lrcat`. The Lightroom
catalog contains usable slider targets as virtual-copy
develop settings: 60,000 catalog rows over 5,000 DNGs, with collections `A` to
`E` holding the five expert variants. For a first catalog-based slider
foundation experiment, build only Expert C. Do not mix all 60,000 FiveK catalog
rows in one unconditioned slider model.

The FiveK path trains `RAW preview + metadata -> Lightroom slider values` from
catalog develop settings. It does not use rendered target images. The verified
local catalog has no blocking lock files, and a 20-row Expert C smoke build
completed with 0 missing files and 0 parse errors.

On the Windows RTX 3050 workstation, start foundation runs at `--batch-size 8`.
The RAW+XMP foundation CLI automatically retries with smaller batch sizes after
CUDA memory failures.

After a foundation checkpoint passes audit, commit and push
`SonnaEditorFoundation\foundation_manifest.json` plus the matching
`SonnaEditorFoundation\checkpoints\<version>.ckpt` and `.json` sidecar. The
`.ckpt` file is handled by Git LFS. Run `git lfs install` once per machine;
after that, normal `git push` uploads checkpoint binaries automatically. Use
`git lfs status` / `git lfs ls-files` before pushing if you want to confirm the
checkpoint is LFS-managed. On a new machine, run `git lfs pull` after clone/pull
to fetch the real checkpoint contents. Do not push `data\training_workspace\`.

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

Lite checkpoints are visible in the UI when they are published into `v1_learning/`. The foundation checkpoint stays in the repo-local hidden foundation folder and is not listed as a frontend profile.

Important: Lite profiles created before the 2026-06-02 Mode B fixes can over-apply the preset because they added the base model's predicted Exposure/colour values on top. Rebuild those old `v1_learning\model-v0.*.ckpt` Lite profiles from the UI or CLI before judging current Mode B output.

Also restart the backend/Electron app after pulling this fix. A running backend
keeps the old Python code loaded, so processing from the UI without a restart can
still write the old double-applied XMPs. Rebuild any old `model-v0.*.ckpt` Lite
profile with the current code before judging Lite output.

Run the published Lite profile with the model-processing CLI:

```powershell
uv run python scripts\process_shoot_model.py `
  --input-dir D:\Shoots\ClientShoot01 `
  --model-path v1_learning\model-v0.2.0.ckpt `
  --output-dir D:\Shoots\ClientShoot01\SahaOutput `
  --auto-straighten
```

`--auto-straighten` is optional and matches the Process UI checkbox. It uses
OpenCV Canny + Hough/LSD line detection on the extracted preview, then scores
horizon, architecture, and mixed-axis evidence before writing Lightroom
Transform metadata (`PerspectiveRotate` plus minimal `PerspectiveScale`). It
does not retrain or alter the selected profile checkpoint. This branch avoids
`CropAngle` and crop bounds entirely for A/B testing against `Auto_Straighten`.
`sonna_predictions.json` records
`straightening_engine`, `scene_type`, `horizon_score`, `axis_score`,
horizontal/vertical line counts, `line_count`, and `line_length_px` for each
photo so a batch can be audited if Lightroom does not appear to show
straightening.

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

Generated datasets belong under `data/training_workspace`. Keep `v1_learning`
for frontend-visible checkpoint and sidecar files only.

```powershell
uv run python scripts\build_dataset.py `
  --input-dir data\training_sources\sonna_personal_001\raw_xmp `
  --output-dir data\training_workspace\sonna_personal_001_dataset `
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
  --output-dir data\training_workspace\sonna_personal_001_dataset `
  --profile-name "sonna_current" `
  --limit 30000 `
  --workers 4 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

Optional flags:

```text
--collection-name "C"            Include only one Lightroom collection.
--include-unedited-looking       Keep sparse catalog rows that look unedited.
                                 Use for FiveK expert collections, not normal Sonna catalogs.
```

### Train from prepared splits

Use the stratified by-shoot splits and train a fresh Personal AI profile. The current frontend Personal AI path warm-starts from the configured foundation checkpoint. The direct CLI command below starts from scratch unless you pass `--base-model-checkpoint` for a warm start or `--resume-from-checkpoint` for an interrupted-run resume. The current default recipe uses 512px input, direct AsShot WB metadata skip, stronger Temperature/Tint/Exposure loss weights, and frontend publishing into `v1_learning/`.

```bash
uv run python scripts/train_profile.py \
  --train-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet \
  --val-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
  --test-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-personal-run01 \
  --profile-name "Sonna Personal Run 01" \
  --batch-size 16 \
  --max-epochs 50
```

Windows PowerShell uses the same command with backticks for line continuation:

```powershell
uv run python scripts/train_profile.py `
  --train-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet `
  --val-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\val.parquet `
  --test-parquet data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\test.parquet `
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

The earlier diagnostic small dataset contained:

```text
data/training_workspace/sonna_personal_001_dataset/dataset.parquet: 189 rows
splits_v2_stratified/train.parquet: 132 rows
splits_v2_stratified/val.parquet: 27 rows
splits_v2_stratified/test.parquet: 30 rows
```

That dataset/checkpoint cache was later cleared for a fresh data reset, so do not assume those files exist locally. New splits are still grouped by shoot and balanced across Temperature correction, Exposure2012, and Tint correction. Fresh current-recipe training also starts output heads from the training-set target medians and uses geometry-only augmentation by default, which directly addresses the earlier brightness/WB drift in small-data runs.

Fresh current-recipe models use the scene-stat architecture, which adds six preview-derived luminance scene statistics to the metadata path. The default loss recipe prioritises visually important sliders: Exposure=5.0, Temperature/Tint=4.0, Contrast/Highlights/Shadows=3.0, and Whites/Blacks/Saturation/Vibrance=2.0 minimums.

Promotion gate: run `scripts/analyse_prediction_collapse.py` after training. The earlier scene-stat experiment lowered some MAE metrics but collapsed harder than the prior published profile, so it was not kept in `v1_learning/` for frontend use.

Inference also stabilises RGB tone-curve endpoints before writing XMP: per-channel black endpoints stay `0,0` and white endpoints stay `255,255`. This avoids pink/red casts in white highlights caused by model drift in the RGB curve endpoints.

Monitor training:

```bash
uv run tensorboard --logdir data/models/sonna-personal-run01
```

