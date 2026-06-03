# Mac Setup And Run Guide

This is the Mac-specific setup and operating checklist for Sonna Editor / Saha.
It assumes Apple Silicon macOS, but the same commands mostly work on Intel Macs.

## 1. Install System Tools

Install Xcode command line tools:

```bash
xcode-select --install
```

Install Homebrew if it is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install the required developer tools:

```bash
brew install git uv python@3.11 node
```

Check the tools:

```bash
git --version
uv --version
python3.11 --version
node --version
npm --version
```

Optional: install Adobe DNG Converter if you need DNG normalisation. Most normal
RAW + XMP training and shoot processing does not require DNG conversion.

```bash
# Optional, only if the default discovery does not find Adobe DNG Converter.
export SONNA_DNG_CONVERTER="/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"
```

## 2. Get The Project

Clone the repo:

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone git@github.com:darshilp16-byte/sonnaeditor.git
cd sonnaeditor
```

If the repo is already copied onto the Mac:

```bash
cd /path/to/SonnaEditor
```

## 3. Install Python Dependencies

Create and sync the uv environment:

```bash
uv sync --extra dev
```

Verify Python and PyTorch:

```bash
uv run python --version
uv run python scripts/verify_environment.py
```

Expected Mac result:

```text
Python 3.11.x
Preferred torch device: mps
MPS available: True
```

If MPS is not available, the app still runs on CPU, but training will be much
slower. Keep training jobs plugged into power.

## 4. Install Frontend Dependencies

```bash
cd saha-app
npm install
cd ..
```

## 5. Start The App

Terminal 1, start the backend:

```bash
uv run python scripts/serve.py --port 8765
```

Check the backend:

```bash
curl http://127.0.0.1:8765/api/health
```

Terminal 2, start the Electron frontend:

```bash
cd saha-app
npm run dev
```

The frontend connects to the backend at `http://127.0.0.1:8765`.

## 6. Verify Profiles

Profiles visible in the frontend are stored under `v1_learning/`:

```bash
ls v1_learning/model-v*.ckpt
ls v1_learning/model-v*.json
```

Check the same list through the backend:

```bash
curl http://127.0.0.1:8765/api/profiles
```

If no profiles are listed, create a Personal AI profile from the frontend or
train/publish one from the CLI.

## 7. Prepare Lightroom Data

Training needs edited Lightroom targets. RAW files alone are not enough.

### RAW + XMP

This can be done from the frontend when creating a Personal AI profile: choose a
folder that contains RAW files with matching `.xmp` sidecars.

CLI equivalent:

```bash
uv run python scripts/build_dataset.py \
  --input-dir /Volumes/Shoots/SonnaTraining \
  --output-dir v1_learning/dataset \
  --profile-name "sonna_v2" \
  --workers 4 \
  --split \
  --val-ratio 0.107 \
  --test-ratio 0.139 \
  --splits-dir-name splits_v2_stratified
```

### Lightroom Catalog

This is currently a CLI path. Close Lightroom first. The catalog is opened
read-only and the RAW files referenced by the catalog must be accessible.

```bash
uv run python scripts/build_dataset_from_catalog.py \
  --catalog-path "/Users/darshil/Pictures/Lightroom/Sonna Catalog.lrcat" \
  --output-dir v1_learning/dataset \
  --profile-name "sonna_v2" \
  --limit 30000 \
  --workers 4 \
  --split \
  --val-ratio 0.107 \
  --test-ratio 0.139 \
  --splits-dir-name splits_v2_stratified
```

Expected outputs:

```text
v1_learning/dataset/dataset.parquet
v1_learning/dataset/thumbnails/
v1_learning/dataset/splits_v2_stratified/train.parquet
v1_learning/dataset/splits_v2_stratified/val.parquet
v1_learning/dataset/splits_v2_stratified/test.parquet
```

## 8. Train A Personal AI Profile

This can be done from the frontend: open the Profiles page, choose Personal AI,
select the RAW + XMP folder, enter the profile name, and start training. The
backend builds the dataset, trains with the production recipe, publishes a
versioned checkpoint into `v1_learning/`, and streams progress to the UI.

CLI equivalent:

```bash
uv run python scripts/train_profile.py \
  --train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --test-parquet v1_learning/dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-personal-run01 \
  --profile-name "Sonna Personal Run 01" \
  --max-epochs 50 \
  --batch-size 16 \
  --num-workers 4
```

Monitor training:

```bash
uv run tensorboard --logdir data/models/sonna-personal-run01
```

Resume an interrupted training run:

```bash
uv run python scripts/train_profile.py \
  --train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --test-parquet v1_learning/dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-personal-run01 \
  --profile-name "Sonna Personal Run 01" \
  --resume-from-checkpoint "data/models/sonna-personal-run01/checkpoints/epoch=...ckpt"
```

## 9. Train A Hidden Foundation Model

This is CLI-only. It is not exposed in the frontend. The foundation checkpoint
is promoted into a separate private foundation repo and is used as the base for
Lite profile creation.

```bash
uv run python scripts/train_foundation_model.py \
  --raw-xmp-dir "$HOME/SonnaEditorTraining/raw/edited-with-xmp" \
  --workspace-dir "$HOME/SonnaEditorTraining/workspace" \
  --foundation-repo "$HOME/Projects/SonnaEditorFoundation" \
  --profile-name "Sonna Foundation" \
  --version-stem foundation-current \
  --max-epochs 100 \
  --batch-size 16 \
  --workers 4 \
  --init-git
```

The foundation repo contains `foundation_manifest.json` and
`checkpoints/foundation-current.ckpt`. Install Git LFS before pushing the
foundation repo to GitHub:

```bash
brew install git-lfs
git lfs install
cd ~/Projects/SonnaEditorFoundation
git add .
git commit -m "Add foundation checkpoint"
```

## 10. Create A Lite Profile

This can be done from the frontend: open Profiles, choose Lite, select a
Lightroom preset, answer the six-question style survey, and create
the profile. Lite uses the configured foundation checkpoint, not the active
Personal AI profile. Initial Lite processing dynamically adjusts Exposure,
Temperature, and Tint while preset look sliders stay fixed.

CLI equivalent:

```bash
uv run python scripts/run_style_survey.py \
  --output v1_learning/wedding-lite-survey.json \
  --non-interactive \
  --answers exposure=0,temperature=1,tint=0,contrast=0,saturation=0,shadows=0
```

```bash
uv run python scripts/build_mode_b_checkpoint.py \
  --preset "/Users/darshil/Lightroom/Presets/Sonna Wedding.xmp" \
  --survey v1_learning/wedding-lite-survey.json \
  --profile-name "Wedding Lite"
```

Without `--output`, the Lite checkpoint is published to the next available
`v1_learning/model-v0.N.0.ckpt` path and appears in the frontend.

## 11. Process A Shoot

This can be done from the frontend: choose Process Shoot, pick the input folder,
choose a profile, and start processing. XMP sidecars are written next to RAWs or
into the selected output folder, depending on the UI options.

CLI equivalent:

```bash
uv run python scripts/process_shoot_model.py \
  --input-dir "/Volumes/Shoots/ClientShoot01" \
  --model-path v1_learning/model-v2.0.0.ckpt \
  --output-dir "/Volumes/Shoots/ClientShoot01/SahaOutput"
```

Direct preset-only processing is CLI-only and does not create a frontend
profile:

```bash
uv run python scripts/process_shoot_preset.py \
  --input-dir "/Volumes/Shoots/ClientShoot01" \
  --preset "/Users/darshil/Lightroom/Presets/Sonna Wedding.xmp" \
  --output-dir "/Volumes/Shoots/ClientShoot01/SahaPresetOutput" \
  --auto-exposure \
  --auto-white-balance
```

## 12. Fine-Tune A Profile

Fine-tuning can be started from the frontend when capture data is available.
Use this after processing a shoot, editing the XMPs in Lightroom, and keeping the
`sonna_predictions.json` from the original Saha run.

CLI dry run:

```bash
uv run python scripts/finetune_profile.py \
  --base-model v1_learning/model-v2.0.0.ckpt \
  --captures-dir data/captures \
  --original-train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --dry-run
```

CLI actual fine-tune:

```bash
uv run python scripts/finetune_profile.py \
  --base-model v1_learning/model-v2.0.0.ckpt \
  --captures-dir data/captures \
  --original-train-parquet v1_learning/dataset/splits_v2_stratified/train.parquet \
  --val-parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --output-dir v1_learning \
  --max-epochs 30 \
  --batch-size 16
```

The fine-tuned checkpoint is versioned under `v1_learning/` and becomes visible
after refreshing profiles in the frontend.

## 13. Diagnostics

Quick training summary check:

```bash
uv run python scripts/quick_diagnostic.py
```

Prediction collapse audit:

```bash
uv run python scripts/analyse_prediction_collapse.py \
  --model-path v1_learning/model-v2.0.0.ckpt \
  --parquet v1_learning/dataset/splits_v2_stratified/val.parquet \
  --output data/audits/prediction_collapse.md \
  --limit 50 \
  --batch-size 16
```

Dataset diversity audit:

```bash
uv run python scripts/audit_dataset_diversity.py \
  --parquet v1_learning/dataset/dataset.parquet \
  --output data/audits/dataset_diversity.md
```

Run tests and frontend build:

```bash
uv run ruff check .
uv run pytest tests -q
cd saha-app
npm run build:vite
cd ..
```

Known local caveat: some full-suite tests require gitignored RAW/XMP fixtures in
`tests/fixtures/`. If those fixtures are missing on the Mac, run focused tests or
restore the fixture files before treating the full suite as a release gate.

## 14. Updating The App

Pull latest code:

```bash
git pull --ff-only
uv sync --extra dev
cd saha-app
npm install
cd ..
```

Restart both backend and frontend after pulling code. A live backend keeps old
Python code in memory.

## 15. Mac Notes

- Keep Lightroom closed for catalog reads.
- Never write to the `.lrcat`; the catalog reader opens it read-only.
- Never move or overwrite original RAW files.
- Do not overwrite checkpoints. The training scripts publish new versions.
- Use `v1_learning/` only for profiles that should appear in the frontend.
- Use `data/models/` for hidden runs, diagnostics, and unpromoted candidates.
- On Apple Silicon, the preferred device should be `mps`; CUDA is only expected
  on Windows/Linux NVIDIA machines.
