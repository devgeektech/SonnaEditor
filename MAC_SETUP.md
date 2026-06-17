# Mac Setup And Run Guide

This is the Mac-specific setup and operating checklist for Sonna Editor / Saha.
It assumes Apple Silicon macOS, VS Code, and the default `zsh` shell. The same
commands mostly work on Intel Macs.

## Command Syntax Rules For Mac

All commands in this file are written for `zsh`/bash in VS Code's integrated
terminal or the macOS Terminal app.

- Use forward slashes in paths: `scripts/train_foundation_model.py`
- Use a trailing backslash for line continuation: `\`
- Do not use Windows PowerShell backticks: `` ` ``
- Do not paste Windows paths like `scripts\train_foundation_model.py` into zsh.
  zsh treats the backslash as an escape and can turn that into
  `scriptstrain_foundation_model.py`.

If zsh prints `command not found: --splits-dir`, the command was split
incorrectly. Re-copy the Mac/zsh command block and make sure every continued
line except the last ends with `\`.

## 1. Install System Tools

Install Apple's command line tools. You do not need the full Xcode app for this
project; this just gives macOS the compiler/git support that Python packages and
developer tools expect.

```bash
xcode-select --install
```

Install Homebrew if it is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install the required developer tools:

```bash
brew install git git-lfs uv python@3.11 node
git lfs install
```

Install VS Code if it is not already installed:

```bash
brew install --cask visual-studio-code
```

Check the tools:

```bash
git --version
uv --version
python3.11 --version
node --version
npm --version
git lfs version
```

Git LFS is required because model checkpoint files are large binary files. The
repo tracks checkpoint paths through `.gitattributes`, including:

```text
SonnaEditorFoundation/checkpoints/*.ckpt
v1_learning/*.ckpt
models/**/*.ckpt
```

You do not need a separate upload command for checkpoints. After `git lfs
install`, a normal `git push` uploads the small git pointer files and the large
`.ckpt` contents to Git LFS automatically. On a fresh Mac clone, run `git lfs
pull` after cloning if checkpoint files are still pointer files.

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
git lfs pull
```

If the repo is already copied onto the Mac:

```bash
cd /path/to/SonnaEditor
```

Open the project in VS Code:

```bash
code .
```

Use VS Code's integrated terminal for the rest of the commands. It should be
running `zsh` by default; confirm with:

```bash
echo $SHELL
```

## 3. Install Python Dependencies

Create and sync the uv environment:

```bash
uv sync --extra dev
```

This project is pinned to Python `3.11.*` in `pyproject.toml` / `uv.lock`.
If uv tries to use Python 3.12 or newer, force Python 3.11:

```bash
uv python pin 3.11
uv sync --extra dev
```

Direct runtime and dev dependencies are exact-pinned in `pyproject.toml` to
reduce Mac resolver drift. macOS installs the public PyTorch wheels
(`torch==2.11.0`, `torchvision==0.26.0`); Windows/Linux x86_64 use the same
public versions but resolve CUDA 12.8 local wheels from the configured PyTorch
index.

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

The repo auto-creates its runtime folders on backend or CLI startup. The
important local structure is:

```text
data/
  training_sources/       # local source learning photos, one child folder per dataset
  training_workspace/     # generated datasets, splits, audits, foundation runs
  models/                 # unpromoted Personal AI experiments
  audits/                 # collapse/diversity/diagnostic reports
v1_learning/              # frontend-visible profiles only
SonnaEditorFoundation/    # hidden foundation manifest + versioned checkpoints
.saha/                    # repo-local app state, jobs, active profile
```

Keep source photos out of `v1_learning/`. That folder is only for profiles the
frontend should list.

## 4. Install Frontend Dependencies

```bash
cd saha-app
npm install
cd ..
```

The one-command launcher in the next section also runs `npm install`
automatically when `saha-app/node_modules/` is missing. Keeping this explicit
command here is useful for setup checks and troubleshooting.

## 5. Start The App

Preferred Mac/zsh command from the repo root:

```bash
bash run_saha.sh
```

This creates the repo-local runtime folders, checks for `uv` and Node/npm,
installs frontend dependencies if needed, then starts the Electron app. The
Electron main process starts or reuses the FastAPI backend on
`http://127.0.0.1:8765` and shuts down its own backend process when the app
quits.

The explicit equivalent is:

```bash
uv run python scripts/run_app.py
```

Use this only if you want to pass launcher flags such as:

```bash
uv run python scripts/run_app.py --skip-install
```

### Legacy Two-Terminal Reference

Keep this path for debugging backend/frontend issues or reading backend logs
separately.

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
npm install
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

The hidden foundation checkpoint is not listed as a frontend profile. Check it
separately:

```bash
uv run python scripts/rollback_foundation.py --list
```

## 7. Prepare Lightroom Data

Training needs edited Lightroom targets. RAW files alone are not enough.
Generated datasets belong under `data/training_workspace`. Keep `v1_learning`
for frontend-visible checkpoint and sidecar files only.

### RAW + XMP

This can be done from the frontend when creating a Personal AI profile: choose a
folder that contains RAW files with matching `.xmp` sidecars.

Recommended Mac source folder:

```text
data/training_sources/sonna_personal_001/raw_xmp/
```

CLI equivalent:

```bash
uv run python scripts/build_dataset.py \
  --input-dir data/training_sources/sonna_personal_001/raw_xmp \
  --output-dir data/training_workspace/sonna_personal_001_dataset \
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
Ordinary catalog builds skip unedited-looking rows by default. Only use
`--include-unedited-looking` for sparse trusted datasets such as FiveK expert
collections, not normal Sonna catalogs.

```bash
uv run python scripts/build_dataset_from_catalog.py \
  --catalog-path "/Users/darshil/Pictures/Lightroom/Sonna Catalog.lrcat" \
  --output-dir data/training_workspace/sonna_personal_001_dataset \
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
data/training_workspace/sonna_personal_001_dataset/dataset.parquet
data/training_workspace/sonna_personal_001_dataset/thumbnails/
data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet
data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet
data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/test.parquet
```

## 8. Train A Personal AI Profile

This can be done from the frontend: open the Profiles page, choose Personal AI,
select the RAW + XMP folder, enter the profile name, and start training. The
backend builds the dataset, trains with the production recipe, publishes a
versioned checkpoint into `v1_learning/`, and streams progress to the UI.

Personal AI training resolves the active hidden foundation checkpoint when one
is configured. Warm-started runs keep learned foundation weights but recalibrate
final output-head biases from the current training split, so stale foundation
colour/exposure priors do not dominate the new profile.

CLI equivalent:

```bash
uv run python scripts/train_profile.py \
  --train-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet \
  --val-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
  --test-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/test.parquet \
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
  --train-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet \
  --val-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
  --test-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/test.parquet \
  --output-dir data/models/sonna-personal-run01 \
  --profile-name "Sonna Personal Run 01" \
  --resume-from-checkpoint "data/models/sonna-personal-run01/checkpoints/epoch=...ckpt"
```

## 9. Train A Hidden Foundation Model

This is CLI-only. It is not exposed in the frontend. The foundation checkpoint
is promoted into the repo-local hidden `SonnaEditorFoundation/` folder and is
used as the base for Lite profile creation.

Foundation promotion is guarded. The CLI refuses normal foundation updates from
fewer than 75 train rows and blocks promotion when held-out metrics fail. Small
foundation splits below 500 train rows automatically train with the ConvNeXt
backbone frozen (`--backbone-unfreeze-strategy custom
--backbone-trainable-layers none`) unless you explicitly pass different
backbone flags for a reviewed ablation. Use `--allow-small-foundation-dataset`
or `--allow-quality-gate-failure` only for deliberate reviewed experiments, not
routine active foundation updates.

If a foundation run is close on white balance but misses tone/presence metrics,
retry from the same audited splits with repeatable
`--field-loss-weight FIELD=WEIGHT` overrides before collecting a new dataset.
This starts a new run and warm-starts model weights from the active foundation
checkpoint by default. It is not `--resume-from-checkpoint`; use resume only for
an interrupted run from that same run's `checkpoints/last.ckpt`.

For RAW+XMP foundation data, first export metadata from Lightroom and keep the
edited RAW/DNG files plus same-stem `.xmp` sidecars in a dedicated source
folder. For important runs, build and audit splits before training:

```bash
uv run python scripts/build_dataset.py \
  --input-dir data/training_sources/sonna_foundation_001/raw_xmp \
  --output-dir data/training_workspace/sonna_foundation_001_dataset \
  --profile-name "sonna_foundation_001" \
  --workers 4 \
  --split \
  --val-ratio 0.107 \
  --test-ratio 0.139 \
  --splits-dir-name splits_v2_stratified
```

```bash
uv run python scripts/audit_catalog.py \
  --parquet-path data/training_workspace/sonna_foundation_001_dataset/dataset.parquet \
  --output-dir data/training_workspace/sonna_foundation_001_dataset/audit
```

Also audit scene/edit diversity:

```bash
uv run python scripts/audit_dataset_diversity.py \
  --parquet data/training_workspace/sonna_foundation_001_dataset/dataset.parquet \
  --output data/training_workspace/sonna_foundation_001_dataset/dataset_diversity.md
```

Train from the inspected splits:

Quick preflight from the repo root:

```bash
pwd
ls scripts/train_foundation_model.py
ls data/training_workspace/sonna_foundation_001_dataset/splits_v2_stratified
```

The splits folder should contain `train.parquet`, `val.parquet`, and
`test.parquet`.

```bash
uv run python scripts/train_foundation_model.py \
  --splits-dir data/training_workspace/sonna_foundation_001_dataset/splits_v2_stratified \
  --workspace-dir data/training_workspace \
  --foundation-repo SonnaEditorFoundation \
  --profile-name "Sonna RAW XMP Foundation" \
  --run-name foundation-sonna-raw-xmp-001 \
  --version-stem foundation-sonna-raw-xmp-001 \
  --max-epochs 100 \
  --batch-size 8 \
  --workers 8
```

Tone/presence focused retry from the same splits:

```bash
uv run python scripts/train_foundation_model.py \
  --splits-dir data/training_workspace/sonna_foundation_001_dataset/splits_v2_stratified \
  --workspace-dir data/training_workspace \
  --foundation-repo SonnaEditorFoundation \
  --profile-name "Sonna RAW XMP Foundation Tone Presence 002" \
  --run-name foundation-sonna-raw-xmp-002-tone-presence \
  --version-stem foundation-sonna-raw-xmp-002-tone-presence \
  --max-epochs 150 \
  --batch-size 8 \
  --workers 8 \
  --field-loss-weight Exposure2012=7 \
  --field-loss-weight Whites2012=6 \
  --field-loss-weight Blacks2012=6 \
  --field-loss-weight Highlights2012=5 \
  --field-loss-weight Shadows2012=4 \
  --field-loss-weight Vibrance=4 \
  --field-loss-weight Saturation=4
```

The foundation folder contains `foundation_manifest.json` and
`checkpoints/foundation-sonna-raw-xmp-001.ckpt`. It is tracked by the parent
repo, with checkpoint binaries routed through Git LFS.

Run diagnostics before trusting or pushing a new foundation:

```bash
uv run python scripts/quick_diagnostic.py \
  --summary-path data/training_workspace/foundation_runs/foundation-sonna-raw-xmp-001/training/training_summary.json
```

```bash
uv run python scripts/analyse_prediction_collapse.py \
  --model-path SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-001.ckpt \
  --parquet data/training_workspace/sonna_foundation_001_dataset/splits_v2_stratified/val.parquet \
  --output data/audits/foundation-sonna-raw-xmp-001-collapse.md \
  --limit 200 \
  --batch-size 16
```

Confirm the active foundation pointer:

```bash
uv run python scripts/rollback_foundation.py --list
```

Commit only the promoted hidden foundation files, not `data/training_workspace/`:

```bash
git lfs status
git add .gitattributes \
        SonnaEditorFoundation/foundation_manifest.json \
        SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-001.ckpt \
        SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-001.json
git commit -m "train foundation checkpoint foundation-sonna-raw-xmp-001"
git push
```

That normal `git push` uploads the checkpoint binary through Git LFS. To confirm
the checkpoint is LFS-managed before pushing, run:

```bash
git lfs ls-files
```

On Apple Silicon, training uses MPS when available. The CUDA auto-batch retry is
mainly for Windows/Linux NVIDIA machines; on Mac, lower `--batch-size` manually
if memory pressure appears.

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

This can be done from the frontend: choose Process Shoot, add either a RAW
folder or a Lightroom `.lrcat` catalog, choose a profile, select the queued
source, and start processing. Folder processing keeps the existing RAW-folder
flow. Catalog processing opens the `.lrcat` read-only to discover accessible RAW
paths, runs the same Mode A/Mode B inference path, writes XMP sidecars next to
RAWs, and writes `sonna_lightroom_edits.lua` beside `sonna_predictions.json`.

To refresh an already-open Lightroom catalog, add
`lightroom/SahaBridge.lrplugin` in Lightroom Classic's Plug-in Manager, then
use its Import Saha Edits command to select `sonna_lightroom_edits.lua`.

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
  --original-train-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet \
  --val-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
  --dry-run
```

CLI actual fine-tune:

```bash
uv run python scripts/finetune_profile.py \
  --base-model v1_learning/model-v2.0.0.ckpt \
  --captures-dir data/captures \
  --original-train-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/train.parquet \
  --val-parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
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

Specific training summary:

```bash
uv run python scripts/quick_diagnostic.py \
  --summary-path data/models/sonna-personal-run01/training_summary.json
```

Prediction collapse audit:

```bash
uv run python scripts/analyse_prediction_collapse.py \
  --model-path v1_learning/model-v2.0.0.ckpt \
  --parquet data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/val.parquet \
  --output data/audits/prediction_collapse.md \
  --limit 50 \
  --batch-size 16
```

Dataset diversity audit:

```bash
uv run python scripts/audit_dataset_diversity.py \
  --parquet data/training_workspace/sonna_personal_001_dataset/dataset.parquet \
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

Fixture-dependent RAW/XMP tests skip automatically when private local files in
`tests/fixtures/` are absent or unreadable. Restore those fixtures only when you
want to exercise live RAW extraction and real Lightroom XMP parsing locally.

## 14. Updating The App

Pull latest code:

```bash
git pull --ff-only
uv sync --extra dev
cd saha-app
npm install
cd ..
```

Restart the app after pulling code. If you are using the legacy two-terminal
flow, restart both backend and frontend. A live backend keeps old Python code
in memory.

## 15. Mac Notes

- Keep Lightroom closed for catalog reads.
- Never write to the `.lrcat`; the catalog reader opens it read-only.
- Never move or overwrite original RAW files.
- Do not overwrite checkpoints. The training scripts publish new versions.
- Use `v1_learning/` only for profiles that should appear in the frontend.
- Use `data/models/` for hidden runs, diagnostics, and unpromoted candidates.
- On Apple Silicon, the preferred device should be `mps`; CUDA is only expected
  on Windows/Linux NVIDIA machines.

