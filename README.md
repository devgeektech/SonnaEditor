# Sonna Editor

Internal AI photo editing tool for Sonna Studios. It predicts Lightroom slider
values from RAW previews and metadata, then writes Lightroom-compatible XMP
sidecars for new shoots.

## Current State

- Backend: Python 3.11, uv, FastAPI, PyTorch, pytest.
- Frontend: Electron + React in `saha-app/`.
- Production profile lineage: frontend-visible profiles live under `v1_learning/`.
- Personal AI profiles train from RAW+XMP through the frontend and warm-start
  from the hidden foundation checkpoint. Lite profiles use the same hidden
  foundation checkpoint plus preset/survey style with adaptive per-photo
  Exposure/WB corrections before fine-tuning.
- Platform target: macOS, Windows, and Linux. CUDA and Apple MPS are used when
  available; CPU fallback is supported for development and small runs.
- Current Windows training workspace: PyTorch `2.11.0+cu128` is pinned through
  `pyproject.toml` / `uv.lock` and verified on an NVIDIA GeForce RTX 3050.
- Training/profile caches were cleared on 2026-06-03 so a fresh dataset can be
  added. There is currently no guaranteed local dataset or visible checkpoint.
- Current training defaults use geometry-only augmentation, Exposure loss
  weight 5.0, and output-head target-prior calibration from the training split.
  Fresh runs zero final head weights before applying priors; warm-started runs
  keep learned weights and recenter final biases to reduce stale foundation
  colour/exposure drift.
- Inference stabilises RGB tone-curve endpoints before XMP write so
  neutral white highlights do not shift pink/red from channel-curve endpoint
  drift.
- Foundation model training is CLI-only. `scripts/train_foundation_model.py`
  trains native `SonnaEditor` slider-regression checkpoints from real Lightroom
  labels: RAW+XMP sidecars or catalog-derived splits such as FiveK Expert C.
- Foundation runs are versioned and warm-start from the active foundation
  checkpoint by default. A successful run promotes the new checkpoint as active
  while keeping older checkpoints available for fallback.
- Foundation promotion is guarded: normal runs need at least 75 train rows and
  must pass held-out quality gates. Splits below 500 train rows automatically
  use heads/fusion-only foundation capacity unless explicit backbone ablation
  flags are supplied.

## Quick Start

```powershell
uv sync --extra dev
uv run python scripts/verify_environment.py
```

For Mac setup from a clean machine through frontend/CLI operation, see
`MAC_SETUP.md`.

Runtime working folders are now created automatically from the project root on
backend or CLI startup, including `data/training_sources/`, `data/raw/`,
`data/datasets/`, `v1_learning/`, and `.saha/`.
That keeps a fresh clone usable even though those directories are gitignored.
Generated foundation training runs stay under `data/training_workspace/` by
default. Promoted foundation checkpoints live under the project child folder
`SonnaEditorFoundation/` unless `SONNA_FOUNDATION_REPO` points somewhere else.
That folder is tracked by the parent repo; `.ckpt` checkpoint binaries are
routed through Git LFS.

On Windows/Linux x86_64, `uv sync --extra dev` installs CUDA-enabled PyTorch
from the pinned PyTorch CUDA 12.8 index. If no NVIDIA GPU is available, the app
still runs with CPU fallback, but training will be slow.

Run the backend:

```powershell
uv run python scripts/serve.py --port 8765
```

Run the Electron app from a second terminal:

```powershell
cd saha-app
npm install
npm run dev
```

On macOS/Linux the same commands work with `python3 -m uv ...` if `python`
does not point at Python 3.11+.

## Optional External Tools

Adobe DNG Converter is only required for workflows that normalise RAW files to
DNG. The app discovers it from:

1. `SONNA_DNG_CONVERTER`
2. Default Adobe install paths on macOS/Windows
3. `PATH`

If it is absent, tests and non-DNG workflows still run.

## Training Data Sources

The trained model is supervised slider regression, so RAW files alone are not
enough. Training requires target Lightroom slider values from one of:

- RAW files with matching exported `.xmp` sidecars.
- A Lightroom Classic `.lrcat` opened read-only, with accessible RAW files.
- Fine-tune captures from previous Saha runs.

Preset + survey creates a Lite profile, but it is not supervised training from
photos. See `CLI_COMMANDS.md` for operator commands and
`FOUNDATION_TRAINING.md` for hidden foundation-model training.

Local learning photos should be kept in separate child folders under
`data/training_sources/`, for example `sonna_personal_001/raw_xmp/` for
RAW+XMP profile data. FiveK foundation training is built from the Lightroom
catalog into `data/training_workspace/fivek_expert_c_catalog_dataset/`. The
whole `data/` tree is gitignored, so those photos stay local.

## Lite Profiles

Lite starts from the configured foundation checkpoint, a Lightroom preset, and
the Lite survey. It does not depend on the currently active Personal AI profile.
The current UI asks six survey questions. The first Lite processing pass uses
the survey and preset metadata while dynamically adjusting only Exposure,
Temperature, and Tint because the preset owns the look sliders. The checkpoint
builder publishes `model-v0.N.0.ckpt` under
`v1_learning/` when no explicit `--output` is provided, so the frontend can
discover it like any trained profile. Initial Lite processing detects
`profile_type: mode_b_initial`, applies the preset as the style baseline, then
computes per-photo Exposure, Temperature, and Tint corrections before writing
XMPs. Preset look sliders such as Contrast, Shadows, Highlights, Whites,
Blacks, Saturation, and Vibrance stay preset-fixed until later fine-tuning.

Lite profiles are visible in the UI when published into `v1_learning/`. If you
created a Lite profile before the 2026-06-02 Mode B fixes, rebuild it so the
sidecar and copied preset/survey metadata match the current flow.

Only profiles placed in `v1_learning/` are scanned by the UI. The foundation
checkpoint lives in the hidden foundation folder configured by
`SONNA_FOUNDATION_REPO` or the default repo-local `SonnaEditorFoundation/`
folder.

Direct preset execution is also available through `scripts/process_shoot_preset.py`
when you only want preset-derived XMP files and do not need a selectable model
profile. This preset-only path uses the same content-aware adjustment logic, but
does not create a profile that the UI can select or later fine-tune.
