# Sonna Editor

Internal AI photo editing tool for Sonna Studios. It predicts Lightroom slider
values from RAW previews and metadata, then writes Lightroom-compatible XMP
sidecars for new shoots.

## Current State

- Backend: Python 3.11, uv, FastAPI, PyTorch, pytest.
- Frontend: Electron + React in `saha-app/`.
- Production profile lineage: v1.2.x Mode A checkpoints under `v1_learning/`.
- Mode B preset/survey checkpoints use the same inference path as Mode A.
- Platform target: macOS, Windows, and Linux. CUDA and Apple MPS are used when
  available; CPU fallback is supported for development and small runs.
- Current Windows training workspace: PyTorch `2.11.0+cu128` is pinned through
  `pyproject.toml` / `uv.lock` and verified on an NVIDIA GeForce RTX 3050.
- The local dataset currently has 189 rows with shoot-grouped balanced splits:
  train=132, val=27, test=30.
- Current v2 training defaults use geometry-only augmentation, Exposure loss
  weight 4.0, and fresh output-head target-prior initialisation from the
  training split to reduce brightness/WB drift on small datasets.
- Mode A inference stabilises RGB tone-curve endpoints before XMP write so
  neutral white highlights do not shift pink/red from channel-curve endpoint
  drift.
- This checkout currently has `v1_learning/model-v2.0.0.ckpt` with matching
  sidecar JSON, so the frontend can discover one local v2 profile.

## Quick Start

```powershell
uv sync --extra dev
uv run python scripts/verify_environment.py
```

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

Preset + survey creates a Mode B initial checkpoint, but it is not supervised
training from photos. See `TRAINING_COMMANDS.md` for the full runbook.

## Preset / Mode B Profiles

Mode B/Lite starts from a trained Mode A checkpoint, a Lightroom preset, and a
six-question style survey. The checkpoint builder publishes `model-v0.N.0.ckpt`
under `v1_learning/` when no explicit `--output` is provided, so the frontend
can discover it like any trained profile.

Direct preset execution is also available through `scripts/process_shoot_preset.py`
when you only want preset-derived XMP files and do not need a selectable model
profile.
