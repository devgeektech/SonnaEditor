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

## Quick Start

```powershell
python -m uv sync --extra dev
python -m uv run python scripts/verify_environment.py
```

Run the backend:

```powershell
python -m uv run python scripts/serve.py --port 8765
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
