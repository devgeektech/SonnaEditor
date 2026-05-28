# Run Sonna Editor

This is the cross-platform local runbook for macOS, Windows, and Linux.

## 1. Python Environment

The repo is uv-managed and pinned to Python 3.11 through `.python-version`.

Windows PowerShell:

```powershell
cd F:\Projects\SonnaEditor
python -m uv sync --extra dev
python -m uv run python --version
```

macOS/Linux shell:

```bash
cd /path/to/SonnaEditor
python3 -m uv sync --extra dev
python3 -m uv run python --version
```

If `uv` is not installed yet:

```bash
python -m pip install --user uv
```

Use `python3` instead of `python` on systems where that is the correct Python
launcher.

## 2. Verify

```bash
python -m uv run python scripts/verify_environment.py
```

The verifier checks Python, imports, and the best available PyTorch device:
CUDA, Apple MPS, or CPU. Adobe DNG Converter is reported as optional unless you
need RAW-to-DNG normalisation.

## 3. Backend API

```bash
python -m uv run python scripts/serve.py --port 8765
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

Use the stratified by-shoot splits and train a fresh v2 profile with the WB
metadata skip enabled (default):

```bash
python -m uv run python scripts/train_profile.py \
  --train-parquet data/splits/train.parquet \
  --val-parquet data/splits/val.parquet \
  --test-parquet data/splits/test.parquet \
  --output-dir data/models/sonna-v2-run01 \
  --slider-set-version v2 \
  --image-resolution 512 \
  --batch-size 16 \
  --max-epochs 100
```

Windows PowerShell uses the same command with backticks for line continuation:

```powershell
python -m uv run python scripts/train_profile.py `
  --train-parquet data\splits\train.parquet `
  --val-parquet data\splits\val.parquet `
  --test-parquet data\splits\test.parquet `
  --output-dir data\models\sonna-v2-run01 `
  --slider-set-version v2 `
  --image-resolution 512 `
  --batch-size 16 `
  --max-epochs 100
```

The script writes `model.ckpt`, `model.json`, TensorBoard logs, and
`training_summary.json` into the output directory. The exported `model.ckpt`
contains the best validation checkpoint, not just the final epoch.

Monitor training:

```bash
python -m uv run tensorboard --logdir data/models/sonna-v2-run01
```
