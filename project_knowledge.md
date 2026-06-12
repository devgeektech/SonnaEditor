# Sonna Editor Project Knowledge

## Project overview

Sonna Editor is a local desktop tool for predicting Lightroom slider adjustments from RAW images. It uses a PyTorch regression model trained on Lightroom-edited photos, then writes XMP sidecars alongside RAW files. The UI is Electron + React and the backend is Python. The runtime target is cross-platform: macOS, Windows, and Linux.

Current local workspace state as of 2026-06-12:
- Windows path: `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`
- Python 3.11.15 via uv 0.11.17. `pyproject.toml` / `uv.lock` require Python `3.11.*`.
- PyTorch `2.11.0+cu128`, CUDA active on NVIDIA GeForce RTX 3050. `torch==2.11.0` / `torchvision==0.26.0` are exact-pinned in `pyproject.toml`; Windows/Linux x86_64 resolve CUDA 12.8 local wheels through the configured PyTorch index, while macOS resolves public wheels.
- Training/profile caches were intentionally cleared for a fresh dataset reset.
- There is no guaranteed local `data/training_workspace/sonna_personal_001_dataset/` split set or frontend-visible `v1_learning/model-v*.ckpt` profile until fresh RAW+XMP data is added and a Personal AI profile is trained or a checkpoint is intentionally published. The duplicate generated folder `v1_learning/dataset` was removed. Generated datasets, splits, thumbnails, audits, and run workspaces belong under `data/training_workspace`; `v1_learning` is reserved for frontend-visible checkpoint/sidecar/preset/survey files.
- Historical diagnostics from the previous 189-photo local dataset remain useful for collapse analysis context, but do not assume those local Parquet/checkpoint files are present in this checkout.
- Supported RAW extension scanning is centralised in `config.SUPPORTED_RAW_EXTENSIONS` and currently covers `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and `.srw` across dataset building, folder/API scans, preset processing, model inference, and fine-tune capture. Decode/conversion still depends on `rawpy`/LibRaw or Adobe DNG Converter supporting the specific camera file.
- Preferred app startup is now one command: `.\run_saha.cmd` on Windows and
  `bash run_saha.sh` on macOS/Linux. The legacy two-terminal backend/frontend
  commands are still documented as a debugging reference.
- Latest full local verification on 2026-06-12 passed: environment `11/11`,
  `uv run ruff check .`, `uv run python -m compileall -q src scripts tests`,
  `npm run build:vite`, and `uv run pytest -q` (`753 passed, 45 skipped,
  1 warning`). The remaining warning is the known PyTorch scalar-conversion
  warning in `tests/test_losses.py`.

The core inference flow is:
- extract RAW preview + metadata
- build image + metadata batches
- predict slider values with a ConvNeXt-based model
- postprocess raw model outputs to Lightroom units
- write XMP sidecars with selective skip rules

## Top-level workspace structure

- `src/sonna_editor/`: Python package for the backend application
- `tests/`: pytest coverage for Python code
- `scripts/`: command-line scripts for training, dataset building, auditing, and inference
- `saha-app/`: Electron + React frontend UI
- `scripts/run_app.py`: one-command development launcher. It bootstraps runtime
  folders, checks for Node/npm, installs frontend npm dependencies when
  `saha-app/node_modules` is missing, then runs the Electron dev app. Electron
  starts or reuses the FastAPI backend on port 8765.
- `run_saha.cmd` / `run_saha.ps1` / `run_saha.sh`: root-level client-friendly
  wrappers around `uv run python scripts/run_app.py`; use `bash run_saha.sh`
  on macOS/Linux so the wrapper does not depend on executable file mode.
- `data/`: gitignored local learning area. `data/training_sources/` stores source RAW/XMP inputs in separate child folders per dataset or run; `data/training_workspace/` stores generated datasets and training runs, including catalog-derived FiveK splits. Fresh clones auto-create this tree at runtime.
- `v1_learning/`: frontend-visible published profile checkpoints plus sidecar/preset/survey files only. Generated datasets do not belong here.
- `.saha/`: repo-local runtime state for active-profile selection, recent folders, job snapshots, Personal AI training scratch runs, and fine-tune scratch runs. Auto-created at runtime and gitignored.
- `MAC_SETUP.md`: Mac-specific setup and run guide covering clean install, backend/frontend startup, frontend-capable workflows, and CLI equivalents.
- `SonnaEditorFoundation/`: repo-local hidden foundation-model folder by default, or `SONNA_FOUNDATION_REPO` if overridden. It contains schema-v2 `foundation_manifest.json` with `active_version` / `versions[]` lineage metadata and versioned `checkpoints/foundation-vN.ckpt` files. The active checkpoint is cumulative across real Lightroom-parameter sources: catalog and RAW+XMP runs update the same native `SonnaEditor` slider-regression checkpoint. Keep this out of gitignored `data/` but inside the SonnaEditor project root so the workspace stays self-contained. Checkpoint binaries are Git LFS-managed through `.gitattributes`; normal `git push` uploads them after `git lfs install`.

 ## Source Package Map (`src/sonna_editor`)

This section tracks what each backend source file/folder does. Keep it updated whenever files move, new entry points are added, or behavior changes.

### Root Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/__init__.py` | Package marker for the backend Python package. |
| `src/sonna_editor/config.py` | Central constants: repo-root/runtime paths, auto-created working-directory helpers, supported RAW extensions, model resolution, six scene-stat metadata field names, 147-slider field order, slider ranges, defaults, loss weights, confidence settings, and frontend-visible checkpoint directory `v1_learning/`. `SUPPORTED_RAW_EXTENSIONS` is the single scanned-format source of truth for training, inference, preset processing, folder APIs, and capture. |
| `src/sonna_editor/foundation.py` | Foundation checkpoint discovery, foundation folder layout creation, schema-v2 manifest writing, version listing, rollback helpers, provenance metadata, and promotion of trained checkpoints into the repo-local hidden foundation folder. |
| `src/sonna_editor/runtime.py` | Runtime helpers for selecting CUDA, Apple MPS, or CPU and configuring data-loader pinned memory safely across platforms. |
| `src/sonna_editor/slider_set.py` | Slider-set version helpers for `v1`/`v2`, preventing checkpoint and tensor shape mismatches. |

### Data Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/data/__init__.py` | Package marker for data extraction/building modules. |
| `src/sonna_editor/data/audit.py` | Dataset quality audit utilities: unedited detection, outlier checks, high-variance checks, plots, and Markdown reports. |
| `src/sonna_editor/data/catalog.py` | Read-only Lightroom Classic `.lrcat` SQLite reader. Refuses lock/journal files, opens the catalog in read-only/query-only mode, finds edited photos, supports exact collection-name filtering for virtual-copy datasets such as FiveK Expert C, extracts develop settings, and can export XMP sidecars without overwriting existing files. |
| `src/sonna_editor/data/catalog_dataset.py` | Lightroom catalog dataset builder. This prepares supervised training rows from catalog develop settings, without requiring matching XMP sidecars. It still reads accessible RAW files for previews, metadata, histograms, AsShot WB, and scene luminance stats. It supports `collection_name` filtering and an optional disabled unedited-looking filter for sparse FiveK catalog blobs. |
| `src/sonna_editor/data/dataset.py` | RAW + XMP dataset builder. Finds RAW files with matching XMP sidecars, skips RAW-only files, extracts labels from XMP, writes deterministic Parquet rows, thumbnails, scene luminance stats, and shoot-grouped train/val/test splits. |
| `src/sonna_editor/data/dng.py` | Adobe DNG Converter wrapper for format-normalisation workflows. It should only read originals and write converted outputs, never mutate RAW files. |
| `src/sonna_editor/data/extract.py` | RAW preview, metadata, AsShot WB, RGB histogram extraction, and preview-derived scene luminance stats. `extract_all()` combines image input features with optional XMP labels for dataset building or inference preparation. |
| `src/sonna_editor/data/xmp.py` | Lightroom XMP read/write logic, slider parsing, tone curve handling, AsShot WB helper, Lightroom namespace/process-version handling, and XMP sidecar output for inference. |

### Model Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/model/__init__.py` | Package marker for model code. |
| `src/sonna_editor/model/architecture.py` | Main PyTorch model stack: `EmbeddingRegistry`, `MetadataEncoder`, `SonnaEditor`, ConvNeXt image backbone, metadata fusion, slider-group heads, WB metadata-skip residual behavior, scene-stat metadata path, fresh `arch_version=3` staged-head conditioning, configurable backbone freeze/unfreeze helpers including compact trainable-layer specs, and native checkpoint save/load. |
| `src/sonna_editor/model/augmentation.py` | Image-only training/validation augmentation. Target slider values are never augmented. |
| `src/sonna_editor/model/losses.py` | `WeightedSliderLoss`, range-normalized MSE, per-field weights, WB bucket losses, sign-wrong penalty, direction stats, and per-field MAE metrics. |
| `src/sonna_editor/model/postprocess.py` | Converts raw model outputs into Lightroom units, including log-Kelvin Temperature exponentiation, range clamping, and tensor-to-slider-dict mapping. |

### Training Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/training/__init__.py` | Package marker for training code. |
| `src/sonna_editor/training/callbacks.py` | Training alert callbacks for NaN loss, overfitting, disk space, ETA, loss balance, critical MAE, and overcorrection warnings. |
| `src/sonna_editor/training/datamodule.py` | Lightning data module and dataset wrapper. Builds embedding registries from parquet rows, loads thumbnails/metadata/histograms/scene stats, emits image tensors, metadata tensors, targets, and sample weights. Old parquets without scene-stat columns are approximated from RGB histograms. |
| `src/sonna_editor/training/diagnostics.py` | Startup diagnostics for training runs: parameter counts, trainable breakdowns, ConvNeXt stage/block freeze summary, dataloader sampler info, and effective optimizer-step estimation. |
| `src/sonna_editor/training/module.py` | Lightning module around `SonnaEditor`: forward pass, train/val/test steps, optimizer/scheduler setup, configurable/custom/progressive backbone freeze/unfreeze handling, loss logging, MAE aggregation, and validation distribution/std-ratio logging for key sliders. |
| `src/sonna_editor/training/unfreeze_callback.py` | Backbone-unfreeze callback that resets early stopping after frozen-backbone warmup completes. |

### Inference Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/inference/__init__.py` | Package marker for inference code. |
| `src/sonna_editor/inference/engine.py` | Checkpoint loading and batched prediction engine. Builds tensors from extracted previews/metadata plus scene stats, maps categorical metadata through the checkpoint registry, supports uncertainty sampling, and postprocesses outputs. |
| `src/sonna_editor/inference/pipeline.py` | End-to-end shoot processing: scan RAW files using the central `config.SUPPORTED_RAW_EXTENSIONS` set, extract features, run inference, apply WB/skip semantics, write XMP sidecars, write `sonna_predictions.json`, and emit progress callbacks. |

### Fine-Tune Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/finetune/__init__.py` | Package marker for continuous-learning code. |
| `src/sonna_editor/finetune/capture.py` | Captures user corrections by comparing Saha predictions with final user-edited XMP files and writes capture datasets. |
| `src/sonna_editor/finetune/delta.py` | Analyzes correction deltas, computes summary/correlation stats, and prepares combined fine-tune parquet rows with sample weights. |
| `src/sonna_editor/finetune/retrain.py` | Fine-tunes an existing checkpoint into a new versioned checkpoint, compares validation behavior, and avoids overwriting trained models. |

### Mode B And Preset Packages

| Path | Purpose |
|---|---|
| `src/sonna_editor/mode_b/__init__.py` | Package marker for Lite/Mode B profile creation. |
| `src/sonna_editor/mode_b/checkpoint_builder.py` | Builds a Lite initial checkpoint/sidecar package from the configured foundation checkpoint, Lightroom preset, and style survey. It keeps pretrained foundation feature layers for future fine-tuning and stores preset/survey provenance for the adaptive initial Lite processing path. This is not supervised photo training. |
| `src/sonna_editor/mode_b/survey.py` | Lite style survey models and conversion from six user answers into slider offsets for Exposure2012, Temperature, Tint, Contrast2012, Saturation, and Shadows2012. Initial Lite runtime applies only Exposure/WB dynamically, but all six answers are stored in the profile package. |
| `src/sonna_editor/preset/__init__.py` | Package marker for preset code. |
| `src/sonna_editor/preset/adjuster.py` | Heuristic content-aware preset adjustments for exposure, WB, shadows/highlights, and similar safe corrections. Auto exposure uses mean luminance with 85th/95th percentile upper-tone guards so dark suits/rooms do not force bright faces/signage into overexposure. |
| `src/sonna_editor/preset/parser.py` | Parses Lightroom `.xmp`, `.xmpsettings`, and `.lrtemplate` presets and validates extreme preset values. |
| `src/sonna_editor/preset/pipeline.py` | Legacy preset application pipeline that writes preset-derived XMP files for a RAW folder. It is useful for Mode B/preset output, not supervised model training. |

### API Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/api/__init__.py` | Package marker for backend API code. |
| `src/sonna_editor/api/callbacks.py` | Bridges inference/training callbacks into job progress records and websocket events for the Electron UI. |
| `src/sonna_editor/api/confidence.py` | Reduces uncertainty samples/per-slider standard deviation into frontend confidence summaries. |
| `src/sonna_editor/api/jobs.py` | In-memory plus persisted job registry for long-running inference/fine-tune jobs, cancellation, recovery, and websocket subscribers. |
| `src/sonna_editor/api/models.py` | Pydantic request/response models for health, profiles, folders, processing, fine-tuning, Personal AI profile creation, Lite profile creation, and jobs. |
| `src/sonna_editor/api/server.py` | FastAPI app factory. Mounts routers, configures CORS, and recovers orphaned jobs at startup. |
| `src/sonna_editor/api/routes/__init__.py` | Package marker for route modules. |
| `src/sonna_editor/api/routes/captures.py` | API endpoints for capture/delta summaries used by continuous learning UI. |
| `src/sonna_editor/api/routes/finetune.py` | Starts asynchronous fine-tune jobs from captures and original training parquet, then writes new checkpoints into the frontend-visible model directory. |
| `src/sonna_editor/api/routes/folders.py` | Folder scan and recent-folder endpoints for UI file selection workflows. |
| `src/sonna_editor/api/routes/health.py` | Health endpoint exposing backend status, device information, git SHA, and model-loaded state. |
| `src/sonna_editor/api/routes/process.py` | Process-shoot job endpoints, job snapshots, cancellation, and websocket event streaming. |
| `src/sonna_editor/api/routes/profiles.py` | Profile management endpoints. Scans `v1_learning/model-v*.ckpt`, reads sidecar JSON files, activates/deletes profiles, starts frontend Personal AI RAW+XMP training jobs warm-started from the configured hidden foundation checkpoint, and creates Lite profiles from the configured foundation checkpoint. |

### Profiles And UI Placeholders

| Path | Purpose |
|---|---|
| `src/sonna_editor/profiles/__init__.py` | Placeholder package for future profile-domain code; active profile operations currently live in API routes and checkpoint helpers. |
| `src/sonna_editor/ui/__init__.py` | Legacy/placeholder UI package. The active app UI is Electron + React under `saha-app/`. |
| `src/sonna_editor/ui/widgets/__init__.py` | Placeholder for old Python widget code. |
| `src/sonna_editor/ui/windows/__init__.py` | Placeholder for old Python window code. |
| `src/sonna_editor/ui/workers/__init__.py` | Placeholder for old Python UI worker code. |

## Key Python Behavior Notes

### `src/sonna_editor/config.py`
- central config values and slider field definitions
- slider set versions (`v1`, `v2`) and field ordering
- training metadata such as normalization constants and loss weights
- application defaults used by training, inference, and UI

### `src/sonna_editor/model/architecture.py`
- `EmbeddingRegistry`: maps categorical metadata strings to integer IDs for embeddings
- `MetadataEncoder`: encodes camera metadata, histogram, and AsShot WB into a feature vector
- `SonnaEditor`: main regression model
  - ConvNeXt backbone
  - fusion MLP with metadata embeddings
  - multiple output heads per slider group
  - slider set version gating (`v1` vs `v2`)
  - v2 WB metadata skip: predicts residuals on top of
    `[log(as_shot_temperature), as_shot_tint]`; legacy checkpoints keep it off
- checkpoint API
  - `save_checkpoint()`: persist model weights and registry
  - `from_checkpoint()`: restore native checkpoints with registry and architecture metadata
- metadata registry growth helpers for new camera/lens/profile/WB labels

### `src/sonna_editor/model/losses.py`
- `WeightedSliderLoss`: weighted regression loss across Lightroom sliders
- range-normalized MSE per field, with temperature and tint bucket losses
- masking invalid rows on NaN or absent metadata
- metrics helpers: `direction_stats()` and `per_field_mae()`

### `src/sonna_editor/model/postprocess.py`
- `postprocess_predictions()`: convert raw model outputs to Lightroom slider units
  - Temperature: exponentiate log-Kelvin output
  - clamp tone curve and other slider ranges
- `predictions_to_dict()`: map tensor predictions to `config.SLIDER_FIELDS` names

### `src/sonna_editor/data/extract.py`
- `extract_preview()`: read RAW preview JPEG or fallback to half-size rawpy decode
- `extract_metadata()`: parse EXIF + XMP metadata from RAW files
- `compute_histogram()`: compute normalized RGB histograms used by the model
- `extract_all()`: helper for building training rows combining preview, metadata, histogram, and labels

### `src/sonna_editor/data/xmp.py`
- `write_xmp()`: serialize Lightroom XMP sidecars from slider dicts
- `read_xmp()`: read Lightroom-compatible slider values back from XMP
- `compute_as_shot_wb()`: derive AsShot Temperature/Tint from RAW white balance matrices
- handles Lightroom 15.4 namespace and preserves pre-Saha snapshot behavior
- implements WB skip semantics and always-on postprocess toggles

### `src/sonna_editor/inference/engine.py`
- `InferenceEngine`: load a trained checkpoint and run batched inference
- `_load_from_checkpoint()`: support Lightning and native SonnaEditor checkpoints
- `warmup()`: run a dummy forward pass so CUDA/MPS/CPU backend setup cost is paid before timed inference
- `_build_batch()`: assemble image tensors and metadata tensors for model input
  - current fix: map string metadata values to checkpoint registry IDs instead of zeroing all categorical IDs
  - fallback unknown values to `unknown` embedding index 0
- `predict()`: run model forward pass and postprocess outputs
- `predict_with_uncertainty()`: MC dropout sampling for uncertainty estimates
- `predict_one()`: convenience wrapper for single-image inference

### `src/sonna_editor/inference/pipeline.py`
- `process_shoot_with_model()`: end-to-end RAW folder inference pipeline
  - scan input dir for supported RAW extensions
  - extract previews and metadata in parallel
  - run `InferenceEngine.predict()` or uncertainty path
  - filter and substitute slider values for XMP write via skip rules
  - write XMP sidecars and optional `sonna_predictions.json`
  - fire `on_photo_complete` callbacks per photo
- defines v1 field skip behavior and WB substitution semantics
- ensures Temperature is epistemically clamped before writing

### `src/sonna_editor/training/datamodule.py`
- `build_registry()`: create an `EmbeddingRegistry` from training data categories
- `SonnaDataset`: dataset wrapper for parquet rows and metadata tensors
- maps training rows to model input tensors
- emits target tensors sized to the requested `slider_set_version`
- emits six scene-stat metadata values; old parquet rows without these columns get histogram-derived approximations
- uses `unknown` fallback IDs consistent with inference behavior

### `src/sonna_editor/training/profile_runner.py`
- packaged training runner used by both the CLI wrapper and the frontend Personal AI route
- supports `--base-model-checkpoint` for foundation warm-starts without resuming optimizer/epoch state; warm-starts keep the new training dataset registry and skip categorical embedding-table copies
- supports configurable backbone unfreeze strategies (`partial`, `full`, `progressive`)
- owns the production training callable, published checkpoint/sidecar creation, optional epoch callbacks for job progress, and cancellation handling before test/save/publish

### `scripts/train_profile.py`
- thin CLI wrapper for the packaged training runner and the current supported command for new Personal AI profiles
- default v2 recipe: 512px input, `slider_set_version="v2"`, fresh `arch_version=3`, batch size 16, lr 1e-4, freeze backbone for 3 epochs, WB metadata skip enabled
- default loss recipe: Exposure2012=5.0, Temperature=4.0, Tint=4.0, Contrast/Highlights/Shadows=3.0, Whites/Blacks/Saturation/Vibrance=2.0 minimums, temperature bucket=0.15, tint bucket=2.0, sign-wrong penalty=0.2
- fresh current-recipe training initialises output-head biases from training-set target medians
- default training augmentation is geometry-only; photometric brightness/contrast/saturation jitter is disabled by default because XMP labels are tied to the original image exposure and colour
- dataset splitting is still by shoot, but now balances Temperature correction, Exposure2012, and Tint correction instead of Temperature alone
- validation logs distribution/std-ratio metrics for key sliders so collapse is visible during training
- Lightning metric logging is guarded when step methods are called without a Trainer, so standalone unit tests do not emit spurious Lightning warnings.
- inference XMP writing stabilises RGB tone-curve endpoints: `ToneCurveRed/Green/Blue_Pt1` are forced to `0,0` and `Pt6` to `255,255` before writing. This prevents model-predicted channel-curve endpoint drift from turning neutral white highlights pink/red while preserving model-predicted mid-curve shape.
- logs default recipe changes as `Training recipe ...`; only user-supplied CLI flags log as `Override ...`
- adapts `log_every_n_steps` to the actual train-batch count so small local splits do not trigger Lightning's logging-interval warning
- saves a native `model.ckpt` from the best validation checkpoint and publishes a versioned copy into `v1_learning/` unless `--no-publish` is set
- use `scripts/train_foundation_model.py` for foundation runs that should be promoted into the repo-local hidden foundation folder

### `src/sonna_editor/api/`
- backend HTTP/WS API bridge for the Electron UI
- callbacks and job management for long-running inference and finetune jobs
- exposes routes for profile management, processing, and health
- `Profile.profile_type` is surfaced from checkpoint sidecar JSON: `None` for legacy trained profiles, `"mode_b_initial"` for Lite preset-derived profiles

## UI and frontend

### `saha-app/`
- `src/App.jsx`: root React app
- `src/components/`: UI pages, editor, profiles, wizard, and job views
- `src/hooks/`: React hooks for jobs, profiles, captures, and recent folders
- `electron/`: Electron main/preload process wiring
- front-end interacts with Python backend via REST and websocket status updates

## Scripts

- `scripts/build_dataset.py`: generate training dataset from RAW + XMP
- `scripts/build_dataset_from_catalog.py`: generate training dataset from Lightroom catalog develop settings. Supports `--collection-name` for exact Lightroom collection filtering and `--include-unedited-looking` for sparse FiveK expert catalog rows.
- `scripts/train_foundation_model.py`: current foundation-training command. It supports real Lightroom slider labels from RAW+XMP folders or trusted prepared splits, including catalog-derived FiveK Expert C splits. It avoids frontend publishing, warm-starts from the active foundation checkpoint by default, saves a new versioned native `SonnaEditor` checkpoint, and promotes that checkpoint into the configured foundation folder. Promotion auto-allocates `foundation-vN` when no version stem is supplied. Foundation defaults use batch size 8, auto-retry with smaller batches after CUDA memory failures, and adaptive capacity: catalog-scale/default runs start with `--backbone-trainable-layers stage:7`, while splits below 500 train rows automatically use `--backbone-unfreeze-strategy custom --backbone-trainable-layers none` unless explicit backbone flags are supplied. It passes repeatable `--field-loss-weight FIELD=WEIGHT` overrides through to the packaged trainer for focused tone/presence ablations.
- Foundation safety behavior: `scripts/train_foundation_model.py` refuses normal foundation training/promotion from fewer than 75 train rows unless `--allow-small-foundation-dataset` is passed, and refuses promotion when held-out quality metrics fail unless `--allow-quality-gate-failure` is passed. The quality gate also falls back to all-slider `test_per_field_mae` when compact Lightning test metrics omit a key slider. This prevents tiny RAW+XMP continuation sets from overwriting a broader active foundation without an explicit reviewed override.
- `scripts/rollback_foundation.py`: lists foundation manifest versions and rolls back the active foundation pointer to a previous version without deleting or overwriting checkpoint files.
- `scripts/analyse_backbone_drift.py`: compares ConvNeXt `backbone_features` tensors between a foundation checkpoint and a Personal AI checkpoint, reporting per-stage relative drift, cosine similarity, and the largest tensor changes.
- `scripts/run_style_survey.py`: create the Mode B/Lite survey JSON used with preset-based checkpoint creation
- `scripts/build_mode_b_checkpoint.py`: create a Lite checkpoint from preset + survey + configured foundation checkpoint; if `--output` is omitted it publishes the next `v1_learning/model-v0.N.0.ckpt` for frontend visibility
- `scripts/run_app.py`: starts Saha from the repo root with one command by
  preparing runtime folders, checking Node/npm, ensuring frontend dependencies
  exist, and running `npm run dev` under `saha-app/`. The Electron main process
  handles backend startup/shutdown.
- `scripts/process_shoot_preset.py`: direct preset-to-XMP execution with heuristic per-photo corrections, without creating a model checkpoint
- `scripts/train_v1_2_0_full_production.py`: train the main v1 model
- `scripts/train_profile.py`: current supported training entry point for new profiles
- `scripts/quick_diagnostic.py`: inspects training summary JSON files and key test metrics; if a summary omits train/val/test row counts, it reads counts from parquet split metadata, falling back to `data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/`. The metric table includes recommended scores and green-circle `OK` pass statuses, and its recommendations/next steps distinguish hidden foundation runs from frontend-published Personal AI profiles.
- `scripts/analyse_prediction_collapse.py`: runs a checkpoint on a validation parquet and reports per-slider prediction variance, target variance, MAE, and collapsed sliders. If the requested parquet path is missing, it now suggests nearby matching split files such as sibling `val.parquet` paths.
- `scripts/audit_dataset_diversity.py`: audits scene/edit diversity buckets from a parquet, using scene-stat columns or histogram-derived approximations
- `scripts/finetune_profile.py`: fine-tune a profile checkpoint
- `scripts/process_shoot_model.py`: inference wrapper for processing a shoot
- `scripts/audit_catalog.py`: catalog consistency audits
- `scripts/verify_environment.py`: environment checks

## Preset / Mode B Flow

Preset-based execution has two paths:

- **Lite checkpoint path:** `run_style_survey.py` writes six-question survey JSON, `build_mode_b_checkpoint.py` combines that survey with a Lightroom `.xmp` preset and the configured foundation checkpoint, then `process_shoot_model.py` runs the generated profile. Initial Lite processing detects `profile_type: mode_b_initial`, keeps preset look sliders fixed, and computes per-photo Exposure/WB corrections through `preset.adjuster` before writing XMPs. This is the preferred path when the preset should become a selectable frontend profile.
- **Direct preset path:** `process_shoot_preset.py` parses a preset, applies the same heuristic per-photo corrections, and writes XMP files directly. This is fast but does not create a model checkpoint and is not trainable by itself.

Lite checkpoints are marked with `profile_type: mode_b_initial` in the sidecar JSON and are discovered by `/api/profiles` when written under `v1_learning/model-v*.ckpt`.

## Current fix and investigation notes

- CUDA environment fix, 2026-06-01: `torch` and `torchvision` are pinned to the PyTorch CUDA 12.8 wheel index for Windows/Linux x86_64 in `pyproject.toml` and `uv.lock`. This resolved the CPU-only `torch 2.11.0+cpu` install that caused Lightning to report `GPU available: False` despite an RTX 3050 being present.
- Dependency pinning pass, 2026-06-10: direct runtime/dev dependencies in `pyproject.toml` are exact-pinned from the current uv environment, and the project now requires Python `3.11.*`. `uv.lock` was refreshed with the narrower Python requirement to reduce Mac setup conflicts from Python 3.12+ wheel resolution while preserving CUDA 12.8 wheels for Windows/Linux x86_64.
- Training log clarity fix, 2026-06-01: `scripts/train_profile.py` no longer labels default v2 recipe settings as overrides. It reports defaults as `Training recipe ...` and reserves `Override ...` for explicit CLI flags.
- Training runner packaging, 2026-06-02: the training callable now lives in `src/sonna_editor/training/profile_runner.py`; `scripts/train_profile.py` is a CLI wrapper. The API imports the packaged runner instead of importing a script module.
- Warm-start calibration, 2026-06-08: `src/sonna_editor/training/profile_runner.py` now recalibrates warm-started model output-head final biases from the current train split while preserving learned final weights. Fresh models still zero final output-head weights before applying target medians. This keeps foundation features but avoids stale output priors, such as FiveK Saturation/Vibrance baselines, dominating small Sonna continuation runs.
- Tone/presence focused loss overrides, verified 2026-06-12: `src/sonna_editor/training/profile_runner.py` accepts repeatable `--field-loss-weight FIELD=WEIGHT` overrides for any slider in `config.SLIDER_FIELDS`, records the parsed map in `training_summary.json`, and `scripts/train_foundation_model.py` passes those overrides through. The foundation CLI also exposes and forwards `--tone-presence-retry`, a reviewed shortcut that raises retry weights for Exposure2012, Whites2012, Blacks2012, Highlights2012, Shadows2012, Vibrance, and Saturation while respecting explicit per-field overrides. Use this before collecting more data when a foundation run is close on WB but misses tone/presence gate metrics.
- Profile runner Pylance cleanup, 2026-06-12: `src/sonna_editor/training/profile_runner.py` now narrows datamodule datasets/registry after `setup()`, types checkpoint path args explicitly, and reads the named `ModelCheckpoint` callback's best path/score through helper functions. This keeps VS Code/Pylance clean without changing training behavior.
- Foundation quality-gate diagnostics, 2026-06-10: `scripts/train_foundation_model.py` now writes `quality_gate_passed` and `foundation_quality_failures` back into `training_summary.json` before returning a promotion failure. `src/sonna_editor/training/profile_runner.py` records `hparams.max_epochs` in summaries, and `scripts/quick_diagnostic.py` prints backbone capacity, field-loss overrides, plus a train-median baseline comparison for failed gate fields when train/test Parquet paths are present.
- Foundation visual checkpoint selection, 2026-06-10: foundation training now passes `checkpoint_monitor="val_visual_score"` into `train_profile()`. `SonnaLightningModule` logs `val_visual_score`, a lower-is-better visual composite over Exposure, WB, tone, presence, HSL average, and key collapse ratios. `train_profile()` still records best true val-loss separately while exporting the checkpoint selected by the configured monitor.
- Foundation quality-gate tiering, 2026-06-10: foundation promotion now distinguishes hard failures from warnings. Hard failures still block promotion unless explicitly overridden after review; moderate misses are persisted as `foundation_quality_warnings`, printed to stderr, and allowed to promote so useful checkpoints are not blocked by one noisy slider.
- Frontend Personal AI training, 2026-06-02 and path cleanup 2026-06-03: `POST /api/profiles/personal` validates an absolute RAW+XMP folder, builds a dataset under `.saha/profile_training_runs/<job_id>/dataset`, trains with the production recipe, publishes into `v1_learning/`, and streams epoch progress through the jobs websocket.
- RAW extension source-of-truth fix, 2026-06-09: `src/sonna_editor/inference/pipeline.py` now derives its `RAW_EXTENSIONS` compatibility alias from `config.SUPPORTED_RAW_EXTENSIONS`, and `.rwl` was added to the central set. This keeps `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and `.srw` scanning consistent across dataset building, API folder scans, preset processing, model inference, and fine-tune capture.
- Dataset timezone-aware capture fix, 2026-06-10: `src/sonna_editor/data/dataset.py::_derive_shoot_id()` normalizes offset-aware capture datetimes to naive UTC before computing 12-hour shoot buckets and strips `tzinfo` from effectively naive datetimes. This prevents Mac RAW+XMP dataset builds from failing on ISO timestamps with offsets, such as `2024-03-15T23:30:00+13:00`, with `can't subtract offset-naive and offset-aware datetimes`.
- Runtime layout auto-create, 2026-06-03 and corrected 2026-06-04: backend/server and key CLI entrypoints call `config.ensure_runtime_directories()`, so `data/training_sources`, `data/raw`, `data/raw/sonna_training`, `v1_learning`, repo-local `.saha` state, and the repo-local `SonnaEditorFoundation/` folder exist automatically on a fresh clone. Source learning photos should be kept in separate child folders under `data/training_sources/`; generated datasets and run artifacts stay under `data/training_workspace/`; promoted foundation checkpoints stay under `SonnaEditorFoundation/`.
- Anti-collapse diagnostics, 2026-06-01: `model-v2.0.0` collapse audit on the 27-row val split found 14 collapsed sliders and Exposure2012 std_ratio=0.115. A fresh `arch_version=2` scene-stats candidate at `data/models/sonna-v2-scene-stats-run01` lowered test MAE but worsened collapse to 29 sliders and near-zero Exposure spread, so it was rejected and not kept frontend-visible.
- Dark-image mismatch diagnosis, 2026-06-01: the Lightroom mismatch on `0H5A4599` was an Exposure2012 model-collapse issue, not an XMP writer issue. The reference/training XMP used `Exposure2012=+1.11`; the then-active `model-v2.0.0` wrote about `+0.105` while nearby tone/WB sliders and curves were close to the reference. Across the previous 189-row dataset, target Exposure std was ~0.454 but model output std was ~0.061, and the darkest luminance quartile needed ~`+0.695` on average while the model predicted only ~`+0.090`.
- Lite profile slider-set compatibility fix, 2026-06-01: `src/sonna_editor/mode_b/checkpoint_builder.py` no longer down-converts the base checkpoint when creating Mode B/Lite checkpoints. This fixes frontend Lite creation while preserving the foundation checkpoint's native field count.
- Mode B adaptive Lite output fix, 2026-06-02: the Lite builder no longer adds preset/survey deltas to the base model's output-head biases. Initial Mode B processing now bypasses `InferenceEngine`, applies the copied preset look baseline, computes per-photo Exposure/WB only, and records the adjusted output in `sonna_predictions.json`.
- Mode B root-cause fix, 2026-06-02: stale `model-v0.1.0` was built with inherited final-layer base weights plus preset bias shifts, matching the observed double-apply failure. A corrected `model-v0.2.0` Lite profile was published, stale `model-v0.1.0` artifacts were removed, and `preset.adjuster` now guards auto exposure with upper-tone percentiles after real-folder testing showed mean-only exposure could still over-lift shadow-heavy event frames.
- Training warning cleanup, 2026-06-01: current training suppresses the upstream Lightning `LeafSpec` deprecation and optional Torch Triton FLOP-counter warning, and adjusts `log_every_n_steps` for tiny datasets. A one-epoch smoke run on the local 132-row split with two workers completed without those three warnings.
- Quick diagnostic clarity fixes, 2026-06-02 and 2026-06-04: `scripts/quick_diagnostic.py` reports train/val/test row counts for older summaries that do not embed those fields by reading split parquet metadata. The latest output adds recommended-score targets beside each critical metric, marks passing metrics with green-circle `OK`, treats missing `published_model` as normal for foundation runs, and prints foundation-aware next steps instead of a generic "run training" checklist.
- Mac setup runbook, updated 2026-06-12: `MAC_SETUP.md` documents Apple Silicon setup from system tools through one-command startup, the legacy two-terminal startup reference, Personal AI, Lite profiles, processing, fine-tuning, diagnostics, and CLI equivalents for frontend-capable steps.
- Foundation/Lite decoupling, 2026-06-03 and paired-image path removal 2026-06-05: Lite profile creation now resolves the configured foundation checkpoint through `sonna_editor.foundation.resolve_foundation_checkpoint()` instead of using the active Personal AI profile. `scripts/train_foundation_model.py` is the canonical CLI for real-parameter foundation training from RAW+XMP or prepared catalog splits, then promotion into the repo-local hidden foundation folder. The old paired rendered-image foundation trainer and hybrid decoder checkpoint path were removed.
- Local FiveK folder review, 2026-06-05: the current extracted folder is `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset`. It contains 5,000 `.dng` inputs under `raw_photos\HQa*`, `raw_photos\fivek.lrcat`, Lightroom `.lrprev` preview files, text/license/category files, and helper/catalog-data folders. The Lightroom catalog has 60,000 `Adobe_images` rows over the same 5,000 DNG files, with 12 virtual-copy/recipe variants per DNG and expert collections A/B/C/D/E at 5,000 rows each. The catalog route is now the supported FiveK foundation path.
- FiveK catalog slider route, 2026-06-05: `scripts/build_dataset_from_catalog.py` now supports `--collection-name "C"` and `--include-unedited-looking`. Use these together for a FiveK Expert C slider-regression foundation experiment. FiveK develop blobs are sparse, so missing/default sliders would otherwise be mistaken for an unedited photo. Do not mix all 60,000 FiveK catalog rows in one plain unconditioned slider model; train one expert collection first, then audit collapse before using it as an active foundation.
- Catalog unedited filter fix, 2026-06-08: `src/sonna_editor/data/catalog_dataset.py` now keeps the `skip_unedited` boolean separate from the skipped-row counter. Ordinary Lightroom catalog builds again skip unedited-looking rows by default; FiveK keeps them only through the explicit `--include-unedited-looking` override.
- FiveK catalog verification, 2026-06-05: the real local path `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset` was checked again. `raw_photos` has 5,000 `.dng` files and a readable `fivek.lrcat`; collections A/B/C/D/E each return 5,000 rows. A 20-row Expert C smoke build into `data\training_workspace\fivek_catalog_verify_20260605` succeeded with 0 missing files and 0 parse errors. FiveK teaches the model through DNG preview/metadata inputs and catalog slider targets; absent catalog slider fields are masked by the loss, with output priors falling back to Lightroom defaults for fields with no labels.
- Foundation checkpoint versioning, 2026-06-03 and schema-v2 update 2026-06-04: promotion never overwrites old foundation checkpoints. Each run updates `foundation_manifest.json` so the new checkpoint becomes active, records `active_version`, `versions[]`, SHA256, capabilities, and training source tags, and `scripts/rollback_foundation.py` can switch the active version without deleting files. `resolve_foundation_checkpoint()` still falls back to the newest remaining checkpoint if the active manifest target has been removed.
- Git LFS checkpoint workflow, 2026-06-10: `.gitattributes` routes `SonnaEditorFoundation/checkpoints/*.ckpt`, `v1_learning/*.ckpt`, and `models/**/*.ckpt` through Git LFS. Operators run `git lfs install` once per machine, use normal `git add` / `git commit` / `git push` for checkpoints, and run `git lfs pull` on new machines to materialize real `.ckpt` files.
- Legacy foundation manifest compatibility, 2026-06-05: `list_foundation_versions()` now includes the active checkpoint from older manifests that only have `active_checkpoint` plus `history`, so `scripts\rollback_foundation.py --list` shows both historical and active entries before the next schema-v2 promotion rewrites the manifest.
- Foundation clean slate, 2026-06-05: previous local trained profile and foundation checkpoint artifacts were removed (`v1_learning\model-v0.1.0.*`, `SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-001.*`, `foundation-sonna-raw-xmp-003.*`, and old `data\training_workspace\foundation_runs\`). `foundation_manifest.json` is now an empty schema-v2 manifest. `resolve_foundation_checkpoint()` raises `FileNotFoundError` for that state, and `scripts\train_foundation_model.py` treats it as no warm-start so the first new FiveK catalog foundation run can train from scratch and then become the default base.
- Foundation warm-start retention, 2026-06-04 and simplification/capacity update 2026-06-05/08: frontend Personal AI uses the progressive backbone schedule by default. Foundation training uses adaptive capacity: larger splits start with the final ConvNeXt stage trainable (`stage:7`) plus feature fusion/heads, while splits below 500 train rows default to heads/fusion-only (`custom` + `none`) to reduce overfitting. Compact specs such as `block:7:2,stage:6`, `block:7:1-2,stage:6`, `stage:7`, and `from:6` are supported for ablations. Warm starts now come from native `SonnaEditor` slider-regression checkpoints only. Use `scripts/analyse_backbone_drift.py` after training to quantify whether foundation features were preserved.
- Foundation bad-run triage, 2026-06-05: `foundation-sonna-raw-xmp-001` was rejected as the active foundation after diagnostics showed the Sonna continuation used only 132 train rows with 16.2M trainable parameters, overfit, and collapsed `Highlights2012`/`Shadows2012`. `SonnaEditorFoundation\foundation_manifest.json` was rolled back to `foundation-fivek-catalog-expert-c-001`. Future training summaries include split row counts, parquet paths, train-batch count, and all-slider `test_per_field_mae`; `quick_diagnostic.py` prints an all-parameter check when available.
- Dataset-location cleanup, 2026-06-05: removed the duplicate generated `v1_learning/dataset` folder and moved remaining active defaults to the canonical `data/training_workspace/sonna_personal_001_dataset` layout. Updated `config.ORIGINAL_TRAIN_PARQUET`, `quick_diagnostic`, `finetune_profile`, legacy v1 audit/training helpers, and `migrate_labels_to_v2`; added config regression coverage so the default train parquet cannot drift back to `v1_learning/dataset`.
- RAW+XMP foundation runbook cleanup, 2026-06-03 and path correction 2026-06-04: `FOUNDATION_TRAINING.md` now documents the parameter-supervised foundation path with direct script commands only: export Lightroom XMP sidecars, place source files under `data/training_sources/`, build inspectable Parquet splits, audit the dataset, then train from `--splits-dir` or use the direct `--raw-xmp-dir` shortcut. `CLI_COMMANDS.md`, `RUN.md`, `MAC_SETUP.md`, `README.md`, `HANDOVER.md`, and `SESSION_STATE.md` are aligned with the repo-local `SonnaEditorFoundation/` folder and the current cleared-local-cache state.
- Dataset audit dependency fix, 2026-06-03: `matplotlib` is now a base dependency so `scripts/audit_catalog.py` can generate ISO and slider-distribution PNGs without optional-import warnings. The audit CLI prints ASCII status labels to avoid Windows PowerShell emoji encoding errors.
- Foundation CUDA OOM fix, 2026-06-03: `scripts/train_foundation_model.py` defaults to batch size 8 and wraps RAW+XMP slider-regression training in CUDA-memory retry logic. On `CUDA out of memory` or `CUDNN_STATUS_EXECUTION_FAILED_CUDART`, it clears the CUDA cache and retries with halved batch sizes. A one-epoch RTX 3050 smoke passed at batch 8.
- Lite exposure lift, 2026-06-03: `preset.adjuster._exposure_delta()` now gives low-light scenes a stronger positive exposure floor when mean/median luminance are dark and upper percentiles are not near clipping. This targets Imagen-like behavior where dark event frames are lifted more decisively, while already-bright/overexposed frames still receive negative exposure.
- Lite survey contract correction, 2026-06-03: `src/sonna_editor/mode_b/survey.py`, `/api/profiles/lite`, and the Saha Lite wizard now capture all six Lite survey answers again. Initial Lite processing still dynamically adjusts only Exposure2012, Temperature, and Tint so the preset owns the initial look sliders.
- Staged-head learning improvement, 2026-06-03: fresh `arch_version=3` models condition WB/presence heads on the tone block output and condition later color/detail/curve heads on tone + presence + WB outputs. Existing checkpoints load with their saved architecture version.
- Earlier UI progress fix: `src/sonna_editor/inference/pipeline.py::process_shoot_with_model()` fires the per-photo `on_photo_complete` callback immediately after predictions are available, before the XMP write.

## Important behavior notes

- `Temperature` is predicted in log-Kelvin space by the model and is exponentiated in postprocessing.
- XMP write semantics intentionally distinguish generic skip fields from WB skip fields.
- Legacy v1 checkpoint support is preserved via checkpoint sidecar heuristics and output count gating.
- Lite profile creation from a v2 base must keep `slider_set_version="v2"`; down-converting via `from_checkpoint(target_slider_set_version="v1")` is intentionally rejected by the model loader.
- Initial Mode B/Lite checkpoints are intentionally profile carriers with preset/survey metadata and the configured foundation checkpoint's native slider set. Before fine-tuning, the UI/CLI processing path is Imagen-aligned Lite execution: uploaded preset controls the look, with per-photo Exposure/WB corrections only.
- Raw metadata extraction uses embedded JPEG EXIF first, then supplements from a `.xmp` sidecar if present.
- Fresh `arch_version=3` models consume preview-derived scene luminance statistics and use staged output-head conditioning. Existing `arch_version=1`/`2` checkpoints load unchanged and keep their saved head shapes.

## Recommended next checks

- Fixture-dependent RAW/XMP tests in `tests/test_extract.py` and `tests/test_xmp.py` now skip automatically when private local files such as `tests/fixtures/sample.cr3`, `sample.xmp`, or `sample_edit.xmp` are absent or unreadable. Restore those fixtures only when you want live RAW extraction and real Lightroom XMP parsing coverage.
- Build a larger edited dataset before trying another full model improvement run; the 189-photo local dataset lets median-prior models look good on MAE while failing prediction-spread/collapse checks.
- Run `scripts/analyse_prediction_collapse.py` after every candidate training run before publishing or activating it.
- Run `scripts/analyse_backbone_drift.py` for every foundation warm-start ablation to measure feature retention from the selected foundation checkpoint into the final Personal AI checkpoint.

