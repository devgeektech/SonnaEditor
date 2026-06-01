# Sonna Editor Project Knowledge

## Project overview

Sonna Editor is a local desktop tool for predicting Lightroom slider adjustments from RAW images. It uses a PyTorch regression model trained on Lightroom-edited photos, then writes XMP sidecars alongside RAW files. The UI is Electron + React and the backend is Python. The runtime target is cross-platform: macOS, Windows, and Linux.

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
- `data/`: storage for raw data, thumbnails, parquet datasets, models, and audit outputs
- `v1_learning/`: legacy model artifacts and training outputs

 ## Source Package Map (`src/sonna_editor`)

This section tracks what each backend source file/folder does. Keep it updated whenever files move, new entry points are added, or behavior changes.

### Root Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/__init__.py` | Package marker for the backend Python package. |
| `src/sonna_editor/config.py` | Central constants: paths, supported RAW extensions, model resolution, 147-slider field order, slider ranges, defaults, loss weights, confidence settings, and frontend-visible checkpoint directory `v1_learning/`. |
| `src/sonna_editor/runtime.py` | Runtime helpers for selecting CUDA, Apple MPS, or CPU and configuring data-loader pinned memory safely across platforms. |
| `src/sonna_editor/slider_set.py` | Slider-set version helpers for `v1`/`v2`, preventing checkpoint and tensor shape mismatches. |

### Data Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/data/__init__.py` | Package marker for data extraction/building modules. |
| `src/sonna_editor/data/audit.py` | Dataset quality audit utilities: unedited detection, outlier checks, high-variance checks, plots, and Markdown reports. |
| `src/sonna_editor/data/catalog.py` | Read-only Lightroom Classic `.lrcat` SQLite reader. Refuses lock/journal files, opens the catalog in read-only/query-only mode, finds edited photos, extracts develop settings, and can export XMP sidecars without overwriting existing files. |
| `src/sonna_editor/data/catalog_dataset.py` | Lightroom catalog dataset builder. This prepares supervised training rows from catalog develop settings, without requiring matching XMP sidecars. It still reads accessible RAW files for previews, metadata, histograms, and AsShot WB. |
| `src/sonna_editor/data/dataset.py` | RAW + XMP dataset builder. Finds RAW files with matching XMP sidecars, skips RAW-only files, extracts labels from XMP, writes deterministic Parquet rows, thumbnails, and shoot-grouped train/val/test splits. |
| `src/sonna_editor/data/dng.py` | Adobe DNG Converter wrapper for format-normalisation workflows. It should only read originals and write converted outputs, never mutate RAW files. |
| `src/sonna_editor/data/extract.py` | RAW preview, metadata, AsShot WB, and histogram extraction. `extract_all()` combines image input features with optional XMP labels for dataset building or inference preparation. |
| `src/sonna_editor/data/xmp.py` | Lightroom XMP read/write logic, slider parsing, tone curve handling, AsShot WB helper, Lightroom namespace/process-version handling, and XMP sidecar output for inference. |

### Model Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/model/__init__.py` | Package marker for model code. |
| `src/sonna_editor/model/architecture.py` | Main PyTorch model stack: `EmbeddingRegistry`, `MetadataEncoder`, `SonnaEditor`, ConvNeXt image backbone, metadata fusion, slider-group heads, WB metadata-skip residual behavior, and native checkpoint save/load. |
| `src/sonna_editor/model/augmentation.py` | Image-only training/validation augmentation. Target slider values are never augmented. |
| `src/sonna_editor/model/losses.py` | `WeightedSliderLoss`, range-normalized MSE, per-field weights, WB bucket losses, sign-wrong penalty, direction stats, and per-field MAE metrics. |
| `src/sonna_editor/model/postprocess.py` | Converts raw model outputs into Lightroom units, including log-Kelvin Temperature exponentiation, range clamping, and tensor-to-slider-dict mapping. |

### Training Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/training/__init__.py` | Package marker for training code. |
| `src/sonna_editor/training/callbacks.py` | Training alert callbacks for NaN loss, overfitting, disk space, ETA, loss balance, critical MAE, and overcorrection warnings. |
| `src/sonna_editor/training/datamodule.py` | Lightning data module and dataset wrapper. Builds embedding registries from parquet rows, loads thumbnails/metadata/histograms, emits image tensors, metadata tensors, targets, and sample weights. |
| `src/sonna_editor/training/module.py` | Lightning module around `SonnaEditor`: forward pass, train/val/test steps, optimizer/scheduler setup, loss logging, and MAE aggregation. |
| `src/sonna_editor/training/unfreeze_callback.py` | Backbone-unfreeze callback that resets early stopping after frozen-backbone warmup completes. |

### Inference Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/inference/__init__.py` | Package marker for inference code. |
| `src/sonna_editor/inference/engine.py` | Checkpoint loading and batched prediction engine. Builds tensors from extracted previews/metadata, maps categorical metadata through the checkpoint registry, supports uncertainty sampling, and postprocesses outputs. |
| `src/sonna_editor/inference/pipeline.py` | End-to-end shoot processing: scan RAW files, extract features, run inference, apply WB/skip semantics, write XMP sidecars, write `sonna_predictions.json`, and emit progress callbacks. |

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
| `src/sonna_editor/mode_b/checkpoint_builder.py` | Builds a Mode B initial checkpoint from a base checkpoint, Lightroom preset, and style survey by shifting output-head biases while preserving base model weights. This is not supervised photo training. |
| `src/sonna_editor/mode_b/survey.py` | Style survey models and conversion from user answers into slider offsets for exposure, temperature, tint, contrast, saturation, and shadows. |
| `src/sonna_editor/preset/__init__.py` | Package marker for preset code. |
| `src/sonna_editor/preset/adjuster.py` | Heuristic content-aware preset adjustments for exposure, WB, shadows/highlights, and similar safe corrections. |
| `src/sonna_editor/preset/parser.py` | Parses Lightroom `.xmp`, `.xmpsettings`, and `.lrtemplate` presets and validates extreme preset values. |
| `src/sonna_editor/preset/pipeline.py` | Legacy preset application pipeline that writes preset-derived XMP files for a RAW folder. It is useful for Mode B/preset output, not supervised model training. |

### API Package

| Path | Purpose |
|---|---|
| `src/sonna_editor/api/__init__.py` | Package marker for backend API code. |
| `src/sonna_editor/api/callbacks.py` | Bridges inference/training callbacks into job progress records and websocket events for the Electron UI. |
| `src/sonna_editor/api/confidence.py` | Reduces uncertainty samples/per-slider standard deviation into frontend confidence summaries. |
| `src/sonna_editor/api/jobs.py` | In-memory plus persisted job registry for long-running inference/fine-tune jobs, cancellation, recovery, and websocket subscribers. |
| `src/sonna_editor/api/models.py` | Pydantic request/response models for health, profiles, folders, processing, fine-tuning, Lite profile creation, and jobs. |
| `src/sonna_editor/api/server.py` | FastAPI app factory. Mounts routers, configures CORS, and recovers orphaned jobs at startup. |
| `src/sonna_editor/api/routes/__init__.py` | Package marker for route modules. |
| `src/sonna_editor/api/routes/captures.py` | API endpoints for capture/delta summaries used by continuous learning UI. |
| `src/sonna_editor/api/routes/finetune.py` | Starts asynchronous fine-tune jobs from captures and original training parquet, then writes new checkpoints into the frontend-visible model directory. |
| `src/sonna_editor/api/routes/folders.py` | Folder scan and recent-folder endpoints for UI file selection workflows. |
| `src/sonna_editor/api/routes/health.py` | Health endpoint exposing backend status, device information, git SHA, and model-loaded state. |
| `src/sonna_editor/api/routes/process.py` | Process-shoot job endpoints, job snapshots, cancellation, and websocket event streaming. |
| `src/sonna_editor/api/routes/profiles.py` | Profile management endpoints. Scans `v1_learning/model-v*.ckpt`, reads sidecar JSON files, activates/deletes profiles, and creates Mode B/Lite profiles. |

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
- uses `unknown` fallback IDs consistent with inference behavior

### `src/sonna_editor/api/`
- backend HTTP/WS API bridge for the Electron UI
- callbacks and job management for long-running inference and finetune jobs
- exposes routes for profile management, processing, and health

## UI and frontend

### `saha-app/`
- `src/App.jsx`: root React app
- `src/components/`: UI pages, editor, profiles, wizard, and job views
- `src/hooks/`: React hooks for jobs, profiles, captures, and recent folders
- `electron/`: Electron main/preload process wiring
- front-end interacts with Python backend via REST and websocket status updates

## Scripts

- `scripts/build_dataset.py`: generate training dataset from RAW + XMP
- `scripts/build_dataset_from_catalog.py`: generate training dataset from Lightroom catalog develop settings
- `scripts/run_style_survey.py`: create the Mode B/Lite survey JSON used with preset-based checkpoint creation
- `scripts/build_mode_b_checkpoint.py`: create a Mode B/Lite checkpoint from preset + survey + base checkpoint; if `--output` is omitted it publishes the next `v1_learning/model-v0.N.0.ckpt` for frontend visibility
- `scripts/process_shoot_preset.py`: direct preset-to-XMP execution with heuristic per-photo corrections, without creating a model checkpoint
- `scripts/train_v1_2_0_full_production.py`: train the main v1 model
- `scripts/train_profile.py`: current supported training entry point for new profiles
- `scripts/finetune_profile.py`: fine-tune a profile checkpoint
- `scripts/process_shoot_model.py`: inference wrapper for processing a shoot
- `scripts/audit_catalog.py`: catalog consistency audits
- `scripts/verify_environment.py`: environment checks

## Preset / Mode B Flow

Preset-based execution has two paths:

- **Mode B/Lite checkpoint path:** `run_style_survey.py` writes survey JSON, `build_mode_b_checkpoint.py` combines that survey with a Lightroom `.xmp` preset and a trained base checkpoint, then `process_shoot_model.py` runs the generated checkpoint through the normal inference pipeline. This is the preferred path when the preset should become a selectable frontend profile.
- **Direct preset path:** `process_shoot_preset.py` parses a preset, applies heuristic per-photo corrections, and writes XMP files directly. This is fast but does not create a model checkpoint and is not trainable by itself.

Mode B checkpoints are marked with `profile_type: mode_b_initial` in the sidecar JSON and are discovered by `/api/profiles` when written under `v1_learning/model-v*.ckpt`.

## Current fix and investigation notes

 - Adjusted `src/sonna_editor/inference/pipeline.py::process_shoot_with_model()` so the per-photo
   `on_photo_complete` callback is fired immediately after predictions are available (before the
   XMP write). This enables the UI to surface progress during extraction/inference rather than
   waiting for the sidecar file write to complete.

## Important behavior notes

- `Temperature` is predicted in log-Kelvin space by the model and is exponentiated in postprocessing.
- XMP write semantics intentionally distinguish generic skip fields from WB skip fields.
- Legacy v1 checkpoint support is preserved via checkpoint sidecar heuristics and output count gating.
- Raw metadata extraction uses embedded JPEG EXIF first, then supplements from a `.xmp` sidecar if present.

## Recommended next checks

- confirm UI progress callbacks are invoked in the early extraction and inference phases
- verify `process_shoot_with_model()` passes full model predictions into `sonna_predictions.json`
- audit any Mode B patch paths for the same metadata registry mapping behavior
