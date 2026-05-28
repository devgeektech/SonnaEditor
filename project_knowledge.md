# Sonna Editor Project Knowledge

## Project overview

Sonna Editor is a local desktop tool for predicting Lightroom slider adjustments from RAW images. It uses a PyTorch regression model trained on Lightroom-edited photos, then writes XMP sidecars alongside RAW files. The UI is Electron + React and the backend is Python.

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

## Key Python modules

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
- `warmup()`: compile MPS kernels with a dummy forward pass
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
- `scripts/train_v1_2_0_full_production.py`: train the main v1 model
- `scripts/finetune_profile.py`: fine-tune a profile checkpoint
- `scripts/process_shoot_model.py`: inference wrapper for processing a shoot
- `scripts/audit_catalog.py`: catalog consistency audits
- `scripts/verify_environment.py`: environment checks

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
