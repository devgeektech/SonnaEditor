# Foundation Model Training

This is the foundation-only runbook. The foundation checkpoint is a hidden base
model used by Personal AI and Lite profile creation. It is not a frontend-visible
profile and must not be published into `v1_learning/`.

## Current Boundary

Foundation training is now **Lightroom-parameter supervised only**:

- RAW/DNG inputs are model inputs.
- Lightroom develop settings are the target labels.
- Targets can come from exported `.xmp` sidecars or from a read-only Lightroom
  catalog build.
- Paired rendered-image foundation training is not supported.

`scripts\train_foundation_model.py` trains the normal `SonnaEditor`
slider-regression checkpoint and promotes it into `SonnaEditorFoundation\`.
Every run creates a new checkpoint. Old checkpoints are never overwritten.

## Versioning And Cumulative Training

Every promoted foundation checkpoint is written under:

```text
SonnaEditorFoundation\checkpoints\<version>.ckpt
```

If `--version-stem` is omitted, the promoter creates the next
`foundation-vN.ckpt`. If `--version-stem` is supplied, it must be new.

By default, each new foundation run warm-starts from the currently active
foundation checkpoint:

```text
new checkpoint = previous active foundation + new dataset training
```

This is how knowledge carries forward:

```text
foundation-fivek-catalog-expert-c-001
  -> foundation-sonna-raw-xmp-001
  -> foundation-fivek-catalog-expert-c-002
  -> foundation-sonna-raw-xmp-002
```

Use `--no-warm-start` only for a deliberate scratch run. If a bad checkpoint is
promoted, roll back the active manifest pointer instead of deleting files:

```powershell
uv run python scripts\rollback_foundation.py --list
uv run python scripts\rollback_foundation.py foundation-v3
```

## Copy-Paste Naming Rule

For every new foundation command, change both identifiers together:

```text
--run-name
--version-stem
```

Keep them identical unless there is a deliberate reason not to.

Example:

```text
foundation-fivek-catalog-expert-c-001
foundation-fivek-catalog-expert-c-002
```

Do not reuse an old `--version-stem`. The command will fail at promotion time
because checkpoint overwrites are blocked. If you are unsure, omit
`--version-stem` and let the system allocate `foundation-vN`.

## FiveK Catalog Foundation Path

The inspected FiveK folder is:

```text
C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset
```

The usable catalog path is:

```text
C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset\raw_photos\fivek.lrcat
```

Catalog review found:

```text
Adobe_images:                         60,000 rows
Unique source files / AgLibraryFile:   5,000 DNGs
Adobe_imageDevelopSettings:           96,458 rows
Active non-empty develop settings:    60,000 rows
Each DNG has:                         12 catalog image rows / virtual copies
Expert collections A/B/C/D/E:          5,000 rows each
```

Verification on 2026-06-05:

```text
raw_photos file count:                 5,000 .dng files
Catalog file:                          raw_photos\fivek.lrcat, about 655 MB
Blocking lock files:                   none found (.lrcat-wal was 0 bytes; .lrcat-shm is harmless)
Expert collection query:               A/B/C/D/E each returned 5,000 rows
Collection C smoke build:              20 rows built successfully, 0 missing files, 0 parse errors
Smoke output:                          data\training_workspace\fivek_catalog_verify_20260605\
```

Use one expert collection first, normally `C`, so the plain slider-regression
model sees one target recipe per DNG. Do not mix A/B/C/D/E in one unconditioned
model unless expert/style conditioning is added.

Close Lightroom Classic before running catalog commands. The catalog is opened
read-only; RAW files are only read for previews, metadata, histograms, and
AsShot white-balance input features.

What the FiveK catalog teaches:

```text
DNG preview + RAW metadata + scene stats -> Lightroom develop settings
```

This is not rendered-image training. The model learns slider regression from
real catalog develop settings while seeing the DNG preview as the image input.
FiveK catalog blobs are sparse: absent slider values are stored as missing
targets and masked out of the loss. Fresh/foundation output priors still fall
back to Lightroom defaults for fields with no labels, so missing catalog fields
do not become random heads. If later quality audits show FiveK should treat
absent catalog sliders as explicit defaults, add that as a separate reviewed
data-policy change before retraining production foundations.

### Build Expert C Splits

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset\raw_photos\fivek.lrcat" `
  --output-dir "data\training_workspace\fivek_expert_c_catalog_dataset" `
  --profile-name "fivek_expert_c_catalog" `
  --collection-name "C" `
  --include-unedited-looking `
  --limit 5000 `
  --workers 8 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

`--include-unedited-looking` is intentional for FiveK. Its catalog develop blobs
are sparse, and many default sliders are absent. Do not use this flag for
ordinary Sonna catalogs unless the dataset has been audited.

### Audit Expert C Splits

```powershell
uv run python scripts\audit_catalog.py `
  --parquet-path "data\training_workspace\fivek_expert_c_catalog_dataset\dataset.parquet" `
  --output-dir "data\training_workspace\fivek_expert_c_catalog_dataset\audit"
```

```powershell
uv run python scripts\audit_dataset_diversity.py `
  --parquet "data\training_workspace\fivek_expert_c_catalog_dataset\dataset.parquet" `
  --output "data\training_workspace\fivek_expert_c_catalog_dataset\dataset_diversity.md"
```

### Train FiveK Catalog Foundation

Only run this after the full 5,000-row split build and audits pass. Do not train
from the 20-row smoke dataset.

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "data\training_workspace\fivek_expert_c_catalog_dataset\splits_v2_stratified" `
  --workspace-dir "data\training_workspace" `
  --foundation-repo "SonnaEditorFoundation" `
  --profile-name "Sonna FiveK Catalog Foundation Expert C" `
  --run-name "foundation-fivek-catalog-expert-c-001" `
  --version-stem "foundation-fivek-catalog-expert-c-001" `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

Expected outputs:

```text
data\training_workspace\foundation_runs\foundation-fivek-catalog-expert-c-001\
data\training_workspace\foundation_runs\foundation-fivek-catalog-expert-c-001\training\model.ckpt
SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.ckpt
SonnaEditorFoundation\foundation_manifest.json
```

### Audit The Trained Checkpoint

```powershell
uv run python scripts\analyse_prediction_collapse.py `
  --model-path "SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.ckpt" `
  --parquet "data\training_workspace\fivek_expert_c_catalog_dataset\splits_v2_stratified\val.parquet" `
  --output "data\audits\foundation-fivek-catalog-expert-c-001-collapse.md" `
  --limit 200 `
  --batch-size 16
```

## RAW+XMP Foundation Path

Use this for Sonna-owned foundation material where edited RAW/DNG files have
matching Lightroom `.xmp` sidecars.

### Prepare Sidecars

In Lightroom Classic:

1. Select the edited training photos.
2. Run `Metadata -> Save Metadata to File`.
3. Confirm each RAW/DNG has a same-stem `.xmp` sidecar next to it.

Recommended source folder:

```text
data\training_sources\sonna_foundation_001\raw_xmp\
```

### Build RAW+XMP Splits

Use this route for serious RAW+XMP foundation training because the generated
dataset and splits can be audited before training.

```powershell
uv run python scripts\build_dataset.py `
  --input-dir "data\training_sources\sonna_foundation_001\raw_xmp" `
  --output-dir "data\training_workspace\sonna_foundation_001_dataset" `
  --profile-name "sonna_foundation_001" `
  --workers 8 `
  --split `
  --val-ratio 0.107 `
  --test-ratio 0.139 `
  --splits-dir-name splits_v2_stratified
```

### Train From RAW+XMP Splits

Do not pass `--no-warm-start` if this run should inherit the active FiveK
foundation checkpoint. This is the intended continuation path:

```text
FiveK catalog foundation -> Sonna RAW+XMP foundation -> later FiveK/Sonna runs
```

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified" `
  --workspace-dir "data\training_workspace" `
  --foundation-repo "SonnaEditorFoundation" `
  --profile-name "Sonna RAW XMP Foundation" `
  --run-name "foundation-sonna-raw-xmp-001" `
  --version-stem "foundation-sonna-raw-xmp-001" `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

### Direct RAW+XMP Shortcut

This builds the dataset inside the run folder, then trains and promotes:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-xmp-dir "data\training_sources\sonna_foundation_001\raw_xmp" `
  --workspace-dir "data\training_workspace" `
  --foundation-repo "SonnaEditorFoundation" `
  --profile-name "Sonna RAW XMP Foundation" `
  --run-name "foundation-sonna-raw-xmp-001" `
  --version-stem "foundation-sonna-raw-xmp-001" `
  --max-epochs 100 `
  --batch-size 8 `
  --workers 8
```

## Resume Interrupted Foundation Training

If a run stops before promotion, resume the underlying profile trainer from the
last Lightning checkpoint, then promote manually after it finishes.

```powershell
uv run python scripts\train_profile.py `
  --train-parquet "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified\train.parquet" `
  --val-parquet "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified\val.parquet" `
  --test-parquet "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified\test.parquet" `
  --output-dir "data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-001\training" `
  --resume-from-checkpoint "data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-001\training\checkpoints\last.ckpt" `
  --max-epochs 100 `
  --batch-size 8 `
  --num-workers 8 `
  --no-publish `
  --profile-name "Sonna RAW XMP Foundation"
```

Promote the resumed final checkpoint with a new version stem:

```powershell
uv run python -c "from pathlib import Path; from sonna_editor.foundation import promote_foundation_checkpoint; promote_foundation_checkpoint(source_ckpt=Path(r'data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-001\training\model.ckpt'), display_name='Sonna RAW XMP Foundation', version_stem='foundation-sonna-raw-xmp-001-resumed', source_run_dir=Path(r'data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-001'))"
```

## Retrain Foundation Model

Retrain into a new run and new version stem:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified" `
  --workspace-dir "data\training_workspace" `
  --foundation-repo "SonnaEditorFoundation" `
  --profile-name "Sonna RAW XMP Foundation 002" `
  --run-name "foundation-sonna-raw-xmp-002" `
  --version-stem "foundation-sonna-raw-xmp-002" `
  --max-epochs 150 `
  --batch-size 8 `
  --workers 8
```

On CUDA machines with limited VRAM, start foundation runs at `--batch-size 8`.
The foundation CLI automatically retries with smaller batch sizes after CUDA
memory failures.

## Use Foundation For Profiles

- **Personal AI / Mode A:** frontend training resolves the active hidden
  foundation checkpoint, warm-starts from it, then publishes only the Personal
  AI profile into `v1_learning\`.
- **Lite / Mode B:** Lite profile creation resolves the active hidden foundation
  checkpoint and builds a `mode_b_initial` profile from that checkpoint plus the
  preset and six-question survey. Later fine-tuning uses the normal training
  path.
