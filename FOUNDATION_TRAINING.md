# Foundation Model Training

This file is the foundation-only runbook. The foundation checkpoint is a hidden
base model used by Personal AI and Lite profile creation. It is not a
frontend-visible profile and should not be published into `v1_learning/`.

## Current Implementation Boundary

There are two foundation concepts in the project now:

- **Implemented today:** `scripts/train_foundation_model.py` trains the existing
  `SonnaEditor` slider-regression model from real Lightroom parameters
  (`RAW + XMP` or catalog-derived develop settings), then promotes that
  checkpoint into the hidden foundation repo.
- **Implemented TIFF direction:** MIT-Adobe FiveK can train an image-supervised
  enhancement backbone from `RAW/DNG -> expert TIFF`. This is a separate
  foundation-only path. Do not force FiveK TIFF targets through the current XMP
  slider-regression pipeline.

The current foundation checkpoint can still warm-start Personal AI and Lite
profiles. The long-term target is a stronger image-supervised foundation
backbone that teaches exposure, white balance, tone, and global colour before
profile-specific XMP prediction.

## Two-Stage Model Strategy

Stage 1 is the **Foundation Enhancement Model**:

- input: RAW/DNG image features
- target: professional edited image, initially a FiveK expert TIFF
- goal: learn general photographic correction, not a photographer's style
- primary concepts: white balance, exposure, contrast, highlights, shadows,
  tone mapping, and global colour correction
- likely losses: L1/MAE plus perceptual image losses such as SSIM and LPIPS
- saved assets: encoder/backbone/feature-extractor weights

Stage 2 is the **Profile-Specific Model**:

- input: RAW image features, metadata, and the foundation-initialised backbone
- target: Lightroom XMP slider values from Sonna/proprietary profile data
- goal: learn photographer, event, wedding, birthday, party, and other creative
  style preferences
- output: Lightroom-compatible slider predictions and XMP sidecars

Conceptually, the final edit should behave like:

```text
Final Edit = Foundation Edit + Profile Residual Adjustment
```

The foundation model should learn the common photographic correction layer. The
profile model should learn the creative residual: Sonna style, photographer
preference, preset behaviour, and event-specific colour grading.

## Foundation Visibility Rule

- Foundation checkpoints live in a hidden foundation repo, by default
  `data/foundation_repo/` inside this project, or a custom path from
  `SONNA_FOUNDATION_REPO`.
- The UI scans only `v1_learning/model-v*.ckpt`, so foundation checkpoints stay
  unseen as long as they remain in the foundation repo.
- Lite profiles and Personal AI profile creation resolve the foundation
  checkpoint from:
  1. `SONNA_FOUNDATION_CHECKPOINT`
  2. `SONNA_FOUNDATION_REPO/foundation_manifest.json`
  3. `SONNA_FOUNDATION_REPO/foundation.ckpt`

## Foundation Versioning Rule

Every foundation training run writes a new versioned checkpoint under:

```text
data/foundation_repo/checkpoints/<version-stem>.ckpt
```

The previous checkpoint is never overwritten. After a successful run,
`foundation_manifest.json` is updated so the new checkpoint becomes the active
default foundation model for Personal AI and Lite profile creation. The manifest
also keeps recent history entries so the previous active checkpoint is visible.

By default, each new foundation run **warm-starts from the currently active
foundation checkpoint**:

- RAW+XMP or catalog-split foundation runs warm-start the `SonnaEditor` model
  from the active foundation checkpoint, then train on the new Lightroom-label
  dataset.
- TIFF/image foundation runs copy the active checkpoint's compatible ConvNeXt
  backbone weights, then train the image-to-image foundation model on the new
  paired-image dataset.

This means a new run is effectively:

```text
new checkpoint = previous active foundation + new dataset training
```

The previous active checkpoint file stays untouched. If a bad new checkpoint is
promoted, remove that bad `.ckpt` file from `data/foundation_repo/checkpoints/`.
When the manifest points at a missing active checkpoint, the resolver falls back
to the newest remaining checkpoint in that folder. For a deliberate scratch
foundation run that should not warm-start from the active checkpoint, pass
`--no-warm-start`.

## FiveK Dataset Notes

MIT-Adobe FiveK is good foundation-learning material because it contains broad
scenes, lighting, RAW DNG inputs, and professional Lightroom-based retouches. It
is not 25,000 independent RAW images. It is 5,000 DNG photos with 5 expert
retouches, producing 25,000 edited TIFF renditions.

FiveK does **not** provide XMP files directly. It may include Lightroom catalog
metadata in `fivek.lrcat`, but reliable Lightroom slider extraction is a future
investigation, not the default training path.

Use FiveK for paired-image foundation learning:

```text
RAW/DNG -> selected expert TIFF
```

Do not generate fake Lightroom slider labels from FiveK TIFFs. Multiple
Lightroom slider combinations can create visually similar TIFF outputs, so
forcing TIFF targets into fake XMP labels would add noisy supervision to the
profile model.

For the first image-supervised foundation run, use one expert target, commonly
Expert C, so each source image has one target rendition. Later, an explicit
expert/style-conditioned setup can use all five experts without conflicting
labels.

FiveK is research-licensed. Keep the downloaded data in a gitignored training
source folder, or point the commands at an external drive if storage gets too
large. Cite the dataset when used.

## Training Source Layout

Keep source photos separate from generated datasets and model outputs. The app
auto-creates `data/training_sources/`, and each learning source should get its
own child folder:

```text
data/training_sources/
  fivek_expert_c/
    raw_dng/
    expert_tiff/
    MITAdobeFiveK/
  sonna_personal_001/
    raw_xmp/
  sonna_personal_002/
    raw_xmp/
```

These folders are for local learning inputs only. They are under gitignored
`data/`, so RAWs, TIFFs, XMPs, and downloaded datasets do not get committed.
Generated datasets, thumbnails, training summaries, and checkpoints stay under
`data/training_workspace/` and `data/foundation_repo/`.

Recommended project-local paths:

```powershell
$env:SONNA_TRAINING_WORKSPACE = "$PWD\data\training_workspace"
$env:SONNA_FOUNDATION_REPO = "$PWD\data\foundation_repo"
$env:FIVEK_ROOT = "$PWD\data\training_sources\fivek_expert_c\MITAdobeFiveK"
```

`data\training_sources\` is part of the auto-created repo-local layout. You
still need to place the actual FiveK files or Sonna RAW+XMP folders there
yourself.

## FiveK Image-To-Image Foundation Training

This is the recommended FiveK foundation direction. The current implementation
uses `scripts\train_foundation_model.py --raw-image-dir ... --target-tiff-dir ...`
and saves an image-foundation checkpoint whose ConvNeXt backbone can warm-start
Personal AI training or Lite profile carriers.

The trainer:

1. Read FiveK DNG inputs and one selected expert TIFF target per image.
2. Build train/val/test splits by source photo, not by rendition.
3. Train an image-supervised enhancement model with L1/MAE and SSIM loss.
   LPIPS is documented as a future optional enhancement and is not enabled in
   the current dependency set.
4. Save reusable ConvNeXt backbone weights in an `image_to_image_v1` checkpoint.
5. Use those weights to initialise the existing `SonnaEditor` XMP-regression
   profile model.

Do not use TIFF files as if they were XMP labels. Do not invent Lightroom
parameters unless running a clearly marked experiment whose results are kept out
of production profile training.

Example:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-image-dir "$PWD\data\training_sources\fivek_expert_c\raw_dng" `
  --target-tiff-dir "$PWD\data\training_sources\fivek_expert_c\expert_tiff" `
  --workspace-dir "$PWD\data\training_workspace" `
  --foundation-repo "$PWD\data\foundation_repo" `
  --profile-name "Sonna FiveK Image Foundation Expert C" `
  --run-name "foundation-fivek-image-expert-c-001" `
  --version-stem "foundation-fivek-image-expert-c-001" `
  --image-resolution 512 `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 8 `
  --l1-weight 1.0 `
  --ssim-weight 0.2 `
  --init-git
```

Expected outputs:

```text
<project>\data\training_workspace\foundation_runs\foundation-fivek-image-expert-c-001\
<project>\data\training_workspace\foundation_runs\foundation-fivek-image-expert-c-001\training\model.ckpt
<project>\data\foundation_repo\checkpoints\foundation-fivek-image-expert-c-001.ckpt
<project>\data\foundation_repo\foundation_manifest.json
```

The promoted checkpoint is not a full Lightroom slider-regression checkpoint.
It contains image-foundation backbone weights. Personal AI warm-start and Lite
profile creation know how to copy those backbone weights into a fresh
`SonnaEditor` model while keeping the RAW+XMP profile training contract intact.

## Build Dataset From RAW + XMP

Use this only when your data is already exported as edited RAW/DNG files with
matching Lightroom `.xmp` sidecars, such as Sonna internal profile data. This is
parameter-supervised training for the existing `SonnaEditor` model. It is not
the recommended FiveK TIFF workflow. This RAW+XMP path is still fully supported.
Unless `--no-warm-start` is supplied, it starts from the active foundation
checkpoint and writes a new versioned foundation checkpoint.

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-xmp-dir "$PWD\data\training_sources\sonna_personal_001\raw_xmp" `
  --workspace-dir "$PWD\data\training_workspace" `
  --foundation-repo "$PWD\data\foundation_repo" `
  --profile-name "Sonna Foundation" `
  --run-name "foundation-sonna-parameter-001" `
  --version-stem "foundation-sonna-parameter-001" `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 8 `
  --init-git
```

Expected outputs:

```text
<project>\data\training_workspace\foundation_runs\foundation-sonna-parameter-001\
<project>\data\training_workspace\foundation_runs\foundation-sonna-parameter-001\training\model.ckpt
<project>\data\foundation_repo\checkpoints\foundation-sonna-parameter-001.ckpt
<project>\data\foundation_repo\foundation_manifest.json
```

## Future FiveK Catalog Investigation

FiveK includes `fivek.lrcat`, which may contain edit metadata and slider
histories. Investigate this later as a separate read-only research task. If
Lightroom slider values can be recovered reliably, then a secondary dataset can
be created:

```text
RAW/DNG + derived Lightroom edit parameters
```

That dataset should be evaluated against the image-supervised foundation
approach before it affects production profile training. Until then, do not make
FiveK catalog extraction the primary foundation route.

For reference, the existing catalog dataset builder can read Lightroom catalogs
read-only when target slider values are known to be useful:

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "$PWD\data\training_sources\fivek_expert_c\MITAdobeFiveK\fivek.lrcat" `
  --output-dir "$PWD\data\training_workspace\fivek_expert_c_dataset" `
  --profile-name "fivek_expert_c" `
  --limit 5000 `
  --workers 8 `
  --split
```

If that experiment produces trusted parameter splits, train and promote a
separate experimental checkpoint from those prepared splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "$PWD\data\training_workspace\fivek_expert_c_dataset\splits_v2_stratified" `
  --workspace-dir "$PWD\data\training_workspace" `
  --foundation-repo "$PWD\data\foundation_repo" `
  --profile-name "Sonna Foundation FiveK Expert C" `
  --run-name "foundation-fivek-expert-c-001" `
  --version-stem "foundation-fivek-expert-c-001" `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 8 `
  --init-git
```

## Resume Interrupted Foundation Training

If a run stops before promotion, resume the underlying profile trainer from the
last native or Lightning checkpoint, then promote manually after it finishes.

```powershell
uv run python scripts\train_profile.py `
  --train-parquet "$PWD\data\training_workspace\sonna_parameter_dataset\splits_v2_stratified\train.parquet" `
  --val-parquet "$PWD\data\training_workspace\sonna_parameter_dataset\splits_v2_stratified\val.parquet" `
  --test-parquet "$PWD\data\training_workspace\sonna_parameter_dataset\splits_v2_stratified\test.parquet" `
  --output-dir "$PWD\data\training_workspace\foundation_runs\foundation-sonna-parameter-001\training" `
  --resume-from-checkpoint "$PWD\data\training_workspace\foundation_runs\foundation-sonna-parameter-001\training\checkpoints\last.ckpt" `
  --max-epochs 100 `
  --batch-size 16 `
  --num-workers 8 `
  --no-publish `
  --profile-name "Sonna Parameter Foundation"
```

Promote the resumed final checkpoint:

```powershell
uv run python -c "from pathlib import Path; from sonna_editor.foundation import promote_foundation_checkpoint; promote_foundation_checkpoint(source_ckpt=Path(r'$PWD\data\training_workspace\foundation_runs\foundation-sonna-parameter-001\training\model.ckpt'), display_name='Sonna Parameter Foundation', version_stem='foundation-sonna-parameter-001', source_run_dir=Path(r'$PWD\data\training_workspace\foundation_runs\foundation-sonna-parameter-001'))"
```

## Retrain Foundation Model

Never overwrite an existing foundation checkpoint. Retrain into a new run and
new version stem:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "$PWD\data\training_workspace\sonna_parameter_dataset\splits_v2_stratified" `
  --workspace-dir "$PWD\data\training_workspace" `
  --foundation-repo "$PWD\data\foundation_repo" `
  --profile-name "Sonna Parameter Foundation" `
  --run-name "foundation-sonna-parameter-002" `
  --version-stem "foundation-sonna-parameter-002" `
  --max-epochs 150 `
  --batch-size 16 `
  --workers 8
```

The manifest will point Lite and foundation-based profile creation at the newest
promoted checkpoint.

## Use Foundation For Profiles

- **Personal AI / Mode A:** train user-facing profiles from Sonna RAW+XMP data.
  The backend resolves the hidden foundation checkpoint and warm-starts training
  from it, then publishes only the Personal AI profile into `v1_learning/`.
  Mode A remains RAW+XMP slider regression.
- **Lite / Mode B:** create a `mode_b_initial` profile from the foundation
  checkpoint plus preset plus six-question survey. Initial processing preserves
  preset look sliders and dynamically adjusts Exposure, Temperature, and Tint.
  Later fine-tuning can move the profile through normal model inference.
