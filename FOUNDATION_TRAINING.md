# Foundation Model Training

This file is the foundation-only runbook. The foundation checkpoint is a hidden
base model used by Personal AI and Lite profile creation. It is not a
frontend-visible profile and should not be published into `v1_learning/`.

## Foundation Visibility Rule

- Foundation checkpoints live in a separate repo, normally
  `../SonnaEditorFoundation` or `SONNA_FOUNDATION_REPO`.
- The UI scans only `v1_learning/model-v*.ckpt`, so foundation checkpoints stay
  unseen as long as they remain in the foundation repo.
- Lite profiles and Personal AI profile creation resolve the foundation
  checkpoint from:
  1. `SONNA_FOUNDATION_CHECKPOINT`
  2. `SONNA_FOUNDATION_REPO/foundation_manifest.json`
  3. `SONNA_FOUNDATION_REPO/foundation.ckpt`

## FiveK Dataset Notes

MIT-Adobe FiveK is a good foundation-learning dataset because it contains RAW
DNG inputs, broad scenes/lighting, Lightroom expert edits, and semantic scene
tags. It is not 25,000 independent RAW images. It is 5,000 DNG photos with 5
expert retouches, so using all experts as one target style would give the same
input image multiple conflicting labels. Use one expert, commonly Expert C, for
a single foundation target, or build an explicit expert/style-conditioned
training setup later. The current `SonnaEditor` training path is safest with one
target edit per source image.

FiveK is research-licensed. Keep the downloaded data outside this app repo and
cite the dataset when used.

Recommended local paths:

```powershell
$env:SONNA_TRAINING_WORKSPACE = "D:\SonnaTraining"
$env:SONNA_FOUNDATION_REPO = "D:\SonnaFoundationModel"
$env:FIVEK_ROOT = "D:\Datasets\MITAdobeFiveK"
```

## Build Dataset From RAW + XMP

Use this when your data is already exported as edited RAW/DNG files with matching
Lightroom `.xmp` sidecars:

```powershell
uv run python scripts\train_foundation_model.py `
  --raw-xmp-dir "D:\SonnaTraining\FoundationRawXmp" `
  --workspace-dir "D:\SonnaTraining" `
  --foundation-repo "D:\SonnaFoundationModel" `
  --profile-name "Sonna Foundation" `
  --run-name "foundation-fivek-expert-c-001" `
  --version-stem "foundation-fivek-expert-c-001" `
  --max-epochs 100 `
  --batch-size 16 `
  --workers 8 `
  --init-git
```

Expected outputs:

```text
D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001\
D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001\training\model.ckpt
D:\SonnaFoundationModel\checkpoints\foundation-fivek-expert-c-001.ckpt
D:\SonnaFoundationModel\foundation_manifest.json
```

## Build Dataset From Lightroom Catalog

Use this when the target slider values are in a Lightroom catalog. The catalog is
opened read-only.

```powershell
uv run python scripts\build_dataset_from_catalog.py `
  --catalog-path "D:\Datasets\MITAdobeFiveK\fivek.lrcat" `
  --output-dir "D:\SonnaTraining\fivek_expert_c_dataset" `
  --profile-name "fivek_expert_c" `
  --limit 5000 `
  --workers 8 `
  --split
```

Then train and promote the foundation checkpoint from the prepared splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "D:\SonnaTraining\fivek_expert_c_dataset\splits_v2_stratified" `
  --workspace-dir "D:\SonnaTraining" `
  --foundation-repo "D:\SonnaFoundationModel" `
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
  --train-parquet "D:\SonnaTraining\fivek_expert_c_dataset\splits_v2_stratified\train.parquet" `
  --val-parquet "D:\SonnaTraining\fivek_expert_c_dataset\splits_v2_stratified\val.parquet" `
  --test-parquet "D:\SonnaTraining\fivek_expert_c_dataset\splits_v2_stratified\test.parquet" `
  --output-dir "D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001\training" `
  --resume-from-checkpoint "D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001\training\checkpoints\last.ckpt" `
  --max-epochs 100 `
  --batch-size 16 `
  --num-workers 8 `
  --no-publish `
  --profile-name "Sonna Foundation FiveK Expert C"
```

Promote the resumed final checkpoint:

```powershell
uv run python -c "from pathlib import Path; from sonna_editor.foundation import promote_foundation_checkpoint; promote_foundation_checkpoint(source_ckpt=Path(r'D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001\training\model.ckpt'), display_name='Sonna Foundation FiveK Expert C', version_stem='foundation-fivek-expert-c-001', source_run_dir=Path(r'D:\SonnaTraining\foundation_runs\foundation-fivek-expert-c-001'))"
```

## Retrain Foundation Model

Never overwrite an existing foundation checkpoint. Retrain into a new run and
new version stem:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "D:\SonnaTraining\fivek_expert_c_dataset\splits_v2_stratified" `
  --workspace-dir "D:\SonnaTraining" `
  --foundation-repo "D:\SonnaFoundationModel" `
  --profile-name "Sonna Foundation FiveK Expert C" `
  --run-name "foundation-fivek-expert-c-002" `
  --version-stem "foundation-fivek-expert-c-002" `
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
- **Lite / Mode B:** create a `mode_b_initial` profile from the foundation
  checkpoint plus preset plus six-question survey. Initial processing preserves
  preset look sliders and dynamically adjusts Exposure, Temperature, and Tint.
  Later fine-tuning can move the profile through normal model inference.
