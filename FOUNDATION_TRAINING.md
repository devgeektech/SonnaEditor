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

## Backbone Capacity And Diagnostics

Foundation training uses an adaptive capacity default. Catalog-scale runs start
with the final ConvNeXt stage trainable:

```text
--backbone-unfreeze-strategy progressive
--backbone-trainable-layers stage:7
```

That means the startup state is no longer heads-only. The output heads,
metadata encoder, feature fusion MLP, WB metadata skip, ConvNeXt stage 7, and
backbone norm train from epoch 0. The progressive schedule then expands to
larger backbone sections at later epochs unless you switch to
`--backbone-unfreeze-strategy custom`.

For small foundation splits below 500 train rows, the CLI automatically switches
the default capacity to:

```text
--backbone-unfreeze-strategy custom
--backbone-trainable-layers none
```

This keeps the ConvNeXt backbone frozen and trains only metadata, fusion, and
output heads. The rejected 132-row Sonna continuation overfit while 16.2M
parameters were trainable, so this small-data default is intentionally safer.
Explicit non-default backbone flags are still respected for reviewed ablations.

Measured v2 trainable-capacity presets:

| Trainable layer spec | Trainable params | Notes |
|---|---:|---|
| `none` | 1.92M | Fusion/metadata/heads only; previous heads-only foundation startup. |
| `block:7:2` | 6.68M | Last ConvNeXt block plus fusion/heads. |
| `block:7:2,stage:6` | 7.86M | Closest practical 8M preset. |
| `block:7:1-2,stage:6` | 12.63M | Closest practical 12M preset. |
| `stage:7` | 16.21M | Final ConvNeXt stage; closest practical 15M preset and foundation default. |
| `from:6` | 17.39M | Stage 6 downsample plus all of stage 7. |

Training startup logs now print total/trainable/frozen parameters, trainable
percentage, per-stage backbone state, train/val/test row counts, batches per
epoch, estimated optimizer steps, sampler type, max_steps and
limit_train_batches status, and effective learning rates. The same diagnostic
payload is written into `training_summary.json` under `startup_diagnostics`.

VRAM and speed expectations on the RTX 3050:

- `none`: lowest VRAM, fastest; only about 6.5% of the model learns.
- `block:7:2,stage:6`: modest activation/gradient increase, roughly the
  practical 8M setting.
- `block:7:1-2,stage:6`: medium VRAM and speed cost, roughly the practical 12M
  setting.
- `stage:7`: highest default capacity, roughly 54.5% trainable. Expect more
  gradient memory and slower epochs than heads-only; keep foundation batch size
  at 8 and let CUDA OOM retry reduce it if needed.

If you want a fixed-capacity ablation, combine the spec with:

```powershell
--backbone-unfreeze-strategy custom
```

## Output Head Prior Initialisation

Training logs a line like:

```text
Initialised fresh output heads from training target medians
(Exposure2012=0.000, Temperature=4900, Tint=3.00)
```

or, for foundation warm starts:

```text
Recalibrated warm-start output biases from training target medians
(Exposure2012=0.000, Temperature=4900, Tint=3.00)
```

This is expected. Fresh models zero the final output-head weights and set the
final biases from the training targets. Warm-started models keep learned final
weights and only recenter those final biases. That preserves useful foundation
features while reducing stale output priors, such as a FiveK colour baseline
carrying into a small Sonna continuation. It does not freeze the heads and it
does not keep forcing those values after training starts.

The values come from the current training parquet's target medians. Missing
slider columns or all-missing targets fall back to Lightroom defaults. With the
direct AsShot WB skip enabled, Temperature/Tint head residuals initialise to
zero so the model initially predicts AsShot WB plus learned residual, rather
than a fixed dataset-median WB.

This behavior should remain enabled for fresh foundation and Personal AI
training. Disable it only for a deliberate ablation:

```powershell
--no-target-prior-init
```

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

## Promotion Guardrails

Foundation training now has production guardrails because a tiny Sonna RAW+XMP
continuation can overfit and damage the broader foundation.

Default behavior:

```text
minimum train rows for normal foundation promotion: 75
quality gate: held-out test loss plus key MAE limits
```

If the train split has fewer than 75 rows, `scripts\train_foundation_model.py`
refuses to train/promote unless you explicitly pass:

```powershell
--allow-small-foundation-dataset
```

Use that only for a smoke test, a private ablation, or a deliberately reviewed
run. Do not use it for normal active foundation updates.

After training, promotion is blocked when held-out metrics fail the quality
gate. Override only after visual review:

```powershell
--allow-quality-gate-failure
```

The rejected example was `foundation-sonna-raw-xmp-001`: it used only 132 train
rows, trained 16.2M parameters, overfit, and collapsed Highlights/Shadows. The
active manifest was rolled back to `foundation-fivek-catalog-expert-c-001`.

Future `training_summary.json` files embed train/val/test row counts, parquet
paths, train-batch count, and all-slider `test_per_field_mae`. Run
`scripts\quick_diagnostic.py` after training to see both the critical metrics
and the all-parameter MAE check.

## Tone/Presence Focused Retry

If a foundation run is close on white balance but fails tone or presence gate
metrics, do not build a new dataset first unless the audits show coverage gaps.
Use the same audited splits with per-field loss overrides so the next run puts
more gradient pressure on the weak Lightroom sliders.

The trainer supports repeatable named slider overrides:

```powershell
--field-loss-weight Whites2012=6 `
--field-loss-weight Blacks2012=6 `
--field-loss-weight Highlights2012=5 `
--field-loss-weight Shadows2012=4 `
--field-loss-weight Vibrance=4 `
--field-loss-weight Saturation=4 `
--field-loss-weight Exposure2012=7
```

These weights are recorded in `training_summary.json` under
`hparams.field_loss_weights`, so the run remains reproducible. If the focused
run still fails, inspect prediction collapse and dataset diversity before
collecting more data. Add data only when the current split lacks enough
low-light, high-contrast, strong Whites/Blacks/Highlights, or strong
presence-edit examples.

This is a fresh training run from the prepared train/val/test Parquet splits.
It is not `--resume-from-checkpoint`. By default it warm-starts model weights
from the active foundation checkpoint, recalibrates output biases to the
current train split, trains a new run folder, and promotes a new checkpoint only
if the quality gate passes. Use `--resume-from-checkpoint` only to continue an
interrupted Lightning run from that same run's `checkpoints/last.ckpt`.

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

## Clean Start Before New Foundation Training

Use this only when you intentionally want to remove previous local trained
profiles and foundation checkpoints before a new foundation run. This does not
delete source RAWs, FiveK data, or generated FiveK catalog datasets.

```powershell
$root = Resolve-Path "."
Get-ChildItem -LiteralPath "$root\v1_learning" -File -Filter "model-v*.ckpt" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Get-ChildItem -LiteralPath "$root\v1_learning" -File -Filter "model-v*.json" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Get-ChildItem -LiteralPath "$root\v1_learning" -File -Filter "model-v*-preset.xmp" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Get-ChildItem -LiteralPath "$root\v1_learning" -File -Filter "model-v*-survey.json" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Get-ChildItem -LiteralPath "$root\SonnaEditorFoundation\checkpoints" -File -Filter "*.ckpt" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Get-ChildItem -LiteralPath "$root\SonnaEditorFoundation\checkpoints" -File -Filter "*.json" -ErrorAction SilentlyContinue |
  Remove-Item -Force
Remove-Item -LiteralPath "$root\data\training_workspace\foundation_runs" -Recurse -Force -ErrorAction SilentlyContinue
```

Reset the foundation manifest to an empty version list:

```powershell
@'
{
  "schema_version": 2,
  "active_version": null,
  "active_checkpoint": null,
  "active_sidecar": null,
  "display_name": null,
  "source_run_dir": null,
  "updated_at": null,
  "foundation_type": "sonna_editor_slider_regression",
  "capabilities": [
    "backbone_features",
    "metadata_encoder",
    "slider_regression"
  ],
  "trained_on": [],
  "versions": [],
  "history": []
}
'@ | Set-Content -LiteralPath "$root\SonnaEditorFoundation\foundation_manifest.json" -Encoding UTF8
```

After this cleanup, `resolve_foundation_checkpoint()` is expected to fail until
the next foundation run promotes a new checkpoint. That is normal. Once the new
run finishes, `foundation_manifest.json` becomes the default base-model pointer
for both Mode A and Mode B.

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

The promoted checkpoint becomes the default base model automatically because
`foundation_manifest.json` is updated to point at this new version. Frontend
Personal AI / Mode A training and Lite / Mode B profile creation resolve the
active foundation checkpoint from that manifest unless `SONNA_FOUNDATION_CHECKPOINT`
overrides it.

### Audit The Trained Checkpoint

```powershell
uv run python scripts\analyse_prediction_collapse.py `
  --model-path "SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.ckpt" `
  --parquet "data\training_workspace\fivek_expert_c_catalog_dataset\splits_v2_stratified\val.parquet" `
  --output "data\audits\foundation-fivek-catalog-expert-c-001-collapse.md" `
  --limit 200 `
  --batch-size 16
```

### Push The Foundation Checkpoint To GitHub

Checkpoint binaries are tracked through Git LFS. The parent repo already has:

```text
SonnaEditorFoundation/checkpoints/*.ckpt filter=lfs diff=lfs merge=lfs -text
```

Run `git lfs install` once on every machine that pushes or pulls checkpoints.
No separate checkpoint upload command is needed: after LFS is installed, normal
`git push` uploads the `.ckpt` contents to LFS automatically. On a new machine,
run `git lfs pull` after clone/pull if checkpoint files are still pointer files.

After training and audit, confirm the manifest points to the checkpoint you want:

```powershell
uv run python scripts\rollback_foundation.py --list
uv run python -c "from sonna_editor.foundation import resolve_foundation_checkpoint; print(resolve_foundation_checkpoint())"
```

Then commit and push the new checkpoint, sidecar, and manifest:

```powershell
git status --short
git lfs status
git add SonnaEditorFoundation\foundation_manifest.json `
        SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.ckpt `
        SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.json `
        FOUNDATION_TRAINING.md CLI_COMMANDS.md RUN.md README.md HANDOVER.md SESSION_STATE.md project_knowledge.md
git status --short
git lfs ls-files
git commit -m "train FiveK catalog foundation checkpoint"
git push origin main
```

If you used a different `--version-stem`, replace the two checkpoint filenames
in the `git add` command with the actual new `.ckpt` and `.json` names. Do not
commit files under `data\training_workspace\`; those are local generated runs
and datasets.

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

Tone/presence focused retry from the same splits:

```powershell
uv run python scripts\train_foundation_model.py `
  --splits-dir "data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified" `
  --workspace-dir "data\training_workspace" `
  --foundation-repo "SonnaEditorFoundation" `
  --profile-name "Sonna RAW XMP Foundation Tone Presence 002" `
  --run-name "foundation-sonna-raw-xmp-002-tone-presence" `
  --version-stem "foundation-sonna-raw-xmp-002-tone-presence" `
  --max-epochs 150 `
  --batch-size 8 `
  --workers 8 `
  --field-loss-weight Exposure2012=7 `
  --field-loss-weight Whites2012=6 `
  --field-loss-weight Blacks2012=6 `
  --field-loss-weight Highlights2012=5 `
  --field-loss-weight Shadows2012=4 `
  --field-loss-weight Vibrance=4 `
  --field-loss-weight Saturation=4
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
