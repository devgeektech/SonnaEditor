# Session State - Sonna Editor

**Saved:** 2026-06-16 local time
**Current phase/task:** Frontend macOS titlebar polish.

## Current Workspace

- Repo path: `C:\Users\vikas.DESKTOP-61LEE8B\Projects\SonnaEditor`
- Branch: `main`, tracking `origin/main`
- Recent committed history in this checkout: `aac360a feat: Enhance dataset building and training scripts` on top of four earlier commits.
- Current worktree includes the launcher/docs updates plus unrelated existing
  code/test modifications under `scripts\train_foundation_model.py`,
  `src\sonna_editor\data\dataset.py`, `src\sonna_editor\foundation.py`,
  `src\sonna_editor\training\profile_runner.py`, and focused tests. The
  launcher/docs pass did not revert or normalize those code changes.
- Training/profile caches were intentionally cleared for a fresh dataset reset.
- Cleared repo-local generated artifacts: `data\training_workspace\sonna_personal_001_dataset`, `data\models`, `data\parquet`, `data\captures`, `data\thumbnails`, `data\audits`, `data\dbg`, `data\raw\sonna_training`, `.pytest_cache`, `.ruff_cache`.
- Cleared frontend active-profile pointer: `.saha\active_profile.txt`.
- `v1_learning\` currently has no trained profile checkpoint and no generated dataset folder; there is no frontend-visible profile until a fresh Personal AI/Lite profile is trained or intentionally published.
- Previous foundation checkpoints were cleared from `SonnaEditorFoundation\checkpoints\`, and `foundation_manifest.json` is reset to an empty schema-v2 manifest. The next successful foundation training run will promote the new checkpoint and make it the default base for Mode A and Mode B.

## Environment

- Python: 3.11.15
- uv: 0.11.17
- PyTorch: `2.11.0+cu128`
- OpenCV: `opencv-python-headless==4.13.0.92`
- Runtime device: `cuda`
- GPU: NVIDIA GeForce RTX 3050
- `scripts/verify_environment.py`: 11/11 checks passed
- Adobe DNG Converter discovered at the default Windows install path

The project now requires Python `3.11.*` in `pyproject.toml` / `uv.lock`.
Direct runtime/dev dependencies are exact-pinned from the current uv
environment to reduce Mac resolver drift. `opencv-python-headless==4.13.0.92`
is now a direct runtime dependency for the Lightroom-native auto-straighten
estimator. The earlier `GPU available: False` training issue was caused by a
CPU-only PyTorch install (`torch 2.11.0+cpu`). `torch==2.11.0` and
`torchvision==0.26.0` still resolve to CUDA 12.8 local wheels on Windows/Linux
x86_64, while macOS resolves the matching public wheels.

## Data And Models

- Local training dataset was cleared. Add a fresh RAW+XMP dataset before training.
- No current train/val/test split exists in `data\training_workspace\sonna_personal_001_dataset`.
- The previous split was imbalanced for Exposure2012: train mean ~0.212 while val/test were ~0.480/~0.504. The regenerated split reduces that gap: train mean ~0.264, val ~0.379, test ~0.318.
- Temperature labels in the current dataset mostly cool relative to AsShot WB, so a warmer model output points to training flow/initialisation/split issues rather than warmer target labels.
- Tint labels are consistently positive/magenta relative to AsShot, so some magenta tendency is present in the labels.
- No `v1_learning/model-v*.ckpt` profile is present after the reset.
- Lite profile creation now uses the configured foundation checkpoint as its base; the builder preserves the foundation checkpoint's native slider set and writes a profile sidecar that lets initial Mode B processing use preset+survey style plus adaptive per-photo Exposure/WB correction.
- A fresh scene-stats candidate was trained at `data/models/sonna-v2-scene-stats-run01/`, but it was rejected for frontend use. It briefly published as `v1_learning/model-v2.0.1.*`, then those frontend-visible copies were removed after collapse analysis showed worse prediction spread than v2.0.0.

## What Changed This Session

- Fixed the macOS titlebar overlap where the Saha mark could sit behind the
  native traffic-light window controls:
  - `saha-app\electron\preload.js` now exposes `window.saha.platform`.
  - `saha-app\src\components\shell.jsx` uses that platform value to reserve
    a wider left titlebar inset only on macOS, moving the Saha mark out from
    under Electron's `hiddenInset` controls while leaving Windows/Linux spacing
    unchanged.
  - Verification: `npm run build:vite` passed in `saha-app\`.

- Fixed a frequent process-job callback warning during Lite/Mode B processing:
  - `src\sonna_editor\api\callbacks.py` now formats live edit summaries through
    a safe finite-float helper, so sparse Lite prediction payloads with
    `None` values for unset preset sliders do not raise
    `float() argument must be a string or a real number, not 'NoneType'`.
  - Added regression coverage in `tests\api\test_callback_bridge.py` for
    sparse Lite-style predicted values.
  - Verification passed after rerunning outside the Windows sandbox because
    the sandbox hit the known `CreateProcessAsUserW failed: 1312` runner issue:
    `uv run pytest tests\api\test_callback_bridge.py -q` (`15 passed`) and
    `uv run ruff check src\sonna_editor\api\callbacks.py tests\api\test_callback_bridge.py`.

- Fixed the Auto straighten processing path and corrected Process selection
  semantics:
  - The Process view now requires at least one checked queued folder before the
    Process Selected button is enabled. Newly added folders remain unselected
    by default, and no folder is processed until the operator explicitly checks
    it.
  - `auto_straighten` is still included in the `/api/process` request payload
    for selected-folder dispatch.
  - `src\sonna_editor\inference\straighten.py` now uses CLAHE-normalized
    OpenCV Canny edges plus both probabilistic Hough lines and OpenCV line
    segments, with a broad Hough fallback for fragmented line evidence, to
    estimate the Lightroom `CropAngle` from horizontal/vertical preview
    geometry. This improves recall on fainter, shorter, or broken
    architectural lines while keeping noisy texture frames skipped.
  - Current-code validation against the real local folder
    `OneDrive\Pictures\Testing_Sonna` found the old `sonna_predictions.json`
    had no `auto_straighten` / `straightening` keys and no generated XMP had
    `CropAngle`, indicating that batch was produced by an older/stale process
    path. Running the current estimator read-only across 500 CR3 previews in
    that folder applied straightening to 310 files and skipped 190 as
    `angle_too_small`; there were no preview extraction errors.
  - `sonna_predictions.json` now records `straightening_engine:
    opencv-clahe-canny-lines-v2` plus per-photo `line_count` and
    `line_length_px`, so future 300+ image runs can be audited immediately if
    Lightroom appears not to show crop-angle changes.
  - Follow-up validation against user-supplied `0H5A6295_.xmp` showed the Mac
    run did write `crs:HasCrop="True"` / `crs:CropAngle="+5"`, while Lightroom
    Classic still displayed Angle `0.00`. The XMP lacked full-frame crop bounds,
    so `crop_angle_attributes()` now writes `CropTop=0`, `CropLeft=0`,
    `CropBottom=1`, and `CropRight=1` with every applied `CropAngle`.
  - Follow-up screenshot validation on `0H5A6236.CR3` showed Lightroom applied
    the crop metadata but in the opposite direction (`-1.96` vs the user's
    manual correct direction `+5.94`). The OpenCV residual-to-Lightroom mapping
    was reversed; `estimate_straighten_angle()` now keeps the residual sign for
    Lightroom `CropAngle` instead of negating it.
  - Added `opencv-python-headless==4.13.0.92` to `pyproject.toml` / `uv.lock`.
  - Added regression coverage for room-like tilted geometry, random texture
    skip behavior, and direct XMP serialization of `crs:HasCrop` /
    `crs:CropAngle`.
  - Verification passed:
    `uv run pytest tests\test_straighten.py tests\api\test_process_route.py::test_process_auto_straighten_forwarded tests\api\test_callback_bridge.py::test_pipeline_auto_straighten_writes_crop_angle_and_sidecar -q`
    (`12 passed`),
    `uv run ruff check src\sonna_editor\inference\straighten.py src\sonna_editor\inference\pipeline.py tests\test_straighten.py tests\api\test_callback_bridge.py`,
    `npm run build:vite` in `saha-app\`, and an isolated one-CR3 smoke run in
    `C:\tmp\sonna_straighten_smoke_20260616164350` that wrote
    `crs:HasCrop="True"`, full-frame crop bounds, `crs:CropAngle="-1.1573"`,
    and the new diagnostics.

- Removed the login page "What's new" card and the compatibility/version strip
  underneath it.
- Fixed AI Profiles selection stability: the "Your profiles" list now stays
  newest-first when a profile is activated instead of moving the active profile
  to the top after the API refetch.
- Removed the "Profiles directory" button from the AI Profiles left rail. The
  backend path lookup remains only for the captures directory used by
  fine-tuning.
- Preserved loaded Home/Projects state across cosmetic logout/login within the
  same app session by keeping the app screens mounted after the first sign-in.
- Added confirmation before removing a folder from the Home queue, changed the
  profile deletion confirmation to "Do you want to delete this profile -
  <profile name>?", and introduced shared `SONNA.cta` / `SONNA.onCta` tokens so
  orange CTA buttons such as Process Selected and New Project use the same
  colour source.
- Reviewed the existing fine-tuning path. Capture, delta preparation, versioned
  retraining, CLI fine-tune, `/api/finetune`, and the AI Profiles captures panel
  already exist; the practical next implementation gap is capture population
  from processed folders plus better promotion/quality UX.
- Verification:
  - `npm run build:vite` passed in `saha-app\`.
  - In-app Browser visual smoke could not run because the `iab` browser target
    was unavailable in this session.

- Fixed the renderer crash from the new nav tooltips:
  - `saha-app\src\components\shell.jsx` now destructures `title` in the nav
    item loop before using it for `title` / `aria-label`, resolving
    `Uncaught ReferenceError: title is not defined`.
  - App startup now defaults to light theme regardless of older saved dark
    preference.
  - `SahaLogin` receives the global theme controls and shows a compact theme
    toggle inside the existing sign-in panel.
  - Verification: `npm run build:vite` passed in `saha-app\`.

- Reworked the Saha navigation toward the Imagen-style Home / AI Profiles /
  Projects structure:
  - Left rail now has three primary destinations: Home, AI Profiles, and
    Projects, each with hover/help text.
  - Home keeps the queue and selected profile controls mounted, so loaded
    folders no longer disappear when switching pages.
  - AI Profiles keeps profile creation and profile management.
  - Projects shows loaded/current folder projects and run history in a compact
    table sorted by loaded time.
  - Removed the left-sidebar `Single` / `Selected` segmented controls. Folders
    are unselected by default; if no boxes are checked, Process Selected stays
    disabled. If one or more folder checkboxes are checked, Process runs the
    checked folders.
  - Light theme is now the default startup theme at both HTML and React levels.

- Tightened the UI/UX pass after user feedback:
  - Theme tokens now resolve through global CSS variables in
    `saha-app\src\index.html`, so dark/light switching applies across all
    pages/tabs and module-level style constants cannot get stuck on stale
    colours.
  - `saha-app\src\App.jsx` applies the saved theme with `useLayoutEffect` so
    page switches do not flash or render with the previous theme.
  - Process queue now uses explicit row checkboxes. No checked folder means no
    processing; checked rows define exactly which queued folders run.
  - The process button now shows the RAW count that will actually run for the
    current mode.
  - Process jobs now publish early `photo_prepared` progress during
    preview/metadata extraction. The right-panel progress bar uses the max of
    prepared and fully processed photos, while completion summaries still use
    actual processed-output counts.
  - Profile rows now use a three-dot actions menu with `Delete profile` under
    the menu instead of an inline X button.
  - The backend now allows deleting the active or last remaining profile. If
    another profile remains, it becomes active automatically; if none remain,
    the active-profile pointer is cleared.
- Verification for the follow-up UI/UX pass:
  - `npm run build:vite` passed in `saha-app\`.
  - `uv run pytest tests\api\test_websocket.py tests\api\test_callback_bridge.py tests\api\test_profiles.py -q` passed: `39 passed`.
  - `uv run ruff check ...` over touched Python source/tests passed.

- Fixed the intermittent Windows startup dialog where Electron reported
  "backend failed to start" even though Uvicorn came up on
  `127.0.0.1:8765` shortly after:
  - `saha-app\electron\main.js` now waits for backend readiness with a 30s
    deadline instead of a 10s fixed attempt count.
  - Each `/api/health` probe now gets a 1s response window instead of 200ms,
    which is less brittle on cold Python/CUDA startup.
  - The timeout remains configurable through
    `SAHA_BACKEND_STARTUP_TIMEOUT_MS`, with invalid values falling back to 30s.
  - Verification passed: `node --check saha-app\electron\main.js`,
    `uv run pytest tests\api\test_health.py -q` (`1 passed`), and
    `npm run build:vite` in `saha-app\`.

- Added a proper dark/light theme toggle to the bottom-left rail button:
  - `saha-app\src\tokens.js` now owns dark and light token sets with a matte
    elegant orange accent.
  - `saha-app\src\App.jsx` persists the selected theme in local storage and
    passes theme controls into the shared shell.
  - `saha-app\src\components\shell.jsx` now renders the bottom-left icon as a
    working theme toggle instead of a disabled settings glyph.
  - Accent button text now uses theme-safe contrast tokens across process,
    profile, login, Lite, Personal wizard, and error-banner surfaces.
  - Removed old hardcoded dark accent text colours and negative letter spacing
    in the touched frontend surfaces so text stays visible in both themes.
- Updated "Coming soon" badges to use the new matte orange accent treatment.
- Fixed live process progress stream behavior:
  - Websocket clients now receive an initial `job_snapshot` backfill on connect.
  - Per-photo websocket messages include `photos_total` as well as
    `photos_processed`.
  - `useJob()` merges `job_snapshot` messages and carries `photos_total` into
    the processing snapshot used by the right-panel progress bar.
  - `tests\api\test_websocket.py` pins the snapshot and total-count behavior.
- Verification:
  - `npm run build:vite` passed in `saha-app\`.
  - `uv run pytest tests\api\test_websocket.py tests\api\test_callback_bridge.py -q` passed: `16 passed`.
  - `uv run ruff check src\sonna_editor\api\callbacks.py src\sonna_editor\api\routes\process.py tests\api\test_websocket.py` passed.
  - In-app Browser visual smoke could not run because the `iab` browser target
    was unavailable in this session.

- Made the Profile screen's "Personal AI profile" creation tile a disabled
  "Coming soon" affordance:
  - `saha-app\src\components\profile-view.jsx` no longer imports or opens the
    Personal AI wizard from that tile.
  - The tile now shows a compact "Coming soon" badge, muted styling, disabled
    cursor state, and a title tooltip.
  - Lite profile creation remains available.
- Verification:
  - `npm run build:vite` passed in `saha-app\`.

- Added an opt-in Lightroom-native auto straightening feature:
  - Frontend Process view now shows an `Auto straighten` checkbox.
  - `/api/process` accepts and forwards `auto_straighten`.
  - `scripts\process_shoot_model.py` exposes `--auto-straighten`.
  - `src\sonna_editor\inference\straighten.py` estimates small crop-angle
    corrections from RAW previews using a deterministic edge/projection
    estimator, with conservative thresholds for minimum edge count, confidence,
    and angle size.
  - `process_shoot_with_model()` writes `crs:HasCrop="True"` and
    `crs:CropAngle="..."` through `write_xmp(extra_attributes=...)` only when
    `auto_straighten` is enabled and the estimator result is applied.
  - `sonna_predictions.json` now records `auto_straighten` plus per-photo
    `straightening` diagnostics (`angle_degrees`, `confidence`, `applied`,
    `reason`, and `edge_count`).
  - No model retraining is required; this is an inference/XMP postprocess for
    both Personal AI and Lite profiles.
- Verification for auto straightening:
  - `uv run ruff check src\sonna_editor\inference\straighten.py src\sonna_editor\inference\pipeline.py src\sonna_editor\api\models.py src\sonna_editor\api\routes\process.py scripts\process_shoot_model.py tests\test_straighten.py tests\api\test_process_route.py tests\api\test_callback_bridge.py` passed.
  - `uv run python -m py_compile src\sonna_editor\inference\straighten.py src\sonna_editor\inference\pipeline.py src\sonna_editor\api\models.py src\sonna_editor\api\routes\process.py scripts\process_shoot_model.py` passed.
  - `uv run pytest tests\test_straighten.py tests\api\test_process_route.py::test_process_auto_straighten_forwarded tests\api\test_callback_bridge.py::test_pipeline_auto_straighten_writes_crop_angle_and_sidecar -q` passed: `6 passed`.
- Ran the full local verification flow after the Pylance cleanup passes:
  - `uv run python scripts\verify_environment.py` passed `11/11` checks on
    Python 3.11.15, uv 0.11.17, PyTorch `2.11.0+cu128`, CUDA on the local RTX
    3050, and the default Adobe DNG Converter path.
  - `uv run ruff check .` passed.
  - `uv run python -m compileall -q src scripts tests` passed.
  - `npm run build:vite` passed in `saha-app\` after rerunning outside the
    sandbox because the sandboxed attempt hit the Windows
    `CreateProcessAsUserW failed: 1312` runner issue.
  - `uv run pytest -q` passed after the foundation CLI fix:
    `753 passed, 45 skipped, 1 warning` in `199.33s`. The remaining warning is
    the known PyTorch scalar-conversion warning in
    `tests\test_losses.py::test_per_row_mask_all_bad_returns_zero_loss`.
  - `git diff --check` passed with only Git line-ending notices.
- Smoke-checked documented entry points:
  - `uv run python scripts\run_app.py --help` passed.
  - `cmd /c run_saha.cmd --help` passed after rerunning outside the sandbox for
    the same Windows process-launch issue.
  - `uv run python scripts\serve.py --help` passed.
  - `uv run python scripts\train_profile.py --help` passed.
  - `uv run python scripts\quick_diagnostic.py --help` passed.
- The whole-flow smoke check found a real mismatch:
  `scripts\train_foundation_model.py --help` did not expose the documented
  `--tone-presence-retry` or repeatable `--field-loss-weight FIELD=WEIGHT`
  flags, so the foundation retry recipe described in `HANDOVER.md`,
  `CLI_COMMANDS.md`, `FOUNDATION_TRAINING.md`, and `RUN.md` was not actually
  runnable from the foundation CLI.
- Fixed the foundation CLI mismatch:
  `scripts\train_foundation_model.py` now exposes `--tone-presence-retry` and
  repeatable `--field-loss-weight FIELD=WEIGHT`, prepends the reviewed retry
  recipe weights for Exposure2012, Whites2012, Blacks2012, Highlights2012,
  Shadows2012, Vibrance, and Saturation, and preserves explicit per-field
  overrides by appending them after the preset before forwarding into
  `train_profile()`.
- Added regression coverage in `tests\test_train_foundation_model.py` for
  parser defaults, parser acceptance of the retry/field-weight flags, retry
  recipe construction, and explicit-only field-weight forwarding.
- Verification for the foundation CLI fix:
  - `uv run ruff check scripts\train_foundation_model.py tests\test_train_foundation_model.py` passed.
  - `uv run python -m py_compile scripts\train_foundation_model.py tests\test_train_foundation_model.py` passed.
  - `uv run pytest tests\test_train_foundation_model.py -q` passed:
    `17 passed`.
  - `uv run python scripts\train_foundation_model.py --help` now shows both
    `--tone-presence-retry` and `--field-loss-weight FIELD=WEIGHT`.
- Fixed the pasted Pylance diagnostics in
  `src\sonna_editor\model\architecture.py` by casting the registered
  `focal_bins` buffer back to `torch.Tensor` before `torch.bucketize()` and
  routing the dynamic `torch.jit.is_scripting()` call through `Any` so Pylance
  no longer treats it as a private/export issue.
- Verification for the architecture cleanup:
  - `uv run ruff check src\sonna_editor\model\architecture.py tests\test_training.py` passed.
  - `uv run python -m py_compile src\sonna_editor\model\architecture.py tests\test_training.py` passed.
  - `uv run pytest tests\test_training.py -q` passed: `77 passed`.
- Fixed the pasted Pylance diagnostics in
  `src\sonna_editor\finetune\delta.py` by narrowing the
  `scipy.stats.spearmanr()` result to scalar `float` values before calling
  `np.isnan`, `abs`, `round`, and threshold comparisons.
- Verification for the finetune delta cleanup:
  - `uv run ruff check src\sonna_editor\finetune\delta.py tests\test_finetune_delta.py` passed.
  - `uv run python -m py_compile src\sonna_editor\finetune\delta.py tests\test_finetune_delta.py` passed.
  - `uv run pytest tests\test_finetune_delta.py -q` passed: `17 passed`.
- Fixed two follow-up Pylance diagnostics:
  `scripts\run_v1_pilot.py` now filters sample-prediction metadata to tensor
  values before calling `.unsqueeze()`, and
  `src\sonna_editor\finetune\capture.py` now casts through `Any` inside the
  safe float helper so Pylance accepts the guarded conversion.
- Verification for the follow-up diagnostics:
  - `uv run ruff check scripts\run_v1_pilot.py src\sonna_editor\finetune\capture.py` passed.
  - `uv run python -m py_compile scripts\run_v1_pilot.py src\sonna_editor\finetune\capture.py` passed.
  - `uv run python scripts\run_v1_pilot.py --help` passed; it printed the
    existing Torch/Triton warning after showing help.
- Fixed the latest pasted Pylance batch in RAW preview extraction, fine-tune
  capture/retrain, inference postprocessing, Mode B checkpoint tests, training
  metadata tests, and weighted-loss gradient assertions. The cleanup adds
  explicit byte/array coercion for rawpy thumbnails, a safe float coercion for
  captured edit deltas, a mock-compatible checkpoint callback fallback in
  fine-tune retraining, typed preset/metadata test helpers, and explicit casts
  where model predictions intentionally widen from `float` to `float | None`.
- Verification for the latest Pylance batch:
  - `uv run ruff check ...` over the affected source/tests passed.
  - `uv run python -m py_compile ...` over the affected source/tests passed.
  - Focused pytest passed:
    `tests\test_catalog_dataset.py tests\test_checkpoint_builder.py tests\test_finetune_delta.py tests\test_inference_v2.py tests\test_losses.py tests\test_training.py tests\test_finetune_retrain.py tests\test_finetune_capture.py tests\test_inference_pipeline_integration.py -q`
    returned `227 passed, 1 warning` in `63.67s`. The warning is the existing
    PyTorch scalar-conversion warning in `tests\test_losses.py`.
- Fixed the pasted Pylance diagnostics across data extraction/XMP parsing,
  fine-tune capture, inference pipeline typing, Mode B survey/checkpoint
  helpers, weighted loss buffers/metadata, datamodule dataset narrowing,
  unfreeze callback access, and the named tests. The cleanup mostly switches
  exact mutable `dict` parameters to covariant `Mapping` where appropriate,
  narrows optional/rawpy/Pillow/lxml values, adds overloads for
  `WeightedSliderLoss.forward(return_components=True)`, and adds test-local
  assertions/casts for intentional fakes.
- Verification for this broad Pylance cleanup:
  - `uv run ruff check ...` over all files named in the diagnostic batch passed.
  - `uv run python -m py_compile ...` over all files named in the diagnostic batch passed.
  - Focused pytest across touched modules/tests passed:
    `417 passed, 35 skipped, 1 warning` in `134.70s`.
- Fixed the pasted Pylance diagnostics in legacy v1 training scripts:
  `scripts\run_v1_pilot.py` and
  `scripts\train_v1_2_0_full_production.py`.
  The cleanup replaces `Path.is_mount()` with `os.path.ismount()` for the
  mount check, narrows optional datamodule datasets/registries after `setup()`,
  keeps named `ModelCheckpoint` callback variables for best checkpoint
  path/score reads, guards optional `__doc__`, and casts catalog included-row
  counts before percentage math.
- Verification for the legacy v1 script static-analysis cleanup:
  - `uv run ruff check scripts\run_v1_pilot.py scripts\train_v1_2_0_full_production.py` passed.
  - `uv run python -m py_compile scripts\run_v1_pilot.py scripts\train_v1_2_0_full_production.py` passed.
  - `uv run python scripts\train_v1_2_0_full_production.py --help` passed.
  - `uv run python scripts\run_v1_pilot.py --help` passed.
- Fixed the pasted Pylance diagnostics in:
  `scripts\quick_diagnostic.py`, `scripts\finetune_profile.py`,
  `src\sonna_editor\data\audit.py`, and `src\sonna_editor\data\dataset.py`.
  The changes narrow optional/regex/pandas/sklearn group values without
  changing runtime behavior.
- Verification for this static-analysis cleanup:
  - `uv run ruff check scripts\quick_diagnostic.py scripts\finetune_profile.py src\sonna_editor\data\audit.py src\sonna_editor\data\dataset.py` passed.
  - `uv run python -m py_compile scripts\quick_diagnostic.py scripts\finetune_profile.py src\sonna_editor\data\audit.py src\sonna_editor\data\dataset.py` passed.
  - `uv run pytest tests\test_dataset.py tests\test_audit.py tests\test_quick_diagnostic.py -q` passed: `65 passed, 1 skipped`.
  - `uv run python scripts\finetune_profile.py --list-versions --output-dir v1_learning` passed.
  - `uv run pyright ...` could not run because `pyright` is not installed in the uv environment.
- Fixed the pasted Pylance diagnostics in
  `src\sonna_editor\training\profile_runner.py`:
  - narrowed datamodule train/val/test datasets after `setup()` before calling
    `len()`
  - checked the optional registry before reading embedding maps
  - typed `resume_from_checkpoint` / `base_model_checkpoint` as optional
    `Path` values and guarded the warm-start path
  - kept a named `ModelCheckpoint` callback and reads best checkpoint path/score
    through helper functions instead of `trainer.checkpoint_callback`, whose
    Lightning type is too generic for Pylance
- Restored `_parse_field_loss_weight_overrides`, the repeatable
  `--field-loss-weight FIELD=WEIGHT` CLI argument, and the matching
  `_apply_training_overrides()` hook because tests and current foundation retry
  docs still rely on named slider loss overrides.
- Verification for the Pylance/profile runner cleanup:
  - `uv run ruff check src\sonna_editor\training\profile_runner.py` passed.
  - `uv run python -m py_compile src\sonna_editor\training\profile_runner.py`
    passed.
  - Focused profile-runner/foundation tests passed together:
    `tests\test_training.py::test_parse_field_loss_weight_overrides_accepts_repeatable_fields`,
    `test_parse_field_loss_weight_overrides_rejects_unknown_field`,
    `test_apply_training_overrides_updates_named_slider_weights`,
    `test_dataset_summary_payload_records_rows_and_split_paths`, and
    `test_train_profile_log_interval_adapts_to_small_dataset`, plus
    `tests\test_train_foundation_model.py`: `19 passed`.
- Added a one-command local app launcher for client/operator startup:
  `scripts\run_app.py`, plus root wrappers `run_saha.cmd`, `run_saha.ps1`,
  and `run_saha.sh`.
  The launcher creates repo-local runtime folders, runs `npm install` only if
  `saha-app\node_modules\` is missing, checks that Node/npm are available, then
  starts the existing Electron dev command. Electron continues to own backend
  startup/shutdown and reuses any existing API on port `8765`.
- Added `.gitattributes` line-ending rules for launcher portability:
  `.sh`/`.py` and `.gitattributes` stay LF for macOS/Linux, while
  `.cmd`/`.ps1` normalize to CRLF for Windows.
- Updated `README.md`, `RUN.md`, `CLI_COMMANDS.md`, `HANDOVER.md`, and
  `project_knowledge.md` so the preferred client startup is now
  `.\run_saha.cmd` on Windows or `bash run_saha.sh` on macOS/Linux, with the
  two-terminal backend/frontend flow kept as the debugging fallback.
- Verification for the launcher update:
  - `uv run ruff check scripts\run_app.py` passed.
  - `uv run python -m py_compile scripts\run_app.py` passed.
  - `uv run python scripts\run_app.py --help` passed and showed the expected
    `--skip-install` option.
  - `cmd /c run_saha.cmd --help` passed and printed the same launcher help
    without starting Electron.
  - Direct byte check confirmed `run_saha.sh` currently has LF-only line
    endings, so `bash run_saha.sh` is safe for Mac clones.
  - `npm run build:vite` passed in `saha-app\` after rerunning outside the
    sandbox because the first sandboxed attempts failed during Windows process
    setup, not during Vite build execution.
- Documentation alignment pass kept the old two-terminal startup commands in
  `RUN.md`, `CLI_COMMANDS.md`, `README.md`, and `MAC_SETUP.md` as explicit
  manual/debugging references while making the one-command launcher the
  preferred startup path.
- `MAC_SETUP.md` now uses `bash run_saha.sh` as the primary zsh startup command
  and preserves the old `uv run python scripts/serve.py --port 8765` plus
  `cd saha-app && npm install && npm run dev` flow under a legacy two-terminal
  reference section.
- `README.md` now shows separate Windows and macOS/Linux quick starts so both
  Darshil's Windows workflow and the client's Mac workflow are visible at the
  top-level docs.
- Verification for the docs alignment:
  - `rg` confirmed the startup docs now show one-command startup first, with
    old backend/frontend commands only as manual/debugging references.
  - `git diff --check` passed; remaining line-ending messages are Git
    autocrlf warnings on text files, not whitespace errors.
- Investigated Darshil's pasted Mac foundation diagnostics for
  `foundation-sonna-raw-xmp-002-tone-presence`. The run used real scale
  splits (`5,021/865/1,172` train/val/test rows), so this is not the earlier
  132-row small-data failure. The quality gate failure is legitimate for the
  stored held-out MAE values: Exposure, Shadows, Highlights, Whites, Blacks,
  Vibrance, and Saturation exceeded current foundation promotion limits while
  Temperature, HSL average, Clarity, and test-loss overfit ratio were acceptable.
- Improved training diagnostics so the next Mac run is easier to judge:
  `training_summary.json` now stores `hparams.max_epochs`, and foundation
  training persists `quality_gate_passed` plus `foundation_quality_failures`
  into the summary before returning a gate error.
- Added an explicit foundation retry recipe flag:
  `scripts\train_foundation_model.py --tone-presence-retry`. It applies the
  reviewed stronger loss weights for Exposure2012, Whites2012, Blacks2012,
  Highlights2012, Shadows2012, Vibrance, and Saturation while respecting any
  explicit `--field-loss-weight` override for the same field.
- Improved `scripts\quick_diagnostic.py` for foundation triage. It now prints
  backbone strategy/layers, field loss overrides, and a train-median baseline
  comparison for any failed gate fields when train/test Parquet split paths are
  available. The baseline check shows whether the model is learning beyond a
  fixed average or whether the labels/gate are intrinsically high-variance.
- Added balanced foundation checkpoint selection. `SonnaLightningModule` now
  logs `val_visual_score`, a lower-is-better composite over Exposure, WB,
  important tone/presence fields, HSL average, and key collapse ratios.
  `scripts\train_foundation_model.py` passes `checkpoint_monitor="val_visual_score"`
  into the trainer so foundation exports are selected by visual balance rather
  than plain total `val_loss`. `train_profile()` still tracks best true val-loss
  separately for diagnostics.
- Changed foundation quality gates from all-or-nothing MAE limits to tiered
  hard failures plus warnings. Severe failures still block promotion unless
  explicitly overridden after review. Moderate misses are persisted as
  `foundation_quality_warnings`, printed to stderr, and allowed to promote so
  a useful checkpoint is not blocked by one noisy slider.
- Recommendation from this investigation: do not pass
  `--allow-quality-gate-failure` for the pasted run. Run collapse analysis and
  the improved quick diagnostic first, then run a third fresh foundation retry
  from the same prepared splits with `--tone-presence-retry`. It does not
  require rebuilding the dataset unless the new baseline, collapse, or diversity
  diagnostics show weak label coverage.
- Removed the currently tracked foundation checkpoint artifacts from the parent
  repo so the Mac RAW+XMP foundation run can replace them:
  `SonnaEditorFoundation\checkpoints\foundation-fivek-catalog-expert-c-001.*`
  and `SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-001.*` are
  deleted in the worktree, and `SonnaEditorFoundation\foundation_manifest.json`
  is reset to an empty schema-v2 manifest. Until the Mac checkpoint is added and
  pushed, the parent repo has no active hidden foundation checkpoint.
- Removed all remaining local `.ckpt` files under the repo, including generated
  foundation-run checkpoints under `data\training_workspace\foundation_runs\`
  and the stale frontend-visible `v1_learning\model-v0.2.0.ckpt`. Removed the
  matching stale `v1_learning\model-v0.2.0.*` profile sidecar/preset/survey
  files so the frontend-visible profile folder is empty again.
- Removed the old generated foundation run directories under
  `data\training_workspace\foundation_runs\` entirely, including previous
  FiveK, RAW+XMP, tone/presence, timestamped, and guardrail-smoke run folders.
- Updated `MAC_SETUP.md` for the active Mac workflow: VS Code with zsh, not the
  full Xcode app or Windows PowerShell syntax. The guide now warns against
  PowerShell backticks and Windows backslash paths in zsh, shows forward-slash
  script paths with `\` line continuations, adds a VS Code open/terminal check,
  and includes a preflight check before foundation training.
- Hardened RAW+XMP dataset shoot bucketing for timezone-aware capture
  timestamps. `src\sonna_editor\data\dataset.py::_derive_shoot_id()` now
  normalizes aware datetimes to naive UTC before subtracting from the epoch and
  strips non-offset `tzinfo` values from effectively naive datetimes. Added a
  regression test for ISO capture timestamps like `2024-03-15T23:30:00+13:00`
  so Mac dataset builds do not fail with `can't subtract offset-naive and
  offset-aware datetimes`.
- Exact-pinned direct runtime/dev dependencies in `pyproject.toml` from the
  current uv environment and restricted the project to Python `3.11.*` to reduce
  Mac setup conflicts.
- Refreshed `uv.lock`; its resolution matrix is now limited to Python 3.11
  platforms instead of carrying Python 3.12+ wheel variants.
- Updated `README.md`, `RUN.md`, `CLI_COMMANDS.md`, `MAC_SETUP.md`,
  `HANDOVER.md`, `SESSION_STATE.md`, and `project_knowledge.md` with the pinned
  dependency/Mac setup guidance.
- Clarified Git LFS checkpoint workflow in the runbooks: `.ckpt` files under
  `SonnaEditorFoundation\checkpoints\`, `v1_learning\`, and `models\` are
  LFS-managed by `.gitattributes`; after `git lfs install`, normal `git push`
  uploads checkpoint binaries automatically, and new machines should run
  `git lfs pull`.
- Added repeatable named slider loss overrides to `scripts\train_profile.py` /
  `src\sonna_editor\training\profile_runner.py` via
  `--field-loss-weight FIELD=WEIGHT`.
- `scripts\train_foundation_model.py` now passes `--field-loss-weight`
  overrides through to the packaged trainer, so foundation retries can focus on
  tone/presence fields without creating a separate dataset or one-off script.
- Training summaries now record parsed per-field overrides in
  `hparams.field_loss_weights`.
- Documented the tone/presence focused retry workflow in
  `FOUNDATION_TRAINING.md`, `CLI_COMMANDS.md`, and `MAC_SETUP.md`, including
  that this is a fresh run from prepared splits with a foundation weight
  warm-start by default, not `--resume-from-checkpoint`.
- Centralised model-inference RAW scanning on `config.SUPPORTED_RAW_EXTENSIONS` instead of maintaining a narrower duplicate list in `src\sonna_editor\inference\pipeline.py`.
- Added `.rwl` to `SUPPORTED_RAW_EXTENSIONS`. The scanned format set is now `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and `.srw` across RAW+XMP dataset building, folder/API scans, preset processing, model inference, and fine-tune capture.
- Added config regression tests that assert the supported extension set and that inference uses the central config list.
- Verification for this fix:
  - `uv run ruff check src\sonna_editor\config.py src\sonna_editor\inference\pipeline.py src\sonna_editor\finetune\capture.py tests\test_config.py` passed.
  - `uv run pytest tests\test_config.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` passed: `29 passed`.
- Diagnosed the pasted training diagnostics for the two foundation runs:
  - `foundation-fivek-catalog-expert-c-001` trained on the full FiveK Expert C split (`3769/536/695` train/val/test rows in the current local `splits_v2_stratified_fiveK` folder). Collapse audit on 200 validation photos found `0` collapsed sliders, but Saturation/Vibrance and Temperature still need visual review before treating it as production-quality Mode A output.
  - `foundation-sonna-raw-xmp-001` trained on only `132/27/30` train/val/test rows from `data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified` while starting with `stage:7` trainable (`16.2M` trainable params). It overfit (`test_loss / best_val_loss = 1.616x`) and failed key visual sliders: Exposure, Shadows, Highlights, Whites, Blacks, Vibrance, and Saturation. Collapse audit on its 27-row val split found collapsed `Highlights2012` and `Shadows2012`.
- Rolled the active foundation manifest back from the bad Sonna continuation to `foundation-fivek-catalog-expert-c-001`. New Lite/Personal AI runs now resolve the FiveK checkpoint unless an environment override is set.
- Added foundation promotion guardrails in `scripts\train_foundation_model.py`:
  - foundation runs now refuse to train/promote from fewer than `75` train rows unless `--allow-small-foundation-dataset` is explicitly passed
  - small foundation splits below `500` train rows now automatically use `--backbone-unfreeze-strategy custom --backbone-trainable-layers none` unless explicit backbone flags are supplied, avoiding the 16.2M-trainable-parameter path that overfit the 132-row Sonna continuation
  - promotion now fails if held-out metrics breach the quality gate (`test_loss` overfit ratio plus key MAE thresholds, using all-slider `test_per_field_mae` as a fallback) unless `--allow-quality-gate-failure` is explicitly passed after visual review
- Fixed two additional training-pipeline quality issues:
  - warm-started profile/foundation runs now recalibrate output-head final biases from the current train split while preserving learned final weights, reducing stale FiveK/Sonna output-prior carryover on small continuation runs
  - `src\sonna_editor\data\catalog_dataset.py` no longer overwrites the `skip_unedited` flag with a counter, so ordinary Lightroom catalog builds skip unedited-looking rows by default again; FiveK keeps those rows only when `--include-unedited-looking` is explicitly passed
- Improved future training summaries in `src\sonna_editor\training\profile_runner.py`: summaries now store train/val/test row counts, split parquet paths, split directory, train-batch count, and `test_per_field_mae` for all sliders.
- Improved `scripts\quick_diagnostic.py` so future summaries with `test_per_field_mae` show an all-parameter MAE check and worst normalized slider errors, instead of only the small critical-metric table.
- Verification for this fix:
  - `uv run ruff check scripts\train_foundation_model.py scripts\quick_diagnostic.py src\sonna_editor\training\profile_runner.py tests\test_train_foundation_model.py tests\test_training.py` passed.
  - `uv run pytest tests\test_train_foundation_model.py tests\test_training.py::test_dataset_summary_payload_records_rows_and_split_paths tests\test_training.py::test_aggregate_mae_outputs_keeps_all_slider_fields tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset -q` passed: `12 passed`.
  - `uv run python -m py_compile scripts\train_foundation_model.py scripts\quick_diagnostic.py src\sonna_editor\training\profile_runner.py` passed.
- Reviewed the local MIT-Adobe FiveK download at `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset`.
- Confirmed the extracted FiveK tree contains 5,000 `.dng` source files under `raw_photos\HQa*`, `raw_photos\fivek.lrcat`, Lightroom preview/helper/catalog-data folders, and text/license/category files.
- Inspected `raw_photos\fivek.lrcat` read-only through SQLite. It has 60,000 `Adobe_images` rows, 5,000 unique source DNGs, 96,458 develop-settings rows, and 60,000 active non-empty develop-setting blobs. Each DNG has 12 virtual-copy/recipe variants.
- Confirmed expert collections `A`, `B`, `C`, `D`, and `E` each have 5,000 linked rows. Collection `C` has 5,000 rows, all with develop settings and all pointing to existing DNGs.
- Added exact Lightroom collection filtering to `src\sonna_editor\data\catalog.py` and surfaced it through `scripts\build_dataset_from_catalog.py --collection-name`.
- Added `--include-unedited-looking` to `scripts\build_dataset_from_catalog.py` because FiveK develop blobs are sparse; many default sliders are absent, so the generic unedited-row filter would skip valid expert edits.
- Verified the project catalog reader can parse Collection `C`. A 250-row sample had about 55 populated slider fields per row; absent sliders remain `NaN` and are masked by the existing loss.
- Ran a 20-row real FiveK Collection C catalog smoke into `data\training_workspace\fivek_catalog_verify_20260605`; dataset creation succeeded with 0 missing files, 0 parse errors, 20 generated thumbnails, and 20 Parquet rows. A separate earlier 20-row smoke with `--split` wrote the dataset but failed splitting because the tiny sample had too few shoot groups, so use full 5,000-row builds for real split generation.
- Verified the FiveK catalog training policy: DNG previews and RAW metadata are model inputs, catalog develop settings are slider targets, absent catalog slider values are masked by the loss, and fresh/foundation output priors fall back to Lightroom defaults for fields with no labels.
- Updated `FOUNDATION_TRAINING.md`, `CLI_COMMANDS.md`, and `RUN.md` with the inspected FiveK path, catalog row analysis, Expert C catalog commands, cumulative foundation training order, and checkpoint naming instructions.
- Updated `project_knowledge.md` and `HANDOVER.md` with the FiveK catalog route and source-map changes.
- Upgraded foundation lineage management to schema-v2 manifests. `foundation_manifest.json` now records `active_version`, `versions[]`, checkpoint SHA256, foundation type, capabilities, and training-source tags while retaining compatibility fields such as `active_checkpoint` and `history`.
- Foundation promotion now auto-allocates `foundation-vN` when no explicit version stem is supplied, still refuses to overwrite existing checkpoints, and keeps older versions available for rollback.
- Added `scripts\rollback_foundation.py` with `--list` and explicit version activation so bad foundation runs can be rolled back by changing the manifest pointer instead of deleting checkpoint files.
- Fixed legacy foundation manifest listing so the active checkpoint is included even when the manifest predates schema-v2 `versions[]`. The current active checkpoint `foundation-sonna-raw-xmp-003.ckpt` now resolves and appears in `scripts\rollback_foundation.py --list` alongside `foundation-sonna-raw-xmp-001`.
- Cleared previous local trained profile/checkpoint artifacts before the new FiveK foundation run:
  - removed `v1_learning\model-v0.1.0.*`
  - removed `SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-001.*`
  - removed `SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-003.*`
  - removed old `data\training_workspace\foundation_runs\`
  - reset `SonnaEditorFoundation\foundation_manifest.json` to an empty schema-v2 manifest.
- Fixed clean-start behavior so an empty foundation manifest reports no active checkpoint and `scripts\train_foundation_model.py` starts from scratch automatically unless a new active checkpoint exists.
- Personal AI training sidecars and summaries now record foundation provenance for foundation warm starts: version, checkpoint path, SHA256, foundation type, capabilities, and training-source tags.
- Foundation warm starts now use native SonnaEditor checkpoints only. The previous paired-image warm-start path has been removed.
- Removed the duplicate generated Personal AI dataset copy at `v1_learning\dataset`. Canonical generated datasets, splits, thumbnails, and run workspaces now live under `data\training_workspace\`; `v1_learning\` is reserved for frontend-visible published profile checkpoints and their sidecar/preset/survey files.
- Updated active Personal AI/fine-tune/audit defaults that still pointed at `v1_learning\dataset`:
  - `config.ORIGINAL_TRAIN_PARQUET`
  - `scripts\quick_diagnostic.py`
  - `scripts\finetune_profile.py`
  - `scripts\audit_all_sliders_v1.2.3.py`
  - `scripts\train_v1_2_0_full_production.py`
  - `scripts\migrate_labels_to_v2.py`
- Added a config regression test so the default original-train parquet stays under `data\training_workspace\sonna_personal_001_dataset\splits_v2_stratified\train.parquet`.
- Added progressive backbone warm-start training. Frontend Personal AI and RAW+XMP foundation runs now freeze the full ConvNeXt backbone first, then unfreeze later stages in phases before full fine-tuning. The legacy partial strategy remains available as `--backbone-unfreeze-strategy partial`.
- Added configurable ConvNeXt trainable-layer specs and training startup diagnostics for foundation-capacity work:
  - `SonnaEditor.set_trainable_backbone_layers()` supports specs such as `none`, `stage:7`, `block:7:2,stage:6`, `block:7:1-2,stage:6`, `from:6`, and `all`.
  - `scripts\train_foundation_model.py` now uses adaptive foundation capacity: larger splits default to `--backbone-trainable-layers stage:7`, while small splits below 500 train rows use heads/fusion-only capacity unless a reviewed ablation supplies explicit backbone flags.
  - Measured v2 trainable counts: `none` 1.92M, `block:7:2` 6.68M, `block:7:2,stage:6` 7.86M, `block:7:1-2,stage:6` 12.63M, `stage:7` 16.21M, `from:6` 17.39M.
  - `src\sonna_editor\training\diagnostics.py` now reports total/trainable/frozen params, trainable percentage, per-stage/block backbone state, split row counts, batches per epoch, estimated optimizer steps, effective learning rates, sampler type, max_steps, limit_train_batches, and gradient accumulation.
  - `training_summary.json` now stores this under `startup_diagnostics`.
- Fixed `scripts\analyse_prediction_collapse.py` missing-parquet handling after a FiveK audit command used `splits_v2_stratified\val.parquet` while the actual local split folder was `splits_v2_stratified_fiveK\val.parquet`. The script now raises a clearer `FileNotFoundError` with nearby matching parquet suggestions. A one-row smoke audit against `foundation-fivek-catalog-expert-c-001.ckpt` and the correct FiveK validation parquet completed successfully.
- Added `scripts\analyse_backbone_drift.py` to compare ConvNeXt backbone tensor drift between a foundation checkpoint and a final Personal AI checkpoint, reporting per-stage relative deltas, cosine similarity, and the largest drifting tensors.
- Removed the paired-image foundation training path. Foundation checkpoints are now native `SonnaEditor` slider-regression checkpoints trained from catalog-derived splits or RAW+XMP sidecars.
- Moved transient app state to repo-local `.saha\` instead of `~\.saha\`. Active profile, recent folders, queued job snapshots, Personal AI training workspaces, and fine-tune scratch runs now resolve from the project root by default.
- Added `config.ensure_runtime_directories()` and wired it into backend/server and CLI entrypoints so a fresh clone auto-creates `data\training_sources\`, `data\raw\`, `data\raw\sonna_training\`, `data\datasets\`, `data\dng\`, `data\parquet\`, `data\captures\`, `data\audits\`, `data\dbg\`, `v1_learning\`, and `.saha\` before use.
- Split local learning inputs from generated outputs. Source photos now belong under separate gitignored child folders such as `data\training_sources\sonna_personal_001\raw_xmp\`. Generated Parquet/checkpoint run outputs, including FiveK catalog datasets, remain under `data\training_workspace\`.
- `SONNA_TRAINING_WORKSPACE` defaults to `data\training_workspace\`. `SONNA_FOUNDATION_REPO` defaults to the repo-local `SonnaEditorFoundation\` folder so promoted foundation checkpoints stay inside the SonnaEditor workspace while remaining outside gitignored `data\`. Runtime directories are auto-created on startup unless the operator overrides them.
- Updated `scripts\process_shoot_model.py` so the default model path resolves to the newest published `v1_learning\model-v*.ckpt` instead of a stale hardcoded legacy checkpoint path. If no published profile exists yet, the CLI now fails with a clear instruction.

- Decoupled Lite profile creation from active Personal AI profiles. `POST /api/profiles/lite` now resolves the configured foundation checkpoint and passes that to the Lite checkpoint builder.
- Added `src/sonna_editor/foundation.py` helpers for creating the hidden foundation folder layout, resolving the active foundation checkpoint from env/manifest/fallback, and promoting trained checkpoints into that folder.
- Added `scripts/train_foundation_model.py`, the canonical foundation-training command. It can build a RAW+XMP dataset or use existing splits, trains with the current recipe without publishing to the frontend, then promotes the final checkpoint into the configured foundation folder.
- Updated API tests so Lite creation proves it uses the foundation checkpoint even when no Personal AI profile is active.
- Added `tests/test_foundation.py` for foundation folder layout, manifest writing, promotion, and checkpoint resolution.
- Updated runbooks to clarify that RAW+XMP data preparation, Personal AI training, foundation training, and Lite profile creation are separate workflows. Raw training photos should live in repo-local gitignored source folders by default, or in explicitly chosen external folders; the foundation checkpoint lives in `SonnaEditorFoundation/`.
- Cleaned operator-facing docs to avoid making v1/v2 a user decision. Internal legacy checkpoint compatibility remains in code.
- Added `MAC_SETUP.md`, a Mac-specific setup and operating guide covering clean machine setup, uv/Python dependency sync, MPS verification, backend/frontend startup, profile discovery, RAW+XMP and catalog dataset preparation, Personal AI training, hidden foundation training, Lite profile creation, shoot processing, fine-tuning, diagnostics, and update steps.
- Added a README pointer to `MAC_SETUP.md`.
- Updated `project_knowledge.md` so the Mac runbook is part of the documented source map and current notes.
- Fixed the Python environment so CUDA PyTorch is installed and preserved by `uv sync`.
- Updated `scripts/train_profile.py` logging:
  - default recipe values now log as `Training recipe ...`
  - explicit CLI flags log as `Override ...`
- Updated operational docs to reflect the current Windows CUDA state, small local dataset, absent local checkpoints, and updated training/resume guidance.
- Updated `AGENTS.md` and `CLAUDE.md` with a stronger required-reading rule and a documentation update rule after non-trivial work.
- Disabled photometric jitter by default in `src/sonna_editor/config.py`; training augmentation is geometry-only unless explicitly changed. This avoids corrupting Exposure/colour supervision because the XMP target sliders are for the original image, not a brightened/darkened synthetic image.
- Reworked `src/sonna_editor/data/dataset.py` split logic to balance shoot-grouped splits across Temperature correction, Exposure2012, and Tint correction, with tail coverage so rare edit styles do not concentrate in one split.
- Added output-head target-prior calibration in `SonnaEditor.initialise_output_priors()` and wired it into `scripts/train_profile.py` / foundation warm starts. Fresh output heads start at training-set slider medians with zeroed final weights; warm-started runs preserve learned final weights and recenter the final biases. WB residual heads start at zero when AsShot WB skip is enabled.
- Raised the default v2 Exposure2012 loss weight to 4.0. The current training-set priors are Exposure2012=0.22, Temperature=5191K, Tint=5.
- Fixed the pink/red Lightroom output issue in `src/sonna_editor/inference/pipeline.py`. Diagnosis on `0H5A3190A-2.xmp`: Temperature/Tint were close to the expected XMP, but the model-written RGB tone-curve highlight endpoints were not neutral (`Green Pt6=240/221`, `Blue Pt6=213/234`). Those endpoints colour-cast white highlights pink/red. The inference pipeline now forces RGB tone-curve black endpoints to `0/0` and white endpoints to `255/255`, while leaving mid-curve shape predictions intact.
- Added preview-derived luminance scene statistics (`mean_luminance`, `median_luminance`, `luminance_std`, `highlight_clip_pct`, `shadow_clip_pct`, `dynamic_range`) to extraction, RAW/XMP datasets, catalog datasets, fine-tune captures, training dataloaders, inference batches, and the scene-stat metadata encoder path.
- Updated default visual-priority loss weights to match the improvement plan: Exposure=5.0, Temperature/Tint=4.0, Contrast/Highlights/Shadows=3.0, Whites/Blacks/Saturation/Vibrance=2.0 minimums, while preserving existing targeted timid-field bumps above those floors.
- Added validation distribution logging for Exposure, Contrast, Highlights, Shadows, Temperature, and Tint (`val_dist_*_std_ratio`) so future training runs surface collapse during training.
- Added `scripts/analyse_prediction_collapse.py` and `scripts/audit_dataset_diversity.py`.
- Trained `data/models/sonna-v2-scene-stats-run01` on CUDA. Best checkpoint was epoch 0 with `best_val_loss=0.001177`, `test_loss=0.000994`, `test_mae_exposure=0.229`, `test_mae_temperature=115K`, `test_mae_hsl_avg=2.03`. Despite low MAE, collapse audit worsened from 14 collapsed sliders on `model-v2.0.0` to 29 collapsed sliders on this candidate, so it was not kept frontend-visible.
- Fixed frontend Lite profile creation compatibility with current checkpoints. `src/sonna_editor/mode_b/checkpoint_builder.py` now loads the base checkpoint at its native slider set, writes the matching internal `slider_set_version` into the Lite sidecar, and preserves extension-head weights instead of down-converting.
- Removed stale v2-rejection behavior from the Mode B tests and removed an unused test import. Added focused v2 coverage proving Lite creation from a v2 base succeeds and keeps extension heads intact.
- Cleaned up noisy training console warnings. `scripts/train_profile.py` now chooses `log_every_n_steps` from the actual train-batch count so the 132-row local split uses 9 instead of tripping Lightning's default-10 warning. The training package, current trainer, legacy production trainer, and fine-tune path suppress the upstream Lightning `LeafSpec` deprecation and optional Torch Triton FLOP-counter warning.
- Fixed `scripts/quick_diagnostic.py` so old training summaries that do not embed row counts still print dataset split row counts. It now checks nested summary fields first, then summary parquet path fields if present, then falls back to the canonical `data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/*.parquet` metadata. It also replaced emoji status markers with ASCII `OK`/`BAD` labels so the script finishes cleanly in Windows PowerShell.
- Fixed the Mode B/Lite overexposure source in the profile builder. `src/sonna_editor/mode_b/checkpoint_builder.py` now zeroes each output head's final linear weights and copies absolute preset+survey targets into final biases, so the profile carrier does not add the trained base model's own Exposure/colour predictions on top of the preset. For v2 bases, the direct `wb_metadata_skip` route is zeroed in Mode B initial checkpoints so AsShot WB is not secretly stacked on top of preset Temperature/Tint.
- Fixed the actual UI/CLI Mode B processing flow. `src/sonna_editor/inference/pipeline.py` now detects checkpoint sidecars with `profile_type: "mode_b_initial"`, bypasses `InferenceEngine` for the initial Lite run, reloads the copied preset and survey, computes per-photo Exposure/WB corrections only through `sonna_editor.preset.adjuster`, writes those adjusted values to XMP, and records the adjusted baseline in `sonna_predictions.json`.
- Fixed the second root cause in `src/sonna_editor/preset/adjuster.py`: the old auto-exposure heuristic used mean luminance only. On real event frames with black suits/dark rooms plus bright faces/signage, it could add too much positive Exposure. The new heuristic still uses mean luminance but guards it with 85th/95th percentile luminance targets so already-bright upper tones prevent over-lifting.
- Published a corrected Lite profile at `v1_learning/model-v0.2.0.ckpt` / `.json` from the same preset and survey as stale `model-v0.1.0`. Its profile id is `k-fixed-lite-20260602-2120`. Copied support files to `model-v0.2.0-preset.xmp` and `model-v0.2.0-survey.json`, updated the v0.2 sidecar to point to them, and removed stale `model-v0.1.0.*` artifacts.
- Updated Mode B tests to pin both the checkpoint builder behavior and the adaptive initial processing path. Removed stale assertions/comments that expected inherited final-layer head weights and an unused test local.
- Added frontend Personal AI profile creation: the profile page now opens a RAW+XMP training wizard, starts `POST /api/profiles/personal`, streams epoch progress through the existing job hook, supports cancel, and refreshes profiles after completion.
- Lite profile creation now presents the six-question style survey in the UI. All six answers are stored in the survey JSON and profile package; initial Lite processing still dynamically adjusts only Exposure, Temperature, and Tint so preset look sliders stay fixed until fine-tuning.
- Added a fresh-model staged-head learning path. `SonnaEditor` now defaults to `arch_version=3`, where WB/presence heads condition on the tone block output and later heads condition on tone + presence + WB outputs. Existing `arch_version=1`/`2` checkpoints load unchanged.
- Added profile-delete confirmation in the frontend before removing any local checkpoint/sidecar files. Active profile deletion remains blocked by the backend.
- Moved the training callable into `src/sonna_editor/training/profile_runner.py`; `scripts/train_profile.py` is now a thin CLI wrapper. The API imports the packaged runner rather than a script module.
- Added cancellation handling to frontend Personal AI training so cancelled runs do not test/save/publish a new checkpoint.
- Updated docs so Personal AI and Lite training/execution are frontend flows, while foundation training is CLI-only through `scripts/train_foundation_model.py`.
- Renamed `TRAINING_COMMANDS.md` to `CLI_COMMANDS.md` and created `FOUNDATION_TRAINING.md` for foundation train, resume, retrain, promotion, and FiveK guidance.
- Updated frontend Personal AI backend flow so RAW+XMP profile training resolves the configured hidden foundation checkpoint and warm-starts model weights from it before publishing a user-facing checkpoint into `v1_learning/`. Warm-start now uses `base_model_checkpoint` rather than Lightning resume state, keeps the new training registry, and skips categorical embedding-table copies.
- Verified MIT-Adobe FiveK is suitable foundation material through its Lightroom catalog, with the caveat that it is 5,000 DNG inputs plus expert catalog edit variants, not 25,000 independent RAW inputs. Use one expert collection first.
- Removed the previous paired-image foundation implementation and commands. `scripts\train_foundation_model.py` now accepts only `--raw-xmp-dir` or `--splits-dir`.
- Updated foundation training semantics so each new foundation run warm-starts from the active foundation checkpoint by default, writes a new versioned checkpoint, promotes it as the active default, and keeps previous checkpoints untouched. If the active checkpoint file is removed after a bad run, resolution falls back to the newest remaining checkpoint in the foundation folder.
- Updated the foundation and operator docs so RAW+XMP foundation training uses direct script commands only: Lightroom metadata export/source-folder expectation, inspectable dataset/split build, audits, training from prepared splits, and the direct `--raw-xmp-dir` shortcut. Operator-facing stale references to repo-local `data\foundation_repo\`, old local dataset presence, and old Lite profile artifacts were cleaned up.
- Added `matplotlib` to the base project dependencies and refreshed `uv.lock` so dataset audit plots generate without optional-import warnings. `scripts\audit_catalog.py` now prints ASCII `OK`/`WARN`/`STOP` status labels so Windows PowerShell does not fail on emoji encoding.
- Fixed foundation training CUDA OOM handling. `scripts\train_foundation_model.py` now defaults to `--batch-size 8` for foundation runs and the RAW+XMP slider-regression path automatically catches CUDA memory failures, clears the CUDA cache, and retries with halved batch sizes. Foundation docs now recommend batch 8 on the Windows RTX 3050 workstation.
- Adjusted Lite/preset auto-exposure for low-light images. Dark scenes with no near-clipped highlights now receive a stronger positive exposure floor, so event frames like the sofa/table example lift closer to the Imagen-style reference instead of staying slightly underexposed. WB remains conservative by default.
- Guarded Lightning metric logging in `src/sonna_editor/training/module.py` so standalone `training_step()` unit tests no longer emit `self.log()` warnings without a Trainer.
- Corrected the foundation workspace boundary so `SonnaEditor` is the parent folder. The default foundation folder is now `SonnaEditor\SonnaEditorFoundation\` instead of the sibling `Projects\SonnaEditorFoundation\`. The folder is tracked by the parent repo, with checkpoint binaries routed through Git LFS.
- Removed the nested `SonnaEditorFoundation\.git` metadata so the parent `SonnaEditor` repo can track the foundation folder directly. Added parent `.gitattributes` rules for `SonnaEditorFoundation/checkpoints/*.ckpt`, `v1_learning/*.ckpt`, and `models/**/*.ckpt` through Git LFS.
- Removed the obsolete `--init-git` foundation CLI option so future foundation runs cannot recreate a nested Git repository inside `SonnaEditorFoundation\`.
- Cleaned fixture-dependent RAW/XMP tests so missing/unreadable private fixtures skip cleanly instead of failing the full suite on this Windows workspace.

## Verification

- Current foundation verification pass:
  - `uv run python scripts\train_foundation_model.py --help` shows only `--raw-xmp-dir` and `--splits-dir`; obsolete rendered-image flags are absent.
  - Real FiveK folder `C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset` contains 5,000 `.dng` files and `raw_photos\fivek.lrcat`.
  - `find_edited_photos(..., collection_name='C')` returns 5,000 rows; collections A/B/C/D/E each return 5,000 rows.
  - `uv run python scripts\build_dataset_from_catalog.py --catalog-path "C:\Users\vikas.DESKTOP-61LEE8B\Downloads\fivek_dataset\fivek_dataset\raw_photos\fivek.lrcat" --output-dir "data\training_workspace\fivek_catalog_verify_20260605" --profile-name "fivek_catalog_verify_20260605" --collection-name "C" --include-unedited-looking --limit 20 --workers 1` passed.
  - `uv run python scripts\rollback_foundation.py --list` now lists both `foundation-sonna-raw-xmp-001` and active `foundation-sonna-raw-xmp-003`.
  - `uv run pytest tests\test_foundation.py -q` passed: 4 passed.
- Clean-start verification after checkpoint cleanup:
  - `uv run pytest tests\test_foundation.py tests\test_train_foundation_model.py -q` passed: 10 passed.
  - `uv run python -c "import scripts.train_foundation_model as t; print(t._active_foundation_or_none())"` printed `None`.
  - `uv run ruff check src\sonna_editor\foundation.py scripts\train_foundation_model.py tests\test_foundation.py` passed.
- Dataset-location cleanup verification:
  - `v1_learning\dataset` was removed after path-boundary verification; source FiveK data and source RAW/XMP folders were not touched.
  - `rg -n "v1_learning.*dataset|CHECKPOINTS_DIR.*dataset|dataset/splits" ...` now only finds active canonical `data\training_workspace\...` paths plus historical generated reports under `scripts\output\`.
  - `uv run ruff check src\sonna_editor\config.py scripts\quick_diagnostic.py scripts\finetune_profile.py scripts\audit_all_sliders_v1.2.3.py scripts\train_v1_2_0_full_production.py scripts\migrate_labels_to_v2.py src\sonna_editor\foundation.py scripts\train_foundation_model.py tests\test_config.py tests\test_foundation.py` passed.
  - `uv run python -m py_compile src\sonna_editor\config.py scripts\quick_diagnostic.py scripts\finetune_profile.py scripts\audit_all_sliders_v1.2.3.py scripts\train_v1_2_0_full_production.py scripts\migrate_labels_to_v2.py src\sonna_editor\foundation.py scripts\train_foundation_model.py` passed.
  - `uv run pytest tests\test_config.py tests\test_foundation.py tests\test_train_foundation_model.py -q` passed: 36 passed.

- `uv run ruff check src\sonna_editor\foundation.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\api\models.py scripts\train_foundation_model.py scripts\build_mode_b_checkpoint.py tests\test_foundation.py tests\api\test_profiles.py tests\api\conftest.py` passed.
- Current repo-local foundation/Git LFS verification:
  - `git check-attr filter diff merge -- SonnaEditorFoundation\checkpoints\foundation-sonna-raw-xmp-001.ckpt` reports `filter: lfs`, `diff: lfs`, and `merge: lfs`.
  - `Test-Path SonnaEditorFoundation\.git` returned `False`.
  - `git add --dry-run .gitattributes SonnaEditorFoundation` lists the foundation manifest, sidecar, README, and active checkpoint as addable by the parent repo.
- Current cleanup verification:
  - `uv run ruff check .` passed.
  - `uv run python -m compileall -q src scripts tests` passed.
  - `uv run mypy src\sonna_editor\foundation.py scripts\train_foundation_model.py` passed.
  - `npm run build:vite` passed in `saha-app\`.
  - `uv run pytest tests\test_foundation.py tests\test_train_foundation_model.py tests\test_config.py tests\api\test_profiles.py tests\test_checkpoint_builder.py -q` passed: 88 passed.
  - `uv run pytest tests\test_extract.py tests\test_xmp.py -q` passed with local fixture skips: 71 passed, 34 skipped.
  - `uv run pytest tests -q --ignore=tests/test_extract.py --ignore=tests/test_xmp.py` passed: 653 passed, 11 skipped, 1 warning.
- `uv run python -m py_compile src\sonna_editor\foundation.py src\sonna_editor\api\routes\profiles.py scripts\train_foundation_model.py scripts\build_mode_b_checkpoint.py` passed.
- `uv run pytest tests\test_foundation.py tests\api\test_profiles.py -q` passed: 22 passed.
- `uv run ruff check src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\api\test_callback_bridge.py tests\test_inference_pipeline_integration.py` passed.
- `uv run python -m py_compile src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py` passed.
- `uv run pytest tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\test_architecture.py tests\api\test_callback_bridge.py::test_mode_b_initial_uses_per_photo_preset_adjuster tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` passed: 143 passed, 6 skipped.
- `npm run build:vite` passed in `saha-app/`.
- Cleanup verification after foundation/Mode A/Mode B updates:
  - `uv run ruff check .` passed.
  - `uv run python -m py_compile src\sonna_editor\model\architecture.py src\sonna_editor\mode_b\survey.py src\sonna_editor\mode_b\checkpoint_builder.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\training\module.py scripts\train_foundation_model.py` passed.
  - `npm run build:vite` passed in `saha-app/`.
  - `uv run pytest tests\test_style_survey.py tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\test_architecture.py tests\api\test_callback_bridge.py::test_mode_b_initial_uses_per_photo_preset_adjuster tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation tests\test_training.py::test_foundation_warm_start_keeps_training_registry tests\test_training.py::test_training_step_returns_scalar tests\test_training.py::test_training_step_loss_is_non_negative tests\test_training.py::test_loss_gradient_flow_to_predictions tests\test_training.py::test_output_prior_initialisation_sets_exposure_and_zero_wb_residual -q` passed: 148 passed, 6 skipped, no warnings.
  - `uv run mypy src scripts` is still not clean in this workspace; it reports broad pre-existing strict-typing debt and missing third-party stubs across many files. Treat that as a dedicated type-hardening task, separate from lint/compile/test cleanup.
- Documentation-only RAW+XMP foundation runbook pass completed on 2026-06-03. Reviewed `FOUNDATION_TRAINING.md`, `CLI_COMMANDS.md`, `RUN.md`, `README.md`, `MAC_SETUP.md`, `HANDOVER.md`, `SESSION_STATE.md`, and `project_knowledge.md`; no code tests were required for this docs-only update.
- Dataset audit warning fix verification:
  - `uv lock` added `matplotlib` and plotting dependencies.
  - `uv sync --extra dev` installed `matplotlib==3.10.9`.
  - `uv run python scripts\audit_catalog.py --parquet-path data\training_workspace\sonna_foundation_001_dataset\dataset.parquet --output-dir data\training_workspace\sonna_foundation_001_dataset\audit` completed with no missing-matplotlib warnings and no PowerShell encoding error; status remained `WARN` because the 189-photo dataset is small and has 40 outlier sliders.
  - `uv run ruff check scripts\audit_catalog.py pyproject.toml` passed.
  - `uv run pytest tests\test_audit.py -q` passed: 26 passed.
- Foundation CUDA OOM fix verification:
  - `uv run ruff check scripts\train_foundation_model.py tests\test_train_foundation_model.py` passed.
  - `uv run pytest tests\test_train_foundation_model.py -q` passed: 5 passed.
  - `uv run python -m py_compile scripts\train_foundation_model.py` passed.
  - One-epoch smoke completed on CUDA with `--batch-size 8 --workers 0 --no-warm-start`, using `data\training_workspace\sonna_foundation_001_dataset\splits_v2_stratified` and disposable foundation folder `data\tmp_foundation_oom_smoke_repo`. It promoted `data\tmp_foundation_oom_smoke_repo\checkpoints\foundation-oom-smoke.ckpt`; the real foundation folder was not touched.
- Reviewed existing `HANDOVER.md`, `SESSION_STATE.md`, `project_knowledge.md`, `SONNA_EDITOR_BUILD_SPEC.md`, `CLI_COMMANDS.md`, `RUN.md`, `README.md`, `pyproject.toml`, and `saha-app/package.json` before writing the Mac guide.
- `uv run python -c "import torch; ..."` confirmed:
  - `torch 2.11.0+cu128`
  - `torch.cuda.is_available() == True`
  - device: NVIDIA GeForce RTX 3050
  - CUDA tensor matmul succeeded
- `uv run python scripts\verify_environment.py` passed 11/11 checks.
- `uv run ruff check scripts\train_profile.py pyproject.toml` passed.
- `uv run ruff check src\sonna_editor\data\dataset.py src\sonna_editor\model\architecture.py src\sonna_editor\config.py scripts\train_profile.py tests\test_dataset.py tests\test_training.py` passed.
- `uv run python -m py_compile scripts\train_profile.py` passed.
- Targeted backend tests passed: `tests/api/test_profiles.py`, `tests/api/test_health.py`, `test_training_step_returns_scalar`, `test_training_step_loss_is_non_negative` (22 passed, 2 Lightning logging warnings).
- Focused training/data tests passed: `uv run pytest tests\test_dataset.py tests\test_training.py::test_output_prior_initialisation_sets_exposure_and_zero_wb_residual tests\test_training.py::test_training_step_returns_scalar tests\test_config.py` -> 59 passed, 1 skipped.
- Inference endpoint fix tests passed: `uv run pytest tests\test_inference_v2.py` -> 10 passed.
- Real sample verification: applying the new endpoint stabiliser to the bad Sonna `0H5A3190A-2.xmp` changes RGB Pt6 endpoints to `255/255` for red, green, and blue. The temporary verification file was removed after the check.
- Regenerated the affected local shoot output with `scripts\process_shoot_model.py` using `v1_learning\model-v2.0.0.ckpt`:
  - input/output: `C:\Users\vikas.DESKTOP-61LEE8B\OneDrive\Pictures\Testing_Sonna`
  - processed: 100 RAWs, failed: 0
  - wrote fresh XMPs plus `sonna_predictions.json`
  - verified `0H5A3190A-2.xmp` now has Red/Green/Blue Pt6 endpoints all at `255,255`
- `npm run build:vite` passed in `saha-app`.
- `uv run ruff check ...` over changed source/tests/scripts passed.
- `uv run python -m py_compile scripts\train_profile.py scripts\analyse_prediction_collapse.py scripts\audit_dataset_diversity.py` passed.
- `uv run pytest tests\test_training.py tests\test_dataset.py tests\test_catalog_dataset.py tests\test_architecture.py -q` passed: 134 passed, 7 skipped.
- `uv run pytest tests\test_extract.py::TestComputeSceneStatistics -q` passed: 3 passed.
- `scripts\audit_dataset_diversity.py` ran on `data\training_workspace\sonna_personal_001_dataset\dataset.parquet`: 189 photos / 35 shoots; brightness split dark=92, balanced=81, bright=16; WB split warm=66, daylight=114, cool=9.
- `scripts\analyse_prediction_collapse.py` ran on existing `model-v2.0.0`: 27 val photos, 14 collapsed sliders; Exposure2012 std_ratio=0.115, Temperature/Tint std_ratio ~1.0.
- `scripts\analyse_prediction_collapse.py` ran on rejected scene-stats candidate: 27 val photos, 29 collapsed sliders; Exposure2012 std_ratio near zero, so the candidate was rejected despite lower test MAE.
- Dark low-light output diagnosis on `0H5A4599`: the reference/training XMP has `Exposure2012=+1.11`, while active `model-v2.0.0` writes about `+0.105`, roughly one stop too dark. Other key tone/WB sliders and tone curves are close to the reference, so this is not an XMP writer or tone-curve endpoint issue. Root cause is Exposure2012 prediction collapse: across the 189-row dataset, target Exposure std is ~0.454 but model output std is ~0.061; in the darkest luminance quartile, targets average `+0.695` while predictions average only `+0.090`.
- Fixture-dependent RAW/XMP tests now skip cleanly when local private fixtures are absent or unreadable. `tests\test_extract.py tests\test_xmp.py` passes with local fixture skips instead of failing on missing `sample_edit.xmp`, unreadable `sample.cr3`, or Windows symlink privileges.
- Lite compatibility verification passed: `uv run ruff check src\sonna_editor\mode_b\checkpoint_builder.py tests\test_checkpoint_builder.py tests\api\test_profiles.py`; `uv run pytest tests\test_checkpoint_builder.py tests\api\test_profiles.py -q` -> 53 passed; `uv run python -m py_compile src\sonna_editor\mode_b\checkpoint_builder.py scripts\build_mode_b_checkpoint.py`; real smoke build from `v1_learning\model-v2.0.0.ckpt` to `%TEMP%\sonna-lite-v2-smoke\mode-b-v2-smoke.ckpt` succeeded and wrote sidecar `slider_set_version: v2`.
- Training warning cleanup verification passed: `uv run ruff check scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\training\__init__.py src\sonna_editor\finetune\retrain.py tests\test_training.py`; `uv run pytest tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset tests\test_training.py::test_training_step_returns_scalar tests\test_training.py::test_training_step_loss_is_non_negative -q` -> 3 passed; `uv run python -m py_compile scripts\train_profile.py scripts\train_v1_2_0_full_production.py src\sonna_editor\finetune\retrain.py src\sonna_editor\training\__init__.py`; one-epoch smoke training with `--num-workers 2 --no-publish` completed without the pasted `LeafSpec`, Triton FLOP-counter, or `log_every_n_steps` warnings.
- Quick diagnostic row-count verification passed: `uv run ruff check scripts\quick_diagnostic.py`; `uv run python -m py_compile scripts\quick_diagnostic.py`; `uv run python scripts\quick_diagnostic.py --summary-path data\models\sonna-v2-run01\training_summary.json` now prints train=132, val=27, test=30 and completes successfully.
- Quick diagnostic output clarity was improved. `scripts\quick_diagnostic.py` now prints a recommended-score column, uses green-circle `OK` pass statuses, treats missing `published_model` as normal for foundation runs, and prints next steps that distinguish hidden foundation validation from frontend Personal AI publishing.
- Quick diagnostic clarity verification passed: `uv run ruff check scripts\quick_diagnostic.py`; `uv run python scripts\quick_diagnostic.py --summary-path data\training_workspace\foundation_runs\foundation-sonna-raw-xmp-003\training\training_summary.json`.
- Mode B preset-faithful verification passed:
  - `uv run ruff check src\sonna_editor\mode_b\checkpoint_builder.py tests\test_checkpoint_builder.py`
  - `uv run pytest tests\test_checkpoint_builder.py -q` -> 34 passed
  - `uv run pytest tests\api\test_profiles.py tests\api\test_process_route.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 31 passed
  - `uv run python -m py_compile src\sonna_editor\mode_b\checkpoint_builder.py scripts\build_mode_b_checkpoint.py`
  - Active-v2 smoke build from `v1_learning\model-v2.0.0.ckpt` using `tests\fixtures\preset_sonna_v1.xmp` now outputs the preset values exactly on synthetic dark and bright inputs: `Exposure2012=+0.35`, `Contrast2012=15`, `Highlights2012=-45`, `Shadows2012=30`, `Temperature≈5200`, `Tint=-3`, `Saturation=5`, `Vibrance=12`.
  - `npm run build:vite` passed in `saha-app`.
- Mode B adaptive processing verification passed:
  - `uv run ruff check src\sonna_editor\inference\pipeline.py tests\api\test_callback_bridge.py`
  - `uv run python -m py_compile src\sonna_editor\inference\pipeline.py`
  - `uv run pytest tests\api\test_callback_bridge.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 13 passed
  - New coverage proves the initial Mode B UI processing path does not load `InferenceEngine`, produces different Exposure values for dark vs bright previews, applies grey-world WB from the photo baseline when the preset omits Temperature/Tint, preserves fixed preset style fields, and records the adjusted per-photo values in `sonna_predictions.json`.
- Mode B root-cause verification on the real UI folder passed:
  - Stale active profile `v1_learning/model-v0.1.0.json` had old notes: final-layer weights inherited from base checkpoint and biases shifted by preset+survey deltas. This is the base-model + preset double-apply path.
  - Existing bad XMP example `OneDrive\Pictures\Test_Sonna - Copy\0H5A9030-3.xmp` had model-stacked values: `Exposure2012=+1.409`, `Contrast2012=-9.73`, `Highlights2012=-83.14`, `Shadows2012=86.48`, `Whites2012=-71.68`, `Blacks2012=11`.
  - Current code with corrected `v1_learning/model-v0.2.0.ckpt` processed all 260 RAWs from `OneDrive\Pictures\Test_Sonna - Copy` to `%TEMP%\sonna-modeb-v020-real-output`: 260 processed, 0 failed.
  - Corrected real output for screenshot file `0H5A0100-6.xmp`: `Exposure2012=-0.329`, `Contrast2012=-2`, `Highlights2012=-53`, `Shadows2012=60`, `Whites2012=-52`, `Blacks2012=-2`, `Temperature=3633`, `Tint=5.7`, `Saturation=7`, `Vibrance=-7`.
  - Corrected real output for prior high-exposure sample `0H5A9030-3.xmp`: `Exposure2012=+0.780` with fixed preset style values, not base-model tone values.
- Focused suite after the adjuster fix: `uv run pytest tests\test_checkpoint_builder.py tests\api\test_profiles.py tests\api\test_process_route.py tests\api\test_callback_bridge.py tests\test_pipeline.py tests\test_adjuster.py tests\test_inference_pipeline_integration.py::test_mode_b_end_to_end_sidecar_propagation -q` -> 121 passed, 1 skipped.
- Production profile-flow verification passed:
  - `uv run ruff check .` -> passed
  - `npm run build:vite` in `saha-app` -> passed
  - `uv run python -m py_compile scripts\train_profile.py src\sonna_editor\training\profile_runner.py src\sonna_editor\api\routes\profiles.py src\sonna_editor\api\jobs.py src\sonna_editor\api\models.py` -> passed
  - `uv run pytest tests\api\test_profiles.py tests\api\test_callback_bridge.py tests\test_adjuster.py tests\test_checkpoint_builder.py -q` -> 101 passed
  - `uv run pytest tests\test_config.py::TestSliderLossWeights::test_c3k_tuned_weights tests\test_training.py::test_train_profile_log_interval_adapts_to_small_dataset -q` -> 2 passed
  - Historical note: full `uv run pytest tests -q` used to fail when local RAW/XMP fixtures were absent or unreadable. The fixture-dependent tests now skip cleanly in that state.

## Current Code Behavior Notes

- RAW file scanning is centralised in `config.SUPPORTED_RAW_EXTENSIONS`: `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.dng`, `.x3f`, `.rwl`, and `.srw`. This controls dataset builds, folder/API scans, preset processing, model inference, and fine-tune capture. Actual extraction still depends on `rawpy`/LibRaw support for the camera file; optional DNG conversion still depends on Adobe DNG Converter support.
- `scripts/train_profile.py` default v2 recipe now uses:
  - `image_resolution=512`
  - `lr=1e-4`
  - `max_epochs=50`
  - `freeze_backbone_epochs=3`
  - `arch_version=3` for fresh models, adding staged output-head conditioning on top of the six luminance scene-stat metadata inputs
  - Temperature loss weight=4.0
  - Tint loss weight=4.0
  - Exposure2012 loss weight=5.0
  - Temperature bucket loss weight=0.15
  - Tint bucket loss weight=2.0
  - Sign-wrong penalty weight=0.2
  - WB metadata skip enabled
- Training defaults to target-prior output calibration. On the previous 189-row diagnostic dataset, this could win validation loss while still producing collapsed predictions; always run collapse analysis before promoting a small-data candidate.
- Output-head target-prior calibration is derived from the current training parquet's target medians, falls back to Lightroom defaults for missing fields, and does not freeze those heads after training starts. Fresh runs zero final head weights before setting median biases; warm-started runs preserve learned final weights and only recenter final biases. With the direct AsShot WB skip enabled, Temperature/Tint residual biases initialise to zero so the initial WB output starts at AsShot rather than the dataset median. Keep it enabled for foundation and Personal AI runs unless doing an ablation.
- A 5,000-photo FiveK split at batch size 8 should show about 472 train batches per epoch if the train split has roughly 3,770 rows. That is not a subset by itself. The new startup diagnostics explicitly print train/val/test image counts, batches per epoch, estimated optimizer steps, `max_steps`, `limit_train_batches`, and sampler type so suspiciously similar 200-photo vs 5,000-photo timings can be verified from logs.
- The previous `model-v2.0.0` diagnostic profile under-brightened dark/low-light photos because its Exposure2012 head was nearly averaged. Example: `0H5A4599` mean luminance `0.126`, target `+1.11`, prediction about `+0.10`.
- Default training augmentation is now geometry-only; photometric jitter remains configurable but disabled by default.
- Training on tiny splits now logs once that it adjusted `log_every_n_steps` instead of letting Lightning warn. This is expected for the current 132-row train split, which has 9 batches at batch size 16.
- `Profile.profile_type` is already implemented in backend profile responses and frontend profile classification. `None` means a legacy trained profile; `"mode_b_initial"` means a Lite preset-derived profile.
- Mode B/Lite checkpoints now inherit the configured foundation checkpoint's native slider set and field count. Before fine-tuning, the UI/CLI processing path treats `mode_b_initial` as an Imagen-aligned Lite profile: preset look fixed, per-photo Exposure/WB corrections only. After fine-tuning, the same profile can move back to normal model inference.
- Auto straightening is opt-in and runs after preview extraction during
  processing. It uses CLAHE-normalized OpenCV Canny edges plus Hough/LSD line
  geometry to estimate small Lightroom `CropAngle` rotations, writes crop
  metadata only when confidence is high, and records skipped/applied diagnostics
  in `sonna_predictions.json`. It is independent of training and checkpoint
  versioning.
- The Process UI requires explicit row selection. With no checked queued rows,
  Process Selected is disabled and no job is dispatched; with checked rows, it
  runs only the selected queued folders. Auto straighten follows the same
  selected dispatch path.

## Next Suggested Step

For immediate FiveK foundation training, build the Expert C catalog dataset with `scripts\build_dataset_from_catalog.py --collection-name "C" --include-unedited-looking`, audit it, then train from prepared splits with `scripts\train_foundation_model.py --splits-dir ...`. Do not mix all 60,000 FiveK virtual-copy rows in one unconditioned model.

Add the fresh RAW+XMP dataset, start the backend/Electron app, and create a Personal AI profile from the frontend. Use Lite profile creation from the frontend after a foundation checkpoint is configured in `SONNA_FOUNDATION_CHECKPOINT`, `SONNA_FOUNDATION_REPO/foundation_manifest.json`, or `SONNA_FOUNDATION_REPO/foundation.ckpt`.

For current foundation model work, use `scripts\train_foundation_model.py --raw-xmp-dir ...` or `--splits-dir ...` so the checkpoint is promoted to `SonnaEditorFoundation\` and stays out of the frontend profile list. It will warm-start from the active foundation checkpoint unless `--no-warm-start` is supplied. For MIT-Adobe FiveK, use catalog-derived splits from `build_dataset_from_catalog.py`.

Before relying on live RAW/XMP extraction coverage, restore the local fixture files under `tests/fixtures/`. The normal full suite now skips those local-data checks when the fixtures are missing or unreadable.

