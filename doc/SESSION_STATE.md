# Session State - Sonna Editor

**Status:** Active | Cross-platform dev environment verified, code optimizations & weakness fixes complete, Mac zero-touch launcher verified, and client distribution packaging built. Full test suite passing.

> [!IMPORTANT]
> **Chronological Change Tracking:**
> For the complete chronological task-by-task, session-by-session change history, implementation notes, and test verification logs, refer directly to [session.md](../session.md) at the project root.
> Per repository rules, every code and documentation modification is recorded in `session.md`.

---

## Current Workspace & Lineage

- **Repository Root:** `.` (cross-platform, git-tracked)
- **Primary Change Tracker:** [session.md](../session.md)
- **Active Foundation Checkpoint:** `foundation-sonna-raw-xmp-004-visual` in `SonnaEditorFoundation/checkpoints/`
  - Sidecar properties: `profile_type=mode_a_trained`, `slider_set_version=v2`, `arch_version=3`, `resolution=512`, `train_rows=5021`, `val_loss=0.02533668652176857`.
  - Managed via Git LFS through `.gitattributes`.
- **Active Lite Profile Checkpoint:** `v1_learning/model-v0.1.0.ckpt`
  - Sidecar properties: `profile_type=mode_b_initial`, anchored to `foundation-sonna-raw-xmp-004-visual`.
  - Mode B processing applies preset baseline look + style survey calibration (Exposure, Temp, Tint) + adaptive per-photo scene Exposure and camera "As Shot" WB anchoring. Full 9-step survey efficacy verified end-to-end.
- **Client Distribution Bundle:** `dist/saha-mac-client-bundle.zip` (106.2 MB)
  - Packaged via `scripts/package_for_client.py` for client testing on macOS Apple Silicon / Intel.
  - Companion testing guide: [doc/MAC_CLIENT_TESTING.md](MAC_CLIENT_TESTING.md).

---

## Environment & Tooling

- **Python Runtime:** Python 3.11.x (exact-pinned in `pyproject.toml` and `.python-version`)
- **Package Manager:** `uv` (fast, cross-platform dependency resolution)
- **ML Framework:** PyTorch 2.11.0 with TorchVision 0.26.0 (CUDA 12.8 on Windows/Linux x86_64, MPS on Apple Silicon, CPU fallback)
- **Frontend Runtime:** Node.js LTS (v22.11.0), Electron desktop runner, Vite bundler
- **Image Processing:** OpenCV (`opencv-python-headless`), rawpy / LibRaw, Pillow, lxml
- **Environment Verifier:** `uv run python scripts/verify_environment.py`

---

## Key Implemented Behaviors & Fixes

1. **Slider Defaults Consolidation:**
   - Single canonical source of truth for 147 default slider values in `config.DEFAULT_SLIDER_VALUES`.
   - `xmp.py` re-exports `LR_DEFAULTS = DEFAULT_SLIDER_VALUES` and `config.SLIDER_DEFAULTS` slices from it, eliminating drift.

2. **Color Wheel Red Cast Protection:**
   - Lightroom Classic defaults missing hue values to 0.0° (pure red) whenever saturation is written.
   - Cleared `default_skip_fields` in profile metadata.
   - Added `COLOR_WHEEL_PAIRS` guard in `src/sonna_editor/data/xmp.py` and `src/sonna_editor/inference/pipeline.py`: positive color wheel saturations are omitted whenever their corresponding hue is missing, eliminating rogue red casts.

3. **Auto-Straightening:**
   - High-confidence Hough and LSD line analysis in `src/sonna_editor/inference/straighten.py`.
   - Conservative thresholds (`_MIN_APPLY_ANGLE = 0.25`, `_MIN_CONFIDENCE = 0.50`, `_MIN_HORIZON_CONFIDENCE = 0.45`, `_MIN_LINE_COUNT = 2`, `_MIN_LINE_LENGTH_PX = 36.0`).
   - Ambiguous event backdrop guard skips large horizon corrections when vertical evidence is comparable to horizontal (`V >= H * 0.75`).
   - Writes Lightroom-native calculated crop bounds (`CropAngle`, `CropTop`, `CropLeft`, `CropBottom`, `CropRight`, `HasCrop=True`, `AlreadyApplied=False`).

4. **White Balance & Temperature:**
   - `_MODE_B_ADJUSTMENT_OPTIONS["auto_white_balance"] = False` by default.
   - Presets without explicit WB keep `Temperature` and `Tint` as `None`, so Lightroom preserves the camera's native **"As Shot"** white balance.

5. **Adaptive Exposure:**
   - `net_delta = target_delta - base_exp` prevents double-bumping presets that carry base exposure.
   - Highlight clipping protection calibrated at `p98` and `p99.5` with smoothed low-light floor.

6. **Inference Pipeline Memory Throttling:**
   - In `src/sonna_editor/inference/pipeline.py`, bounded sliding-window execution prevents memory spikes when extracting previews across large shoots (5,000+ photos).

7. **Zero-Touch macOS Launcher:**
   - `DOUBLE_CLICK_MAC_START_SAHA.command` automatically discovers Apple Silicon Homebrew (`/opt/homebrew/bin`), auto-installs `uv`, and auto-bootstraps portable Node.js LTS for `darwin-arm64` if needed.
   - Electron main process augments `process.env.PATH` to ensure Python and `uv` work seamlessly when opened from Finder.

8. **Imagen AI UI Realignment & Burst Smoothing:**
   - Redesigned frontend UI matching Imagen AI reference screenshots: dedicated Home welcome & 3-step onboarding cards, Lite vs Personal profile qualification flow, 9-genre selector, multi-step project creation modal with AI tools flyout, full-width profile table with 3-dot context menu, and profile baseline tuning modal.
   - Clean top window chrome with serif logo, `↑ Uploader` pill, and theme toggle (unnecessary notification bell, help, info `?`, and avatar pills removed per studio production spec).
   - Realigned the left sidebar into 78px width with 64x56px rounded button tiles (Home, AI Profiles, Projects, Social Spotlight with notification red dot badge, Profile Shop, and Inspector).
   - Preserves Sonna dark warm ochre/charcoal palette (`#16140F`, `#1F1D17`, `#D4A359`) with zero ad banners, purchasing prompts, or subscription paywalls.
   - `smooth_burst_predictions()` stabilizes exposure and temperature across burst sequences (frames <= 4.0s apart with matching ISO) via a 70/30 current-to-median blend.
   - 4-tier ISO noise reduction (subtle, moderate, strong, stronger) in `_apply_denoise_if_needed()`.

9. **Catalog Ingestion, Review in Adobe, & Uploader State Sync:**
   - Multi-step catalog wizard with folder breakdown, metadata filtering, and genre selection matching reference. On "Choose a source", footer is cleanly suppressed; on "Choose photos", initial selection starts at 0 with gated "Next" button until folders are checked.

10. **Profile Uniqueness & Registry Truth:**
    - Strict unique profile name enforcement across backend endpoints (`create_lite_profile` and `create_personal_profile`) and frontend wizards (`LiteProfileWizard` and `PersonalProfileWizard`). Case-insensitive and trimmed names prevent duplicate profiles.
    - Profiles on disk returned by `GET /api/profiles` represent the single authoritative source of truth and are never filtered by client-side suppression lists. Local draft deletion is kept isolated to local mock drafts. Newly created profiles automatically activate and sync with project creation workflows.
   - Projects list/table view with real-time status transitions (`• Uploading photos · 0%` -> `• Editing` -> `• Ready for review` -> `● Fine-tune profile`).
   - "Review edits ▾" dropdown offering `Review in Adobe` (triggers XMP metadata download wizard) and `Review in Saha` (in-app photo inspection).
   - "Download edits" modal with folder location picker, download options (`Edits and ratings` / `Edits only`), and export metadata progress card with `Open folder` and `Train profile` transition.
   - Global `Manage uploads` popover and `ProjectDetailView` with photo grid editing badges.
   - Process lifecycle self-healing: proactive cleanup of stale zombie port bindings and synchronous tree termination on Windows prevents `backend failed to start` errors upon double-clicking or restarting.

10. **Lite Profile 9-Step Visual Style Survey Parity:**
   - Full 9-step benchmark visual style survey across 3 lighting archetypes (Backlit outdoor by water, Dim indoor shadow, and Direct sunlight with saturated foliage).
   - 45 original uncompressed 960x1440 and 1440x961 master WebP image assets extracted directly from Imagen's application bundle into `saha-app/public/survey/` and bundled into distribution assets (eliminating all blurriness and visual artifacts).
   - Captures user's aesthetic preferences for Exposure, Temperature, and Tint across all 3 scenarios, averaging them to set baseline profile offsets (`exposure_offset`, `temperature_offset`, `tint_offset`) on the Mode B carrier checkpoint.
   - Visual indicator dashes (9 dashes) and signature lime-yellow (`#D4F49C`) selection outline with smooth 250ms auto-advance and `Build profile` CTA on final step.

11. **Production Codebase Hygiene & Dead Code Purge:**
   - Purged dead frontend components (`genre-grid.jsx`) and unused hooks (`useCaptures.js`, `useRecentFolders.js`).
   - Stripped duplicate image directories (`saha-app/src/public/`) and redundant 45 `.jpg` exports, preserving only the 45 master `.webp` assets.
   - Removed empty legacy package `src/sonna_editor/profiles/` and stale local server PID artifacts (`.saha/jobs/.serve.pid`).
   - Cleaned distribution package: `dist/saha-v1.50.zip` down to 186 essential files (saved 139 files and ~34 MB).

---

## Master Documentation Map

| Document | Purpose |
|---|---|
| [session.md](../session.md) | **Live Master Change Tracker**: all chronological tasks, refactoring, and test outputs |
| [doc/HANDOVER.md](HANDOVER.md) | Comprehensive project handover, architecture, and background rationale |
| [doc/SESSION_STATE.md](SESSION_STATE.md) | High-level architecture status, model lineage, and behavior reference |
| [doc/RUN.md](RUN.md) | Developer runbook and setup guide for macOS, Windows, and Linux |
| [doc/MAC_CLIENT_TESTING.md](MAC_CLIENT_TESTING.md) | Targeted macOS testing guide for client review and Lightroom plugin testing |
| [doc/CLI_COMMANDS.md](CLI_COMMANDS.md) | Full CLI reference for datasets, training, inference, and diagnostics |
| [doc/FOUNDATION_TRAINING.md](FOUNDATION_TRAINING.md) | Foundation dataset prep, training, and promotion runbook |
| [doc/project_knowledge.md](project_knowledge.md) | Source package map, module responsibilities, and runtime details |
| [doc/SONNA_EDITOR_BUILD_SPEC.md](SONNA_EDITOR_BUILD_SPEC.md) | Task-by-task build specification |
| [AGENTS.md](../AGENTS.md) | Core AI agent operational rules, standards, and change recording flow |
