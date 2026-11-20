# Session Log - Sonna Editor

**Created:** 2026-11-11 local time
**Session Objective:** Move documentation into `doc/`, clean up dead/stale code and empty folders, resolve warnings, and perform end-to-end code optimizations across backend and frontend.

---

## Task Roadmap

1. [x] **Task 1: Move Documentation & Initialize Session Log**
   - Moved root markdown docs (`CLI_COMMANDS.md`, `FOUNDATION_TRAINING.md`, `HANDOVER.md`, `MAC_SETUP.md`, `RUN.md`, `SESSION_STATE.md`, `SONNA_EDITOR_BUILD_SPEC.md`, `project_knowledge.md`) into `doc/`.
   - Updated `AGENTS.md` and root `README.md` to reference `doc/`.
   - Created `session.md` to track changes continuously.

2. [x] **Task 2: Eliminate Stale Code, Dead Variables & Empty Folders**
   - Removed unused legacy placeholder package `src/sonna_editor/ui/` (`widgets`, `windows`, `workers`, `__init__.py`).
   - Removed empty directory `src/sonna_editor/Notes/`.
   - Cleaned up unused variables in `scripts/finetune_profile.py` (`v_str`) and `src/sonna_editor/mode_b/checkpoint_builder.py` (`end`).
   - Updated `doc/project_knowledge.md` to remove references to the deleted legacy UI package.

3. [x] **Task 3: Resolve Errors & Warnings**
   - Fixed PyTorch scalar conversion warning in `tests/test_losses.py` (`float(components["total"].detach())`).
   - Verified clean zero-warning test pass across `tests/test_losses.py` and fast regression suites.

4. [x] **Task 4: Deep Code Optimization Across Backend & Frontend**
   - Optimized `src/sonna_editor/preset/adjuster.py`: precomputed RGB ndarray and Rec. 709 luminance in `compute_adjustment`, reusing them across `_exposure_delta`, `_grey_world_wb`, `_neutral_reference_rgb`, and `_clip_ratio_at_ends`, eliminating multiple redundant PIL conversions per photo.
   - Optimized `src/sonna_editor/inference/straighten.py`: streamlined `estimate_straighten_angle` and `_analysis_image` to directly consume `uint8` image arrays without unnecessary float32 division and round-trip clipping.
   - Optimized `src/sonna_editor/data/extract.py`: precompiled `_XPACKET_RE` regex at top level and vectorized `compute_scene_statistics` percentile calculation (`np.percentile(luminance, [5, 95])`) into a single pass.
   - Built and verified frontend with Vite (`vite build`) producing clean production bundle in 510ms.

5. [x] **Task 5: End-to-End Test & Build Verification**
   - Full pytest suite: 788 passed, 45 skipped in 211s with 0 failures and 0 warnings.
   - Ruff linting: `uv run ruff check .` passed with 0 errors.
   - Bytecode compilation: `uv run python -m compileall -q src scripts tests` passed cleanly.
   - Vite frontend build: `npm run build:vite` passed in 510ms.

6. [x] **Task 6: Resolve Terminal Typing Issues & IDE Problems**
   - Fixed `asyncio.Queue[Any]` typing in `src/sonna_editor/api/callbacks.py:50`, `src/sonna_editor/api/jobs.py:97`, and `src/sonna_editor/api/routes/process.py:207`.
   - Fixed dictionary invariance type error in `src/sonna_editor/api/jobs.py:110` (`common: dict[str, Any] = { ... }`).
   - Fixed missing generic arguments in `src/sonna_editor/api/models.py`, `src/sonna_editor/api/routes/folders.py`, `src/sonna_editor/api/routes/profiles.py`, and `src/sonna_editor/data/catalog.py`.
   - Fixed file handle shadowing bug in `src/sonna_editor/training/datamodule.py:187`.
   - Fixed model metaclass attribute lookups in `src/sonna_editor/training/profile_runner.py:646-648`.
   - Fixed binary extract type narrowing in `src/sonna_editor/data/xmp.py:503`.
   - Fixed syntax/semantic error in `src/sonna_editor/api/routes/process.py:228` (illegal `return` inside `finally` block moved outside).
   - Added PEP 561 marker `src/sonna_editor/py.typed` and configured `ignore_missing_imports = true` for third-party libraries in `pyproject.toml`.
   - Configured `.vscode/settings.json` and `pyrightconfig.json` with `.venv` interpreter path and `src` extraPaths, eliminating ~890 missing-import and Pandas union diagnostic false alarms.
   - Verified 0 issues across all files via Pyright (`0 errors, 0 warnings, 0 informations`).

7. [x] **Task 7: Fix Exposure, Straightening, Temperature, and Color Wheel Red Cast Issues**
   - **Color Wheels Major (Fixed)**:
     - Diagnosed root cause: `default_skip_fields` in `v1_learning/model-v0.1.0.json` skipped `ColorGradeMidtoneHue` and `SplitToningShadowHue`. In Lightroom Classic, omitting a Hue attribute while writing a positive Saturation attribute (`crs:ColorGradeMidtoneSat="7"`, `crs:SplitToningShadowSaturation="6"`) forces Lightroom to default Hue to 0.0° (pure red), resulting in dark and midtone areas turning pure red.
     - Removed `ColorGradeMidtoneHue` and `SplitToningShadowHue` from `v1_learning/model-v0.1.0.json`'s `default_skip_fields`, allowing the user's preset hues (38° and 337°) to be written.
     - Added `COLOR_WHEEL_PAIRS` hue-saturation coupling guard in `src/sonna_editor/data/xmp.py` (`write_xmp`) and `src/sonna_editor/inference/pipeline.py`: positive color wheel saturations are omitted whenever their corresponding hue is missing or skipped, preventing Lightroom from ever rendering red color casts from missing hues.
   - **Straightening Major (Fixed)**:
     - Diagnosed root cause: Over-sensitive thresholds in `src/sonna_editor/inference/straighten.py` (`_MIN_APPLY_ANGLE = 0.08`, `_MIN_CONFIDENCE = 0.28`, `_MIN_HORIZON_CONFIDENCE = 0.22`, `_MIN_LINE_COUNT = 1`, `_MIN_LINE_LENGTH_PX = 24.0`). A single 24px line (e.g., bouquet, arm, garland, microphone stand) with 22% confidence was triggering crop and rotation on straight photos, cutting into heads and tilting framing.
     - Raised thresholds to conservative standards: `_MIN_APPLY_ANGLE = 0.25`, `_MIN_CONFIDENCE = 0.50`, `_MIN_HORIZON_CONFIDENCE = 0.45`, `_MIN_MIXED_CONFIDENCE = 0.55`, `_MIN_LINE_COUNT = 2`, `_MIN_LINE_LENGTH_PX = 36.0`, requiring substantial full-width line span (>= 0.75 * min dimension) if only a single line is present.
   - **Temperature Major (Fixed)**:
     - Diagnosed root cause: In `src/sonna_editor/inference/pipeline.py`, `_MODE_B_ADJUSTMENT_OPTIONS` set `"auto_white_balance": True`, which ran `_grey_world_wb` on every photo and shifted Kelvin up to ±300K and Tint up to ±5 based on scene color content. This caused inconsistent WB jumps between adjacent shots and forced WB from "As Shot" to "Custom". Also, missing As-Shot WB was defaulting to 5500K, turning tungsten indoor photos orange.
     - Set `"auto_white_balance": False` in `_MODE_B_ADJUSTMENT_OPTIONS` by default.
     - Updated `_anchor_mode_b_wb_to_as_shot`: presets without explicit WB keep `Temperature` and `Tint` as `None`, so `write_xmp` omits them and Lightroom preserves native camera **"As Shot"** WB.
   - **Exposure Minor (Fixed)**:
     - Diagnosed root cause: In `src/sonna_editor/preset/adjuster.py`, `compute_adjustment` calculated `raw_delta` from raw luminance to hit target luminance 118, and `apply_adjustment` added `base_preset["Exposure2012"] + raw_delta`. If the preset had base exposure (e.g. +0.40), the final exposure was double-bumped (+0.90). Additionally, `p95` highlight capping was overly aggressive on dark frames with diffuse white elements, and `low_light_floor` had a sharp cliff at `p95 < 230`.
     - In `compute_adjustment`, adjusted exposure delta to account for base preset: `net_delta = target_delta - base_exp`, guaranteeing the final applied exposure matches the intended scene target without double bumps.
     - Switched highlight protection in `_exposure_delta` to `p98` and `p99.5` with smoothed low-light floor, preventing dark scenes from being throttled by small background specular lights.
   - **Verification**:
     - 791 pytest tests passed (0 failures, 41 skipped, 4 deselected).
     - `uv run ruff check .` passed with 0 errors.
     - Mypy passed with 0 errors across modified files and `api/`.

8. [x] **Task 8: Cross-Platform Environment Setup, Code Weakness Fixes, Mac Compatibility & Client Distribution**
   - Bootstrapped `uv` 0.13.8, portable Node.js LTS v22.11.0, official signed CPython 3.11.9.
   - Consolidated 147 default slider values in `config.py` as `DEFAULT_SLIDER_VALUES`, re-exported in `xmp.py` and sliced in `config.py`, eliminating duplicate tables.
   - Sanitized hardcoded absolute paths in `SonnaEditorFoundation/checkpoints/` to relative repository paths.
   - Enabled Personal AI profile creation in frontend UI (`profile-view.jsx`).
   - Added bounded sliding window in `pipeline.py` to throttle memory on large shoots.
   - Created zero-touch macOS launcher `start_saha_mac.command` with auto Homebrew/uv/Node.js setup and Finder execution fix in Electron.
   - Packaged 106.2 MB ready-to-test client bundle `dist/saha-mac-client-bundle.zip` and authored `doc/MAC_CLIENT_TESTING.md`.

9. [x] **Task 9: Documentation Pruning, Consolidation & Enforced Change Recording Flow**
   - Consolidated duplicate instruction file: reduced `CLAUDE.md` to a clean pointer directing Claude Code to `AGENTS.md`.
   - Merged unique macOS developer prerequisites into `doc/RUN.md` and deleted redundant 694-line `doc/MAC_SETUP.md`.
   - Streamlined `doc/SESSION_STATE.md` from 1,315 lines (121 KB) to ~80 lines, eliminating 1,200 lines of obsolete July 2026 terminal dumps.
   - Codified the **Mandatory Change Recording Flow** in `AGENTS.md`: every code modification, optimization, bug fix, and doc change MUST be recorded in `session.md`.
   - Updated `README.md`, `doc/project_knowledge.md`, `doc/HANDOVER.md`, and `doc/SESSION_STATE.md` to form a cohesive, unbloated documentation hierarchy.

10. [x] **Task 10: Store UI Reference Images & Directory Setup**
   - Created `references/` directory at the project root for UI and workflow reference assets.
   - Copied and sequentially organized 40 reference screenshots covering Imagen's onboarding, project creation, profile selection, preset upload, visual style survey steps (Exposure, Temperature, and Tint comparisons across different scenes), building/ready confirmation modals, catalog/folder ingestion choices, profile settings fine-tuning overrides, Personal AI profile training wizard steps, genre categorization, supported RAW formats, Denoise strength side-by-side controls, and active project processing states.
   - Authored comprehensive `references/README.md` indexing all 40 screenshots across 10 functional workflow groups with instructions for adding future reference assets.

11. [x] **Task 11: Fix Windows App Startup, Node.js PATH Detection & Double-Click Launchers**
   - Diagnosed startup failure when clicking launcher scripts on Windows: `C:\Program Files\nodejs` was missing from user/machine PATH registry, causing `start_saha_windows.bat` to fail with `[saha] Node.js not found on PATH.` and `scripts/run_app.py` to raise `RuntimeError: Node.js was not found on PATH.`. Additionally, clicking `start_saha_mac.command` on Windows failed because `.command` is a macOS shell script.
   - Permanently registered `C:\Program Files\nodejs` and `%USERPROFILE%\.local\bin` into Windows User PATH registry.
   - Enhanced `start_saha_windows.bat` and `run_saha.cmd` with automatic fallback path injection for Node.js, `uv`, and Python, falling back directly to `.venv\Scripts\python.exe` if `uv` is absent.
   - Updated `scripts/run_app.py` with `_ensure_windows_paths()` fallback discovery and environment injection so child processes (`npm`, `vite`, `electron`) inherit the Node directory on Windows.
   - Augmented `process.env.PATH` on `win32` in `saha-app/electron/main.js` so Electron finds local binaries without relying on inherited system PATH.
   - Created root convenience aliases `start.bat` and `START_SAHA.bat`.

12. [x] **Task 12: Cross-Platform Zip Portability & Local Disk Self-Healing Architecture**
   - Implemented self-healing cross-platform environment resets in `start_saha_mac.command`, `start_saha_windows.bat`, and `scripts/run_app.py`: when the project is zipped on Windows and unzipped on macOS (or vice versa), the launchers automatically detect OS binary mismatches in `.venv` and `node_modules`, cleanly reset them, and build native binaries for the host OS.
   - Enhanced `scripts/package_for_client.py` to bundle all local model weights (`SonnaEditorFoundation/checkpoints/` and `v1_learning/`), Lightroom bridge plugin, documentation, and launchers with POSIX executable permissions (`0o100755`) for macOS `.command` scripts.
   - Added root one-click packaging utility `package.bat` and verified 213.6 MB universal distribution bundle `dist/saha-app-bundle.zip`.

13. [x] **Task 13: Windows Batch Script Direct Path Discovery & Delayed Expansion Fix**
   - Diagnosed root cause for `[saha] Node.js not found on PATH.` remaining in cmd.exe: `setlocal enabledelayedexpansion` mangled `PATH` when `!PATH!` contained parentheses (`C:\Program Files (x86)`).
   - Replaced `where node` with direct path verification (`if exist "C:\Program Files\nodejs\node.exe"`) and cmd built-in path resolution (`for %%i in (node.exe) do if not "%%~$PATH:i"==""`), eliminating reliance on external `where.exe`.
   - Removed delayed expansion mangling, prepending all standard paths cleanly before launching.

14. [x] **Task 14: Launcher Redundancy Elimination, CLAUDE.md Removal, Cursor Integration & Explicit Commit Control**
   - Eliminated redundant root launch wrappers (`START_SAHA.bat`, `start.bat`, `run_saha.cmd`, `run_saha.ps1`, `run_saha.sh`).
   - Retained a single canonical pair: `start_saha_windows.bat` for Windows and `start_saha_mac.command` for macOS.
   - Removed `CLAUDE.md` and created `.cursorrules` pointing directly to `AGENTS.md` for seamless Cursor IDE integration alongside Antigravity.
   - Enforced the user rule in `AGENTS.md`: "DO NOT commit or push to git automatically. Only commit or push when explicitly commanded by the user."
   - Updated documentation (`README.md`, `doc/RUN.md`, `doc/CLI_COMMANDS.md`, `doc/HANDOVER.md`, `doc/project_knowledge.md`) and `scripts/package_for_client.py`.

15. [x] **Task 15: Cache Purge, Dev/Test Exclusion Filtering & One-Click Production Packaging Tool**
   - Cleaned all workspace caches (`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `saha-app/node_modules/.vite`, 18 `__pycache__` directories outside `.venv`).
   - Authored `scripts/package_app.py` with automatic cache pre-cleaning and production filtering: excludes `tests/`, test fixtures, 10 one-time testing/audit scripts, UI references, development caches, and `.git`, while bundling all core backend libraries, Electron frontend, foundation weights, trained profile weights, Lightroom plugin, and launchers.
   - Created one-click double-clickable launchers: `package.bat` (Windows) and `package.sh` (macOS/Linux).
   - Verified production distribution archive `dist/saha-production-bundle.zip` (139 files, 211.3 MB) containing only essential runtime assets with POSIX executable permissions (`0o100755`) preserved.

16. [x] **Task 16: Automated Versioning & Automatic Previous Package Deletion**
   - Implemented `resolve_next_version()` in `scripts/package_app.py`: auto-increments package versions (`saha-v1.0.zip`, `saha-v1.1.zip`, `saha-v1.2.zip`) while supporting explicit custom versions (`package.bat 2.0`).
   - Implemented `delete_old_packages()` in `scripts/package_app.py`: automatically purges all older `.zip` packages in `dist/` before and after building, ensuring only the single latest versioned archive remains on disk.
   - Tested version incrementation and verified deletion of previous archives (`saha-v1.0.zip` automatically replaced by `saha-v1.1.zip`).

17. [x] **Task 17: Renamed Run Launchers to CLICK_ME_MAC/WINDOWS_TO_OPEN_APP**
   - Renamed macOS double-clickable launcher from `start_saha_mac.command` to `CLICK_ME_MAC_TO_OPEN_APP.command`.
   - Renamed Windows double-clickable launcher from `start_saha_windows.bat` to `CLICK_ME_WINDOWS_TO_OPEN_APP.bat`.
   - Updated packaging logic in `scripts/package_app.py` to bundle new launcher filenames and maintain POSIX executable permissions (`0o100755`) for the macOS launcher.
   - Updated all operational documentation (`README.md`, `doc/RUN.md`, `doc/MAC_CLIENT_TESTING.md`, `doc/CLI_COMMANDS.md`, `doc/HANDOVER.md`, `doc/SESSION_STATE.md`, `doc/project_knowledge.md`).
   - Packaged and verified `dist/saha-v1.3.zip`.

18. [x] **Task 18: Rename Run Launchers to DOUBLE_CLICK_MAC/WIN_START_SAHA & Commit Changes**
   - Renamed macOS launcher to `DOUBLE_CLICK_MAC_START_SAHA.command`.
   - Renamed Windows launcher to `DOUBLE_CLICK_WIN_START_SAHA.bat`.
   - Updated `scripts/package_app.py` (`INCLUDE_PATHS`, `EXECUTABLE_NAMES`, `QUICK_START.txt`).
   - Synchronized all documentation across repository.
   - Rebuilt distribution package `dist/saha-v1.4.zip`.
   - Executed clean git commit per user command.

---

## Detailed Change Log

### Task 1: Documentation Restructuring
- Created `doc/` directory.
- Moved:
  - `CLI_COMMANDS.md` -> `doc/CLI_COMMANDS.md`
  - `FOUNDATION_TRAINING.md` -> `doc/FOUNDATION_TRAINING.md`
  - `HANDOVER.md` -> `doc/HANDOVER.md`
  - `MAC_SETUP.md` -> `doc/MAC_SETUP.md`
  - `RUN.md` -> `doc/RUN.md`
  - `SESSION_STATE.md` -> `doc/SESSION_STATE.md`
  - `SONNA_EDITOR_BUILD_SPEC.md` -> `doc/SONNA_EDITOR_BUILD_SPEC.md`
  - `project_knowledge.md` -> `doc/project_knowledge.md`
- Created `session.md` at project root.

### Task 2: Code & Structure Cleanup
- Removed empty folder `src/sonna_editor/Notes/`.
- Removed legacy unused placeholder UI package `src/sonna_editor/ui/`.
- Cleaned unused variable `v_str` in `scripts/finetune_profile.py`.
- Cleaned unused variable `end` in `src/sonna_editor/mode_b/checkpoint_builder.py`.

### Task 3: Warnings & Error Resolution
- Resolved PyTorch `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior` in `tests/test_losses.py` by applying `.detach()` before converting to float scalar.

### Task 4 & 5: Optimization & Full Verification
- `src/sonna_editor/preset/adjuster.py`: eliminates up to 3 separate PIL Image conversions per photo by sharing single float32 RGB and luminance ndarrays.
- `src/sonna_editor/inference/straighten.py`: avoids float32 promotion/division and clipping round-trips for horizon/straighten analysis.
- `src/sonna_editor/data/extract.py`: regex precompilation and single-pass dual-percentile computation.
- Full pytest verification: 788 tests passed, 0 failures, 0 warnings.
- Frontend build verified: `dist/index.html` (2.13 kB) and `dist/assets/index-f2qWcwkj.js` (236.17 kB) built in 510ms.

### Task 8: Cross-Platform Environment Setup, Code Weakness Fixes, Mac Compatibility & Client Distribution
- **Environment Bootstrapping**: Installed `uv` 0.13.8, portable Node.js LTS v22.11.0, official signed CPython 3.11.9 (resolving Windows Smart App Control error 4551), and synced full `.venv` with CUDA 12.8 PyTorch, TorchVision, rawpy, OpenCV, etc.
- **Slider Defaults Consolidation**: Defined single canonical `DEFAULT_SLIDER_VALUES` for all 147 sliders in `src/sonna_editor/config.py`. Made `SLIDER_DEFAULTS` in `config.py` a clean slice and `LR_DEFAULTS` in `src/sonna_editor/data/xmp.py` re-export it, removing 95 lines of duplicate tables and preventing any table drift.
- **Checkpoint Path Portability**: Sanitized machine-specific `/Users/sonnawork/...` absolute paths in `SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-004-visual.json` and `SonnaEditorFoundation/foundation_manifest.json` to relative repository paths.
- **Personal AI Profile UI Wiring**: Enabled the "Personal AI profile" button in `saha-app/src/components/profile-view.jsx`, removing the hardcoded disabled "Coming soon" state and wiring `PersonalProfileWizard` with complete step-by-step training from RAW+XMP folders.
- **Memory Throttling in Inference Pipeline**: Replaced unbounded parallel submission in `src/sonna_editor/inference/pipeline.py` with a sliding window (bounded in-flight tasks) to eliminate memory spikes on large shoots (5,000+ images).
- **Mac Launcher & Finder Compatibility**:
  - Enhanced `start_saha_mac.command` to auto-detect Homebrew on Apple Silicon (`/opt/homebrew/bin`), auto-install `uv`, and auto-bootstrap portable Node.js LTS for `darwin-arm64` if Node is not on the Mac.
  - Augmented `process.env.PATH` in `saha-app/electron/main.js` so Electron finds `uv` and Python when double-clicking `Saha.app` or running from Finder.
- **Client Packaging**: Created `scripts/package_for_client.py` and generated a self-contained 106MB client test bundle `dist/saha-mac-client-bundle.zip` with `start_saha_mac.command`, pretrained foundation AI weights, Lightroom Classic plugin, and quick-start instructions for Darshil. Created `doc/MAC_CLIENT_TESTING.md`.
- **Verification**: Frontend Vite build passed in 31s; `ruff check src/ scripts/` passed with 0 errors; pytest suites (`test_config.py`, `test_xmp.py`) passed 100%.

### Task 9: Documentation Pruning, Consolidation & Enforced Change Recording Flow
- **Consolidated Duplicate Instruction Files**:
  - `CLAUDE.md` reduced from 153 redundant lines to a clean pointer directing Claude Code directly to `AGENTS.md`.
  - `AGENTS.md` established as the single authoritative operational rules and agent instructions document.
- **Eliminated Redundant Documentation**:
  - Deleted `doc/MAC_SETUP.md` (694 lines) which duplicated commands already maintained in `doc/RUN.md`, `doc/CLI_COMMANDS.md`, and `doc/MAC_CLIENT_TESTING.md`.
  - Incorporated all unique macOS developer setup steps (Xcode command line tools, Homebrew setup, Git LFS `.ckpt` pull instructions) into a dedicated section in `doc/RUN.md`.
  - Sanitized hardcoded machine paths in `doc/RUN.md` to generic cross-platform paths.
- **Streamlined `doc/SESSION_STATE.md`**:
  - Stripped 1,200 lines of obsolete June/July 2026 terminal dumps from a defunct workstation (`vikas.DESKTOP-61LEE8B`), reducing file size from 121 KB (1,315 lines) to ~80 lines.
  - Refocused `doc/SESSION_STATE.md` as a clean architecture reference documenting active foundation checkpoint lineage (`foundation-sonna-raw-xmp-004-visual`), Lite profile lineage, and runtime behaviors.
- **Codified Mandatory Change Recording Flow**:
  - Added explicit instructions in `AGENTS.md`: every code modification, optimization, bug fix, and doc change MUST be recorded in `session.md` at the project root with rationale, files modified, and verification results.
  - Aligned `README.md`, `doc/HANDOVER.md`, `doc/project_knowledge.md`, and `doc/SESSION_STATE.md` with this unified documentation hierarchy.
- **Verification**:
  - `ruff check .` passed with 0 errors.
  - Regression pytest suites passed with 100%.
  - Re-generated client package `dist/saha-mac-client-bundle.zip` with the pruned, clean documentation.

### Task 10: Store UI Reference Images & Directory Setup (40 Screenshots)
- Created `references/` directory at project root to house reference screenshots and UX workflow patterns from Imagen.
- Organized 40 reference images into 10 structured functional workflow categories:
  - 1. Onboarding & Core Views (`01` - `03`)
  - 2. Profile Selection & Qualification Wizard (`04` - `07`)
  - 3. Mode B (Lite Profile) Setup (`08` - `10`)
  - 4. Mode B (Lite Profile) Style Survey (`11` - `19`)
  - 5. Profile Generation & Ready States (`20` - `21`)
  - 6. Project Ingestion & Workflow Sources (`22` - `23`)
  - 7. Profile Management & Tuning Overrides (`24` - `26`)
  - 8. Mode A Training Wizard Setup (`27` - `30`)
  - 9. Mode A Profile Creation & Catalog Ingestion (`31` - `35`)
  - 10. Project Creation & Advanced AI Tools (`36` - `40`)
- Authored `references/README.md` cataloging each screenshot with descriptions, core UX patterns, and instructions for adding future references.

### Task 11: Fix Windows App Startup, Node.js PATH Detection & Double-Click Launchers
- **Root Cause Analysis**:
  - The installed Node.js binary was located at `C:\Program Files\nodejs\node.exe`, but neither the Windows User PATH nor System PATH in registry contained this directory.
  - When double-clicking `start_saha_windows.bat`, `where node` failed with exit code 1, displaying `[saha] Node.js not found on PATH.` and pausing.
  - When running `scripts/run_app.py`, `_node_command()` raised `RuntimeError: Node.js was not found on PATH.`
  - Double-clicking `start_saha_mac.command` on Windows failed because `.command` is a macOS-only file association.
- **Fixes Applied**:
  - Permanently added `C:\Program Files\nodejs` and `%USERPROFILE%\.local\bin` to User PATH in registry (`HKCU\Environment`).
  - Updated `start_saha_windows.bat` to auto-detect and prepend standard paths (`C:\Program Files\nodejs`, `%USERPROFILE%\.local\bin`, `%APPDATA%\npm`, Python) and fall back directly to `.venv\Scripts\python.exe` if `uv` is not present. Added exit error pause for debugging visibility.
  - Updated `run_saha.cmd` with the same PATH injection and fallback logic.
  - Added `_ensure_windows_paths()` and fallback path searches in `scripts/run_app.py` for both `_node_command()` and `_npm_command()`, and injected `C:\Program Files\nodejs` into `os.environ["PATH"]` for child process execution (`npm`, `vite`, `electron`).
  - Augmented `process.env.PATH` on Windows in `saha-app/electron/main.js` matching macOS behavior.
  - Created root-level double-clickable convenience scripts `start.bat` and `START_SAHA.bat`.
- **Verification**:
  - `start_saha_windows.bat --help`, `start.bat --help`, `START_SAHA.bat --help`, and `run_saha.cmd --help` all execute cleanly with zero errors.
  - `uv run ruff check scripts/run_app.py` passed with 0 errors.
  - Pytest regression suite (`test_config.py`, `test_xmp.py`) passed 100% (95 passed, 10 skipped in 11.74s).

### Task 12: Cross-Platform Zip Portability & Local Disk Self-Healing Architecture
- **Cross-Platform Mismatch Detection & Self-Healing**:
  - `start_saha_mac.command`: Automatically detects if `.venv` or `saha-app/node_modules` was copied from a Windows machine (e.g. presence of `.venv\Scripts` instead of `.venv/bin/python` or missing `Electron.app` Mach-O bundle). Cleans up foreign binaries so `uv` and `npm` build native macOS binaries.
  - `start_saha_windows.bat`: Automatically detects if `.venv` or `saha-app/node_modules` was copied from a Mac (missing `python.exe` or `electron.exe`) and resets them for Windows.
  - `scripts/run_app.py`: Implemented `_is_node_modules_valid_for_platform(app_dir)` to inspect Electron binaries across Windows (`electron.exe`), macOS (`Electron.app`), and Linux (`electron`), automatically reinstalling if a mismatch is found.
- **Complete Local Asset Bundling**:
  - Updated `scripts/package_for_client.py` `INCLUDE_PATHS` to include `v1_learning/` (contains trained model `model-v0.1.0.ckpt`), `SonnaEditorFoundation/checkpoints/` (contains foundation model `foundation-sonna-raw-xmp-004-visual.ckpt`), `references/`, `.python-version`, and all double-click launchers.
  - Preserved POSIX executable bit `0o100755` (`-rwxr-xr-x`) for `start_saha_mac.command` and `run_saha.sh` inside the zip archive, guaranteeing double-click execution on macOS upon unzipping.
  - Created root one-click packaging utility `package.bat`.
  - Built universal client test bundle `dist/saha-app-bundle.zip` (213.6 MB, 195 entries).
- **Verification**:
  - `uv run ruff check scripts/run_app.py scripts/package_for_client.py` passed with 0 errors.
  - Verified `dist/saha-app-bundle.zip` entries: confirmed `QUICK_START.txt`, foundation weights, v1 learning weights, and POSIX `0o100755` attribute on `start_saha_mac.command`.

### Task 13: Windows Batch Script Direct Path Discovery & Delayed Expansion Fix
- Diagnosed root cause for `[saha] Node.js not found on PATH.` remaining in cmd.exe:
  - `setlocal enabledelayedexpansion` in `start_saha_windows.bat` mangled `PATH` when `!PATH!` contained parentheses like `C:\Program Files (x86)` or `%PATH%`.
  - Nested `if exist` parenthesis blocks triggered a syntax parser error (`. was unexpected at this time.`).
- Replaced `where node` with direct path verification (`if exist "C:\Program Files\nodejs\node.exe"`) and cmd built-in path resolution (`for %%i in (node.exe) do if not "%%~$PATH:i"==""`), eliminating reliance on external `where.exe`.
- Removed delayed expansion mangling, prepending all standard paths cleanly before launching.
- Verified: `start.bat --help`, `START_SAHA.bat --help`, `start_saha_windows.bat --help`, and `run_saha.cmd --help` all succeed with returncode 0 and zero stderr.
- Rebuilt `dist/saha-app-bundle.zip`.

### Task 14: Launcher Redundancy Elimination, CLAUDE.md Removal, Cursor Integration & Explicit Commit Control
- **Eliminated Redundant Launchers**:
  - Deleted duplicate wrappers: `START_SAHA.bat`, `start.bat`, `run_saha.cmd`, `run_saha.ps1`, `run_saha.sh`.
  - Standardized on a single, clear pair:
    - Windows: `start_saha_windows.bat`
    - macOS: `start_saha_mac.command`
- **Removed CLAUDE.md & Added Cursor Support**:
  - Deleted `CLAUDE.md` per user directive to focus exclusively on Antigravity and Cursor.
  - Added `.cursorrules` pointing Cursor directly to `AGENTS.md` as the single authoritative source of project instructions and rules.
- **Enforced Explicit Git Commit Control**:
  - Updated `AGENTS.md`: "DO NOT commit or push to git automatically. Only create git commits or push to remote when explicitly commanded by the user."
  - Updated all documentation references in `README.md`, `doc/RUN.md`, `doc/CLI_COMMANDS.md`, `doc/HANDOVER.md`, `doc/project_knowledge.md`, and `scripts/package_for_client.py`.
- **Verification**:
  - `start_saha_windows.bat --help` executed with code 0.
  - `uv run ruff check scripts/package_for_client.py` passed with 0 errors.
  - No automatic git commit executed per user command.

### Task 15: Cache Purge, Dev/Test Exclusion Filtering & One-Click Production Packaging Tool
- **Cache Purge**:
  - Removed `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, and `saha-app/node_modules/.vite`.
  - Removed 18 `__pycache__` directories across `src/`, `scripts/`, `tests/` outside `.venv`.
- **Production Packaging Tool (`scripts/package_app.py`)**:
  - Implemented pre-packaging cache purge (`clean_local_caches`).
  - Strict production filtering (`is_path_excluded`):
    - Excludes `tests/`, test fixtures, and test caches.
    - Excludes 10 one-time test/audit/diagnostic scripts: `analyse_backbone_drift.py`, `analyse_prediction_collapse.py`, `audit_all_sliders_v1.2.3.py`, `audit_catalog.py`, `audit_dataset_diversity.py`, `quick_diagnostic.py`, `run_v1_pilot.py`, `train_v1_2_0_full_production.py`, `migrate_labels_to_v2.py`, `rollback_foundation.py`.
    - Excludes `references/` (UI reference screenshots).
    - Excludes `.git`, `.venv`, `node_modules`, `scratch`, `dist`, `dist-app`.
    - Preserves POSIX executable bit (`0o100755`) for macOS `.command` and `.sh` scripts.
  - Retains all core runtime code (`src/sonna_editor/`), full frontend (`saha-app/`), foundation weights, v1 learning profile weights, Lightroom plugin, essential runtime scripts, and launchers.
- **One-Click Distribution Launchers**:
  - Updated `package.bat` for Windows (one-click double-click builder).
  - Created `package.sh` for macOS/Linux (`bash package.sh`).
  - Updated `scripts/package_for_client.py` as a seamless alias to `scripts/package_app.py`.
- **Verification**:
  - `uv run ruff check scripts/package_app.py scripts/package_for_client.py` passed with 0 errors.
  - Verified `dist/saha-production-bundle.zip` (139 files, 211.3 MB zip / 229.2 MB uncompressed): confirmed 0 test files, 0 audit scripts, 0 cache directories, and valid POSIX permissions.
  - No automatic git commit executed per user command.

### Task 16: Automated Versioning & Automatic Previous Package Deletion
- **Automated Versioning**:
  - Added `resolve_next_version(dist_dir, requested_version)` in `scripts/package_app.py`.
  - Format: `saha-vX.Y.zip` (e.g. `saha-v1.0.zip`, `saha-v1.1.zip`, `saha-v1.2.zip`).
  - Auto-increments minor version on each packaging run if no version argument is specified.
  - Supports explicit version arguments (e.g. `package.bat 2.0` or `uv run python scripts/package_app.py 2.0`).
  - Injects dynamic version metadata (`Saha v{version}`) into bundled `QUICK_START.txt`.
- **Automatic Old Package Purging**:
  - Added `delete_old_packages(dist_dir, exclude_file)` in `scripts/package_app.py`.
  - Automatically identifies and deletes all previous `.zip` bundles in `dist/` before creating the new archive, ensuring disk space is conserved and only the latest build exists.
- **Verification**:
  - Ran first execution: created `saha-v1.0.zip` and deleted previous unversioned bundles (`saha-app-bundle.zip`, `saha-production-bundle.zip`).
  - Ran second execution: auto-incremented to `saha-v1.1.zip`, deleted `saha-v1.0.zip`, and verified that only `dist/saha-v1.1.zip` remains in `dist/`.
  - No automatic git commit executed per user command.

### Task 17: Renamed Run Launchers to CLICK_ME_MAC/WINDOWS_TO_OPEN_APP
- **Renamed Run Launchers**:
  - Renamed Windows launcher: `start_saha_windows.bat` -> `CLICK_ME_WINDOWS_TO_OPEN_APP.bat`.
  - Renamed macOS launcher: `start_saha_mac.command` -> `CLICK_ME_MAC_TO_OPEN_APP.command`.
- **Packaging & Permission Updates**:
  - Updated `scripts/package_app.py` `INCLUDE_PATHS` to package the new filenames.
  - Maintained POSIX executable permission bit (`0o100755`) for `CLICK_ME_MAC_TO_OPEN_APP.command` so unzipping on macOS preserves double-click execution.
  - Updated embedded `QUICK_START.txt` instructions to direct users to double-click `CLICK_ME_WINDOWS_TO_OPEN_APP.bat` on Windows and `CLICK_ME_MAC_TO_OPEN_APP.command` on macOS.
- **Documentation Synchronization**:
  - Synchronized references across `README.md`, `doc/RUN.md`, `doc/MAC_CLIENT_TESTING.md`, `doc/CLI_COMMANDS.md`, `doc/HANDOVER.md`, `doc/SESSION_STATE.md`, and `doc/project_knowledge.md`.
- **Verification**:
  - Tested Windows launcher: `cmd /c CLICK_ME_WINDOWS_TO_OPEN_APP.bat --help` exited with code 0.
  - Generated distribution archive `dist/saha-v1.3.zip` (138 files, 211.3 MB zip / 229.2 MB uncompressed).
  - Verified previous archive `dist/saha-v1.2.zip` was purged and only `dist/saha-v1.3.zip` remains in `dist/`.
  - No automatic git commit executed per user command.

### Task 18: Rename Run Launchers to DOUBLE_CLICK_MAC/WIN_START_SAHA & Commit Changes
- **Launcher Renames**:
  - Renamed Windows launcher: `DOUBLE_CLICK_WIN_START_SAHA.bat`.
  - Renamed macOS launcher: `DOUBLE_CLICK_MAC_START_SAHA.command`.
- **Packaging Updates**:
  - Updated `scripts/package_app.py` `INCLUDE_PATHS` and `EXECUTABLE_NAMES` (`{"DOUBLE_CLICK_MAC_START_SAHA.command", "package.sh"}`).
  - Updated embedded `QUICK_START.txt` instructions in `scripts/package_app.py`.
- **Documentation Synchronization**:
  - Synchronized all documentation across repository (`README.md`, `doc/RUN.md`, `doc/MAC_CLIENT_TESTING.md`, `doc/CLI_COMMANDS.md`, `doc/HANDOVER.md`, `doc/SESSION_STATE.md`, `doc/project_knowledge.md`).
- **Packaging & Verification**:
  - Ran `scripts/package_app.py`: auto-incremented to `saha-v1.4.zip`, purged `saha-v1.3.zip`, and embedded launchers with POSIX executable permissions (`0o100755`).
  - Tested `cmd /c DOUBLE_CLICK_WIN_START_SAHA.bat --help`: exited with code 0.
  - Executed git commit per user command.


### Task 47: Implement 4-Step New Project Setup Wizard Matching Reference UI Specification
- **Redesigned New Project Flow into 4-Step Wizard in [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx)**:
  - **Universal Stepper Header**: Replaced single scrollable modal with 4-step wizard header (`① Photos` ── `② Project setup` ── `③ Your workflow` ── `④ AI tools` + `✕` close button) with completed step checkmarks `✓` and active step highlights matching reference screenshots.
  - **Step 1 (Photos)**: Preserves dual-mode source selection (Lightroom Classic catalog and Folder) with drag-and-drop support.
  - **Step 2 (Project setup - Image 2 match)**:
    - Top card displaying folder path, total detected photos count, and `Name your project` input prefilled with directory name.
    - Bottom card with `Choose a photography type` displaying 10 genres in a 2x5 grid (Wedding, School, Family / newborn, Sports, Events, Real estate, Portraits, Boudoir, Landscape / nature, Other) with crisp SVG icons.
    - Footer with `${detectedPhotoCount} Photos selected`, `Back`, and theme-colored `Next` button.
  - **Step 3 (Your workflow - Image 3 match)**:
    - Streamlined to strictly contain the `AI Profile` option (omits AI culling per user instructions).
    - Lists available created profiles (both Lite and Personal AI) with radio selection cards.
    - Zero-profile state displays clear guidance with a `+ Create AI profile` button that redirects to the profile creation modal.
    - The `Next` button in the footer is strictly disabled until an AI profile is selected.
  - **Step 4 (AI tools - Image 4 match)**:
    - Exclusively provides the 3 specified tools: **Crop**, **Straighten**, and **Denoise**.
    - Top bar with active tool dismissible pills (`Crop ×`, `Straighten ×`, `Denoise ×`) and `Clear all` button.
    - Left column displays tool toggles and granular controls (Aspect ratio chips for Crop, CropAngle for Straighten, ISO strength and trigger slider for Denoise).
    - Right column displays interactive before/after split preview comparison card with draggable slider handle and tailored tool descriptions.
    - Footer features `Create project` CTA in signature theme color `#E05353`.
- **App Integration in [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx)**:
  - Connected `onOpenCreateProfile` prop to `NewProjectModal`, closing the project modal and smoothly opening the profile creation type selector when no profile exists.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 498ms with 0 errors (bundle: 278.30 kB).
  - `ruff check src scripts tests`: All checks passed.
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 8.48s with 0 failures.
  - Packaged distribution archive: `dist/saha-v1.28.zip` (139 files, 106.1 MB zip / 115.7 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 48: Preserve 3rd Step State in New Project Modal During AI Profile Creation
- **Preserved Ingestion Progress and Modal Layering**:
  - Identified that opening profile creation from Step 3 ("Your workflow") previously unmounted `NewProjectModal`, resetting the wizard and losing the user's picked folder, detected photo count, project name, and photography genre.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Changed `onOpenCreateProfile` to keep `isNewProjectOpen` active, opening `ChooseProfileTypeModal` in a stacked modal layer.
    - Added `handleProfileCreated` to track the ID of the newly created profile (`newlyCreatedProfileId`) and trigger `profilesQ.refetch()`.
    - Passed `newlyCreatedProfileId` to `NewProjectModal`.
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Added `wasOpenRef` guard to ensure the reset effect only fires when the modal transitions from closed to open, preventing profile refetches or child modal interactions from wiping state or jumping back to Step 1.
    - Added an effect to auto-select `newlyCreatedProfileId` as soon as it arrives, highlighting the newly created profile and enabling the `Next` button.
  - **Z-Index Modal Hierarchy Realignment**:
    - `NewProjectModal`: `zIndex: 1000`.
    - `ChooseProfileTypeModal` ([saha-app/src/components/profile-type-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-type-modal.jsx)): raised to `zIndex: 1100`.
    - `LiteProfileWizard` ([saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx)): raised to `zIndex: 1100`.
    - `PersonalProfileWizard` ([saha-app/src/components/personal-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/personal-wizard.jsx)): raised to `zIndex: 1100`.
    - `ConfirmModal` ([saha-app/src/components/confirm-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/confirm-modal.jsx)): raised to `zIndex: 1200`.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 549ms with 0 errors (bundle: 278.77 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 1.31s with 0 failures.
  - `ruff check src scripts tests`: All checks passed.
  - Packaged distribution archive: `dist/saha-v1.29.zip` (143 files, 211.3 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 49: Update Step 3 AI Profile Selector to Dropdown Menu Matching Reference UI
- **Redesigned Step 3 Profile Selection in [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx)**:
  - Matched reference screenshot `media_1794861157298_5b5557e9.png`:
    - Replaced the full profile card grid with the clean two-column status & dropdown layout.
    - **Left Side**: Displays `No profile selected` when unselected, or the profile name with an active indicator and `Lite` / `Personal` badge when selected.
    - **Right Side (When profiles exist)**: Displays the pill button `Choose profile ▾` that triggers a popover dropdown menu anchored below.
    - **Dropdown Menu**: Displays `Your AI Profiles` header, uppercase profile rows with hover highlights, selection checkmarks, and a direct `+ Create new profile` action at the bottom.
    - **Right Side (When no profiles exist)**: Directly provides the `+ Create AI profile` button in theme accent `#E05353` per user requirement.
    - Added an outside-click listener via `dropdownRef` to smoothly dismiss the dropdown on backdrop clicks.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 688ms with 0 errors (bundle: 280.03 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 1.02s with 0 failures.
  - `ruff check src scripts tests`: All checks passed.
  - Packaged distribution archive: `dist/saha-v1.30.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 50: Realign Step 4 AI Tools Layout with Recommended and Composition Menus
- **Aligned Step 4 to Reference Specification [media_1794861659425_f693902c.png] in [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx)**:
  - Replaced the two-column container with the full 3-column layout matching the reference UI:
    - **Column 1 (`AI tools` Sidebar)**: Vertical navigation featuring **Recommended** (`✨`) and **Composition** (`◫`) tabs. Composition displays the active tool count badge (`1 active` / `2 active`) in blue.
    - **Column 2 (Center Tools List)**:
      - Category subheadings: `✨ Recommended for ${genre}` and `◫ Composition`.
      - **Crop**: Toggleable with `(+)` or `(✓)` circular buttons, with aspect ratio chip selection when active.
      - **Straighten**: Accompanied by the signature `Top used` badge (`#2563EB`) and circular toggle button.
    - **Column 3 (Right Preview Pane)**:
      - Photo preview illustrating tilted beach couple horizon with corner brackets, rule-of-thirds grid, and rotation indicator arrow (`↷`).
      - Title: `Straighten` (or `Crop`).
      - Subtitle: `Automatically rotates every photo according to its horizon` (or `Automatically crops every photo according to its composition`).
  - Seamlessly keeps both Crop and Straighten synced with `onCreateProject` parameters.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 633ms with 0 errors (bundle: 282.21 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 1.00s with 0 failures.
  - `ruff check src scripts tests`: All checks passed.
  - Packaged distribution archive: `dist/saha-v1.31.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 51: Instant Hover Preview Switching for Crop and Straighten in Step 4
- **Implemented Dynamic Hover Preview Matching Screen Recording [media_1794862074203_87e9338c.mp4]**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Added `onMouseEnter={() => setPreviewTool('crop')}` and `onMouseEnter={() => setPreviewTool('straighten')}` across both the **Recommended** and **Composition** tool cards, giving immediate preview updates when moving the cursor across tools.
    - Added hover highlight styling (`#2A2E36` background and `rgba(255, 255, 255, 0.24)` border) on card rows.
    - Updated **Crop Preview**:
      - Renders the golden meadow embracing couple under sunset sky.
      - 3x3 rule-of-thirds grid lines, white corner marks, edge center marks, and bottom-left diagonal resize arrow `⤡`.
      - Updated subtitle text matching video: *"Crops to a 2x3 aspect ratio for optimal composition in many types of photography."*
    - Updated **Straighten Preview**:
      - Renders the sunset beach couple walking along the ocean tilted at -4°.
      - 3x3 grid lines, white corner marks, and right-hand rotation arrow `↷`.
      - Subtitle text matching video: *"Automatically rotates every photo according to its horizon"*.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 517ms with 0 errors (bundle: 283.42 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 4.74s with 0 failures.
  - `ruff check src scripts tests`: All checks passed.
  - Packaged distribution archive: `dist/saha-v1.32.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 52: Default Dark Theme Across App and Instant Load Without Flash
- **Root and State Configuration for Dark Theme as System Default**:
  - In [saha-app/src/index.html](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.html):
    - Changed `<html lang="en">` to `<html lang="en" data-theme="dark">`.
    - Swapped root CSS variable mappings so `:root, html[data-theme="dark"]` serves as the primary dark theme definition (`--saha-bg-deep: #16140F`, `--saha-bg-panel: #1F1D17`, etc.).
    - Preserved `html[data-theme="light"]` as an explicit override block for light mode.
    - Prevents any white or cream background flash prior to React mounting.
  - In [saha-app/src/tokens.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/tokens.js):
    - Changed default token theme to `theme: 'dark'` and fallback in `applyTheme(theme)` to `'dark'`.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Initialized `theme` state to default to `'dark'` while respecting user preferences stored in `localStorage.getItem('saha-theme')`.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 574ms with 0 errors (bundle: 283.52 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 4.85s with 0 failures.
  - `ruff check src scripts tests`: All checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.33.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 53: Enforce Default Dark Theme on Cold Start and Purge Stale Storage
- **Identified Cold-Start Light Theme Cause**:
  - `saha-theme` in the browser / Electron `localStorage` had been populated with `'light'` during earlier development sessions before the dark theme default was introduced.
  - In `App.jsx`, reading `window.localStorage?.getItem('saha-theme') || 'dark'` evaluated to `'light'` on cold start because `'light'` was already saved, overriding the new dark default.
- **Enforced Direct Dark Theme & Purged Legacy Storage**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Initialized state directly to `const [theme, setTheme] = useState('dark');` ensuring every cold start unconditionally loads in dark theme.
    - Updated `useLayoutEffect` to purge any stale `'light'` entry from `localStorage` whenever in dark theme.
  - In [saha-app/src/index.html](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.html):
    - Added an inline `<script>` in `<head>` preceding any CSS or React initialization to remove any stale `saha-theme: light` from `localStorage` and set `document.documentElement.dataset.theme = 'dark'`.
  - In [saha-app/src/index.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.jsx):
    - Called `applyTheme('dark')` at top-level before mounting the React root.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 593ms with 0 errors (bundle: 283.46 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 13.49s with 0 failures.
  - `ruff check src scripts tests`: All checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.34.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 54: Clean Active Badge Formatting, Remove Crop Aspect Options, and Reduce Card Margin
- **Fixed Active Count Badge in Step 4 Navigation**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Widened the AI tools sidebar column from `175px` to `195px`.
    - Added `whiteSpace: 'nowrap'`, `flexShrink: 0`, and explicit vertical centering (`lineHeight: 13px`, `padding: 2px 7px`, `borderRadius: 10px`) to the `{compActiveCount} active` badge on the Composition tab.
    - Prevents the badge text from wrapping into two vertical lines (`2` on top of `active`), keeping it as a crisp, single-line pill.
- **Removed Aspect Ratio Options from Crop**:
  - Removed the `Aspect ratio: Original, 1:1, 4:5, 16:9, Free` button group from the Crop card across both Recommended and Composition categories.
  - Matches the streamlined single-line card design from Imagen reference.
- **Reduced Margin Between Crop and Straighten Cards**:
  - Encapsulated Crop and Straighten cards in a dedicated container with a tight `gap: 6px`.
  - Reduced individual card padding from `14px 18px` to `11px 16px` for a compact, cohesive stack.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 45 modules transformed in 575ms with 0 errors (bundle: 282.39 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 1.10s with 0 failures.
  - `ruff check src scripts tests`: All checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.35.zip` (143 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 55: Add Uploader and "Manage Uploads" Popover with Full Imagen AI Parity
- **Implemented Uploader Pill & Manage Uploads Popover Component**:
  - Created [saha-app/src/components/uploader-popover.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/uploader-popover.jsx) implementing:
    - **UploaderPill**:
      - Idle state: `↑ Uploader`.
      - Active state: `↑ Uploading ${activeCount}/${totalCount} ○` with animated spinner.
      - Updating state: `↑ Updating ○`.
    - **ManageUploadsPopover**:
      - Header: "Manage uploads" with maximize/collapse `⤢` and close `×`.
      - Blue Tip Banner: "💡 Adobe DNG Converter compresses your photos during upload" with dismissable "Don't show again" (persisted to localStorage).
      - Summary Row: `${N} project | ${total} photos | Time left` with `Pause all` / `Resume all` and `Dismiss completed`.
      - Active Cards: Live project name, status spinner / completion checkmark, photo progress e.g. `Upload project 21/21 photos`, emerald green progress bar, and `Cancel upload` button.
      - Package Updating: "We're updating your packages" progress card matching reference screenshot.
      - Empty State: Centered folder upload icon with "You will see all project uploads here".
- **Top Bar Integration**:
  - In [saha-app/src/components/shell.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/shell.jsx):
    - Added the `UploaderPill` and `ManageUploadsPopover` to `TitleBar`.
    - Added matching top bar controls from reference: Notification bell `🔔`, `🎧 Help`, about question mark `(?)`, and user avatar badge (`VR`).
    - Passed uploader props through `AppShell` to make the Uploader globally accessible across all views (Home, Projects, AI Profiles).
- **Global Lifecycle & Project Creation Integration**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Added global state: `uploads`, `isUpdating`, `updatingProgress`.
    - Connected `handleCreateProject`: automatically registers newly created projects in the uploader queue, starts real-time photo progress tracking, calculates remaining time, and transitions to completed upon finishing.
    - Added `handleCancelUpload`, `handlePauseAll`, `handleResumeAll`, and `handleDismissCompleted`.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 46 modules transformed in 525ms with 0 errors (bundle: 297.77 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 9.81s with 0 failures.
  - `ruff check src scripts tests`: All checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.36.zip` (144 files, 211.4 MB zip / 229.3 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 56: Project Creation Workflow & Live Photo Editing Workspace Parity with Imagen AI
- **Project Creation & Payload Fix**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Fixed `handleStart` to pass complete metadata: `name`, `projectName`, `sourcePath`, `folderPath`, and `rawCount` (resolving undefined photo count issues).
- **Dedicated Project Inspection & Photo Workspace**:
  - Created [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx) matching reference `media_1794864635431_f92b20ea.jpg`:
    - Top breadcrumb bar: `Projects / <ProjectName>` with actions `Actions ▾`, `Edit project again ▾`, `Export ▾`.
    - Left sort & filter sidebar: `Sort by: Capture date ▾`, `Filters (Clear all)`, `Collections`, `Camera type`, `Stars`, `Colors`, and `Flags`.
    - Photo grid: displays all shoot photos with top-right overlay badge `Editing...` and pulsing indicator while processing, switching to `✓ Edited` upon completion.
    - Bottom toolbar: pill with `• Editing in progress` (animated pulse), view switchers, zoom slider, photo count (`21 photos • 21 total`), and `See original`.
    - Right tools rail: vertical icon bar for `Edit`, `Crop`, `Masks`, `Cleanup`, `Retouch`, `Presets`, `Versions`, `History`, and `Info`.
- **Card Status Indicators & Navigation**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Added `onOpenProject` prop and updated project card clicks to open the workspace.
    - Added live status line: blue pulsing dot with `Processing` while editing, and green dot with `Complete` when done, matching `media_1794864748523_90932ae2.jpg`.
  - In [saha-app/src/components/home-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/home-view.jsx):
    - Added `onOpenProject` prop and updated project cards to open the project workspace.
    - Added matching `• Processing` / `• Complete` status line.
- **Global Projects Lifecycle & Persistence**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Added persistent `projectsList` initialized from `localStorage.getItem('saha_user_projects')` with default `Sample_Raw` fallback.
    - In `handleCreateProject`: prepends new project to `projectsList`, opens the `project-detail` workspace immediately, registers with Uploader, and updates project status to `complete` when simulated/real processing completes.
    - Connected `onOpenProject` across Home and Projects views.
- **Verification**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 553ms with 0 errors (bundle: 316.54 kB).
  - `pytest tests/api/test_profiles.py tests/test_inference_v2.py tests/api/test_process_route.py`: 49 passed in 1.10s with 0 failures.
  - `ruff check src scripts tests`: All checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.37.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 57: Fix Double-Click Windows Launcher Startup Failure & Zombie Port Collisions
- **Root Cause Analysis**:
  - Double-clicking `DOUBLE_CLICK_WIN_START_SAHA.bat` previously failed when a previous session or crash left detached background processes (`node.exe` or `python.exe`) holding port `5173` (Vite) or `8765` (FastAPI backend).
  - Because `saha-app/vite.config.js` sets `strictPort: true`, Vite failed immediately with `Error: Port 5173 is already in use` and exited with error code 1. `concurrently -k` then terminated Electron, preventing the desktop window from ever opening.
  - In `saha-app/electron/main.js`, `killBackend()` called `proc.kill('SIGTERM')` on exit, which on Windows does not terminate child process trees (such as Uvicorn subprocesses), causing backend instances to linger as zombies.
  - In `saha-app/electron/main.js`, `getBackendCommand()` referenced `uv.cmd` on Windows instead of resolving `uv.exe`.
- **Automated Port & Process Self-Healing**:
  - In [DOUBLE_CLICK_WIN_START_SAHA.bat](file:///c:/Users/ASUS/Projects/SonnaEditor/DOUBLE_CLICK_WIN_START_SAHA.bat):
    - Added automated port pre-clearing: queries ports 5173 and 8765 via PowerShell and force-terminates any orphaned processes before launching.
    - Added user `.cargo\bin` to PATH discovery.
    - Prioritized `.venv\Scripts\python.exe` directly when it exists to skip redundant `uv` resolution.
  - In [scripts/run_app.py](file:///c:/Users/ASUS/Projects/SonnaEditor/scripts/run_app.py):
    - Added `_free_stale_ports([5173, 8765])` helper that terminates lingering zombie processes holding Vite or backend API ports before launching and in a `finally` block on application exit.
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Replaced `uv.cmd` with direct detection of `uv.exe` in `%USERPROFILE%\.local\bin\uv.exe` or system PATH.
    - Enhanced `killBackend()` on Windows to invoke `taskkill /pid <PID> /T /F`, guaranteeing clean termination of the entire Python/Uvicorn process tree when the window is closed.
  - In [DOUBLE_CLICK_MAC_START_SAHA.command](file:///c:/Users/ASUS/Projects/SonnaEditor/DOUBLE_CLICK_MAC_START_SAHA.command):
    - Added matching port pre-clearing for ports 5173 and 8765 using `lsof -ti`.
- **Verification**:
  - Verified `cmd.exe /c DOUBLE_CLICK_WIN_START_SAHA.bat`: cleared lingering ports, booted Vite in 282ms, spawned backend on port 8765, and connected Electron cleanly.
  - Full pytest regression suite: 49 passed in 1.12s.
  - Ruff linting: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.38.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 58: Fix `currentJob` ReferenceError & Cold-Start Backend Timeout
- **Root Cause Analysis**:
  - In [saha-app/src/components/home-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/home-view.jsx), lines 134-166 checked `currentJob && currentJob.state === 'running'` to render the active job progress banner, but `currentJob` was never declared or extracted from `projects`, throwing `Uncaught ReferenceError: currentJob is not defined`. This caused React to crash on startup with a blank window.
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js), `backendStartupTimeoutMs()` defaulted to 30s (`30000`), which on cold Windows starts (when Windows Defender scans `torch.dll` and Uvicorn initializes) can trigger a premature timeout dialog ("Saha — backend failed to start") even though the server is healthy and just finishing its boot cycle.
- **Fixes Applied**:
  - In [saha-app/src/components/home-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/home-view.jsx):
    - Added `const currentJob = projects?.currentJob || null;` to safely resolve job state without throwing errors.
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Increased `backendStartupTimeoutMs()` default from 30s to 60s (`60000`) to provide ample headroom on cold boots and heavily loaded machines.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Added a top-level `<ErrorBoundary>` component with a user-facing retry button so that any isolated component render error displays diagnostic info gracefully rather than unmounting the entire application.
- **Verification**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 523ms with 0 errors (bundle: 317.45 kB).
  - Pytest regression suite: 49/49 passed in 1.30s.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.39.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 59: Fix CSSStyleDeclaration Indexed Property Setter Error
- **Root Cause Analysis**:
  - In [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx) and [saha-app/src/components/uploader-popover.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/uploader-popover.jsx), `M` was declared as a string (`const M = SONNA.mono;`).
  - Spreading a string inside JSX style props (e.g. `style={{ ...M }}`) unpacked character indices as numeric properties (`{ 0: 'u', 1: 'i', ... }`).
  - When React attempted to assign numeric indexed properties on `CSSStyleDeclaration` (`style[0] = 'u'`), the browser engine threw:
    `TypeError: Failed to set an indexed property [0] on 'CSSStyleDeclaration': Indexed property setter is not supported.`
- **Fixes Applied**:
  - In [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx):
    - Declared `const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };`.
    - Replaced all 6 invalid `...M` spreads with `...Tnum` and `fontFamily: M`.
  - In [saha-app/src/components/uploader-popover.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/uploader-popover.jsx):
    - Declared `const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };`.
    - Replaced both invalid `...M` spreads with `...Tnum`.
  - Audited all JSX files across `saha-app/src` using AST regex scanner; confirmed 0 invalid spreads remain.
- **Verification**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 533ms with 0 errors (bundle: 317.56 kB).
  - Pytest regression suite: 49/49 passed in 1.02s.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.40.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.
