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


### Task 60: Catalog Upload Sequence & Imagen AI Workflow Parity
- **End-to-End Catalog Ingestion Flow Matching Reference Screenshots**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - **Screen 1a (`catalog-source`)**: Lightroom Classic catalog selector dropdown (`My Lightroom catalogs ▾`), catalog path display box (`Lightroom Catalog.lrcat`), and `🔍 Browse catalog` button.
    - **Screen 1b (`choose-photos`)**: Folder navigation table displaying detected folder (`Train_sample`), 5 total photos, 5 filtered photos, project date `Feb 10, 2026`, and filter toolbar (Flag status `⚐`/`×`/None, Star ratings `☆ 1-5`, 6 color label dots, and `RAW`/`JPEG` toggle).
    - **Screen 2 (`setup`)**: Pre-filled project name (`Train_sample`), 10 photography genre cards.
    - **Screen 3 (`workflow`)**: Dual card layout with `AI culling` toggle (`2 Free projects` badge) and `AI Profile` card with selected profile (`WARM SKIN TONES`), format badges (`RAW`, `Color`, `Talent`), and `Adjust` / `Change ▾` dropdown.
    - **Screen 4 (`tools`)**: Streamlined Crop & Straighten layout with 210px sidebar width, non-wrapping `2 active` pill tag, removed aspect ratio options from Crop, and tight 6px card gap.
- **Projects Table/List View & Live Upload States**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Added List View table with columns: `Project name`, `Profile`, `Status`, `Photos`, `Date`, `Cloud`, `Next step`, `⋮`.
    - Added search bar `Search for a project 🔍` and view toggle between Table/List and Grid.
    - Added live status transitions: `• Uploading photos · ${progress}%`, `• Editing` with progress bar, and `• Ready for review` with green dot.
    - Added coral button `Review edits ▾` in the Next Step column that opens the project workspace.
- **Global Uploader Popover & Workspace Synchronization**:
  - In [saha-app/src/components/uploader-popover.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/uploader-popover.jsx):
    - Supports both in-progress upload state (`○ Upload for editing 5 photos` with progress bar & `Cancel upload` button) and completed state (`✓ Upload for editing Upload complete. You uploaded 5 photos.` with `Dismiss` button).
    - Preserved blue Adobe DNG Converter banner and empty state illustration.
  - In [saha-app/src/components/shell.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/shell.jsx), [saha-app/src/components/home-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/home-view.jsx), and [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx):
    - Connected `isUploadsOpen` and `onToggleUploadsOpen` across all views so clicking the top bar pill opens/closes the popover globally.
    - Updated `ProjectDetailView` to show `Editing...` overlay pills on photos and `• Editing in progress` bottom pill while editing.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Wired `handleCreateProject` to open Projects view with Uploader popover open, simulating upload progression `0% -> 100%`, transitioning to `editing`, and finally to `Ready for review`.
- **Backend Spawn & Launch Resiliency**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Added multi-candidate `resolveRepoRoot()` and Python lookup in `getBackendCommand()`, preventing path resolution failures and timeouts on cold start.
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 735ms with 0 errors (bundle: 341.61 kB).
  - Pytest regression suite (`test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`, `test_process_route.py`): 68 passed in 5.68s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.41.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 61: Review in Adobe Selection & Edit Metadata Export Lifecycle
- **"Review edits ▾" Popover Dropdown (Image 1 Parity)**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Added an action dropdown anchored directly below the coral `Review edits ▾` button.
    - **Row 1 (`Review in Adobe`)**: Accompanied by tray download SVG icon; triggers the `Download edits` modal.
    - **Row 2 (`Review in Saha`)**: Accompanied by eye inspection SVG icon; opens the in-app photo grid workspace (`ProjectDetailView`).
    - Added outside-click dismiss listener to close the popover cleanly when clicking elsewhere.
- **"Download edits" Modal (Image 2 Parity)**:
  - Implemented the centered `Download edits` modal dialog:
    - **"Are your photos still here?"**: Text input displaying the project directory path (`C:\Users\ASUS\Downloads\Sample_Raw`), subtext guidance, and a `Browse` button that invokes Electron native folder dialog via `window.saha.pickFolder()`.
    - **"Download options"**: Custom radio selectors for `(●) Edits and ratings` (default selected) and `(○) Edits only`.
    - **Blue Info Banner**: `💡 Downloading Saha edits replaces existing edits.` with lightbulb icon.
    - **Footer**: Selected photo count indicator (`${photos} photos selected • Local offline editing`), and coral CTA button `Download edits`.
- **"Downloading edit metadata" Modal (Image 3 & Image 4 Parity)**:
  - **In-Progress State (Image 3)**:
    - Card displaying animated spinner `○`, project name, `Reading metadata. This might take a while.`, and emerald progress bar animating from `0.0%` to `100.0%`.
    - Footer caption: `Downloading can take time. Keep the app open for now.`
  - **Completed State (Image 4)**:
    - Dismissable emerald green banner: `✓ Download Complete! ${photos} photos were downloaded.`
    - Summary card with green checkmark badge: `✓ ${name}` and `${photos} photos`.
    - Modal actions: `Close` button, `🗁 Open folder` button (invokes `window.saha.revealPath()` to open the folder in Windows Explorer / Finder), and coral `Next` button.
- **Post-Export State Transition & Training Pipeline Linkage**:
  - Clicking `Next` in the completed modal calls `onUpdateProjectStatus(projectId, 'fine_tune')`.
  - In the Projects table, the project transitions to:
    - **Status column**: `● Fine-tune profile` with amber dot (`#F59E0B`).
    - **Next step column**: Coral button `Train profile` (`#E05353`), which launches the Personal AI profile training wizard (`isPersonalWizardOpen`).
  - Supports direct `Review edits in Adobe` single-click button for shoots configured in Adobe review mode (e.g. `Train_sampleee`).
- **Global Projects Synchronization in [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx)**:
  - Added `handleUpdateProjectStatus` callback to update and persist project lifecycle transitions in `projectsList` and `localStorage`.
  - Initialized default projects matching user reference screenshots (`Train_sampleee` with 5 photos and `Sample_Raw` with 21 photos).
- **Packaging & Verification**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 660ms with 0 errors (bundle: 356.89 kB).
  - Pytest regression suite (`test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`, `test_process_route.py`): 68 passed in 6.93s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.42.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - No automatic git commit executed per user rules.


### Task 62: Fix Tnum ReferenceError in ProjectsView & Runtime Resiliency
- **Tnum Declaration Resolution**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Declared `const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };` at the module header, bringing it into parity with all other view components (`home-view.jsx`, `uploader-popover.jsx`, `project-detail-view.jsx`).
    - Fixed runtime `ReferenceError: Tnum is not defined` when rendering the projects list and modal progress meters.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 645ms with 0 errors (bundle: 356.85 kB).
  - Pytest regression suite (`test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`, `test_process_route.py`): 68 passed in 1.40s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.43.zip` (145 files, 211.4 MB zip / 229.4 MB uncompressed).
  - Cleaned up prior archive `saha-v1.42.zip`.
  - No automatic git commit executed per user rules.


### Task 63: Remove Mock Samples, Add Real Catalog Collections & File Size Parity
- **Sample Cleanup Across Frontend**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Removed hardcoded default projects (`Train_sampleee`, `Sample_Raw`).
    - Added sanitizer in `projectsList` initialization to filter out any mock sample projects cached in `localStorage` (`proj-train-sample`, `proj-sample-raw`). Defaults to empty list (`[]`).
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Removed `Sample_Shoots.lrcat` and mock `f-sample-raw` / `Train_sample` sample defaults.
    - Initialized `projectName` to `''`, `checkedFolderIds` to `new Set()`, `checkedCollectionIds` to `new Set()`, and `detectedPhotoCount` to `0`.
    - Updated Project Name input placeholder in Step 2 to `e.g. Wedding Shoot`.
  - In [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx):
    - Replaced breadcrumb fallback `'Sample_Raw'` with `'Untitled Project'`.
- **Remove "Restore defaults"**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Completely removed the "Restore defaults" button and wrapper from the filter toolbar.
- **Accurate Dynamic Catalog File Size**:
  - In [src/sonna_editor/data/catalog.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/data/catalog.py):
    - Implemented `format_catalog_size(size_bytes)` matching binary truncation/formatting to single decimal place (e.g. `2084864` bytes -> `1.9 MB`, matching Imagen's exact file size display).
    - Added `inspect_catalog(catalog_path: Path)` returning `size_bytes`, `size_formatted`, `folders`, and `collections`.
  - In [src/sonna_editor/api/models.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/models.py) & [src/sonna_editor/api/routes/folders.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/routes/folders.py):
    - Added `catalog_size_bytes`, `catalog_size_formatted`, `folders`, and `collections` fields to `FolderScanResponse`.
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js) & [saha-app/electron/preload.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/preload.js):
    - Added `saha:get-file-stat` IPC handler and exposed `getFileStat` on `window.saha` to read filesystem stats directly.
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Replaced hardcoded `'1.9 MB'` / `'3.4 MB'` sizes with dynamic formatted sizes from catalog inspection.
    - Wired ↻ refresh button in catalog header to re-scan and update catalog size, folders, and collections.
- **Dedicated Collections View & Selection**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - When `Collections` tab is selected:
      - Searches filter collections by name (`placeholder="Search for a collection name"`).
      - Table header displays `Collection name` with collections count and photos total (`Collections: X  Total photos: Y`).
      - Rows display Lightroom collection icon, collection name, total photos, filtered photos, and date.
      - Checkboxes track collection selections independently via `checkedCollectionIds`.
      - Advancing to Step 2 pre-populates project name and photo count from the selected collection.
    - When `Folders` tab is selected:
      - Searches filter folders by path/name (`placeholder="Search for a folder name or a path"`).
      - Rows display folder path with folders count and photos total.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 591ms with 0 errors (bundle: 361.25 kB).
  - Pytest suite (`test_folders.py`, `test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`): 67 passed in 1.64s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.44.zip` (145 files, 211.4 MB zip / 229.5 MB uncompressed).
  - Cleaned up prior archive `saha-v1.43.zip`.
  - No automatic git commit executed per user rules.


### Task 64: Eliminate Cold-Start Catalog Preload & Gate Catalog Selection
- **Deferred Catalog Loading**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Changed initial state for `selectedCatalog` to `null` and `sourcePath` to `''`.
    - Removed automatic cold-start scanning and auto-selection of default catalog on modal open.
    - Suppressed rendering of the catalog path box (`selectedCatalog?.path`) until a catalog is explicitly selected.
    - Initialized `detectedPhotoCount` to `0`, ensuring no false "5 Photos selected" banner appears on cold start.
    - Gated the "Next" CTA on the `catalog-source` step (`disabled={!selectedCatalog}`) so users cannot advance without selecting a catalog.
- **On-Demand Catalog Selection via Dropdown & Browse**:
  - Clicking the coral `My Lightroom catalogs ▾` button opens the dropdown menu with available detected catalogs (`knownCatalogs`).
  - Clicking a catalog from the dropdown scans its contents (folders, collections, size) and transitions to `choose-photos`.
  - Added `addKnownCatalog` to automatically record and persist user-selected catalogs in `localStorage` (`saha_known_catalogs`) when picking via `Browse catalog` or Drag & Drop.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 684ms with 0 errors (bundle: 361.42 kB).
  - Pytest suite (`test_folders.py`, `test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`): 67 passed in 1.61s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.45.zip` (145 files, 211.4 MB zip / 229.5 MB uncompressed).
  - Cleaned up prior archive `saha-v1.44.zip`.
  - No automatic git commit executed per user rules.


### Task 65: Resolve Backend Startup Hang, Clean Top Nav & Align Modal Workflow with Imagen
- **Resolve Backend Startup Hang on Launch / Restart**:
  - **Root Cause Diagnosis**: When restarting Saha or double-clicking the launcher after a previous run, port 8765 remained held in `FinWait1` by an orphaned Python server process (PID 40864). Because Electron's `main.js` previously only tested `probeHealth`, when the probe failed, it attempted to spawn a new backend that crashed with `[Errno 10048] address already in use`, causing `waitForBackendReady` to time out and display `"Saha — backend failed to start"`.
  - **Fix in [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js)**:
    - Added `freeStalePort(port)` to proactively detect and terminate any zombie process holding port 8765 or executing `scripts/serve.py` before spawning the backend.
    - Upgraded `killBackend()` on Windows to use synchronous process tree termination via `spawnSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'])` followed by `freeStalePort(PORT)` to ensure zero orphaned processes.
    - Added `app.on('before-quit')` lifecycle hook to guarantee child process cleanup.
  - **Fix in [DOUBLE_CLICK_WIN_START_SAHA.bat](file:///c:/Users/ASUS/Projects/SonnaEditor/DOUBLE_CLICK_WIN_START_SAHA.bat) & [scripts/run_app.py](file:///c:/Users/ASUS/Projects/SonnaEditor/scripts/run_app.py)**:
    - Enhanced port clearing routines to terminate both port connections and any lingering `scripts/serve.py` python processes.
- **Top Nav Bar Clean-up**:
  - In [saha-app/src/components/shell.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/shell.jsx):
    - Removed notification bell button (`title="Notifications"`).
    - Removed help button (`title="Help & Support"`).
    - Removed question mark info button (`title="About Saha"`).
    - Removed user initials avatar pill (`title="Saha Studio User"` / "VR").
    - Preserved `UploaderPill`, theme toggle button (`onToggleTheme`), and logout button.
- **Modal Workflow Parity with Imagen (Reference Images 2-5)**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Suppressed the modal footer completely on the `catalog-source` step (`step === 'source' && catalogSubStep === 'catalog-source'`), matching Imagen Image 2 where no bottom bar is rendered.
    - Selecting a catalog now resets folder/collection checkboxes to empty `new Set()` and photo count to `0`.
    - In `choose-photos`, the footer now displays `0 photos selected` with `Back` and `Next` (disabled until a folder is checked), matching Imagen Image 3.
    - Advancing to Step 2 (`setup`), Step 3 (`workflow`), and Step 4 (`tools`) reflects the checked folder name and photo count (`5 photos selected`).
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 590ms with 0 errors (bundle: 358.59 kB).
  - Pytest regression suite (`test_folders.py`, `test_callback_bridge.py`, `test_profiles.py`, `test_inference_v2.py`): 67 passed in 1.63s with 0 failures.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.46.zip` (145 files, 211.4 MB zip / 229.5 MB uncompressed).
  - Cleaned up prior archive `saha-v1.45.zip`.
  - No automatic git commit executed per user rules.


### Task 66: Remove Filter Toolbar, Implement Auto-Discovery for All Catalogs & Explain Size Display
- **Removal of Filter Toolbar (Stars and Colored Circles)**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Completely removed the second bar / Filter Toolbar (`flags`, `☆1-5 ratings`, `6 color circles`, and `RAW / JPEG` toggle) below the catalog path header.
    - Cleaned up unused state hooks (`flagFilter`, `ratingFilter`, `colorFilter`, `formatFilter`).
- **Comprehensive Lightroom Catalog Discovery System**:
  - In [src/sonna_editor/data/catalog.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/data/catalog.py):
    - Implemented `discover_lightroom_catalogs()`: scans Adobe Lightroom preferences files (`%APPDATA%\Adobe\Lightroom\Preferences\*.agprefs` on Windows, `~/Library/...` on macOS) parsing `libraryToLoad` and `recentLibraries` for all user `.lrcat` files. Also recursively scans standard locations (`Pictures/Lightroom`, `OneDrive/Pictures/Lightroom`, `Documents/Lightroom`, and `recent_folders.json`).
  - In [src/sonna_editor/api/models.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/models.py) & [src/sonna_editor/api/routes/folders.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/routes/folders.py):
    - Added `DiscoveredCatalog` model and `GET /api/catalogs` endpoint returning all detected catalogs with their names, paths, and formatted sizes.
  - In [saha-app/src/api/client.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/api/client.js):
    - Added `listCatalogs` API client method.
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Wired `loadDiscoveredCatalogs` on modal open and on clicking `My Lightroom catalogs ▾`, ensuring all detected catalogs on the user's computer are loaded and selectable in the dropdown.
- **Folder Path Styling & Date Formatting Parity with Imagen**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Split folder path rendering into light grey parent directory (`C:/Users/ASUS/Downloads/`) and bold white folder name (`Train _sample`), matching Imagen Image 3.
    - Added `formatProjectDate` to render dates like `Feb 10, 2026` rather than raw ISO `2026-02-10`.
- **Clarification on Folder vs Catalog Size**:
  - Clarified that `1.9 MB` in both Imagen and Saha is the size of the Lightroom catalog file (`Lightroom Catalog.lrcat` is 2,084,864 bytes = 1.9 MB), whereas `Train _sample` is the folder of 5 CR3 RAW files (~109.8 MB on disk).
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 564ms with 0 errors (bundle: 356.61 kB).
  - Pytest suite: 14 passed in `tests/api/test_folders.py`, fast regression passing.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.47.zip` (145 files, 211.4 MB zip / 229.5 MB uncompressed).
  - Cleaned up prior archive `saha-v1.46.zip`.
  - No automatic git commit executed per user rules.

### Task 67: Parity Implementation of 9-Step Lite Style Survey with Benchmark Photos & Survey Architecture Clarification
- **Why and How Imagen's Lite Style Survey Works**:
  - **The "Why 3 times each" rationale**:
    - A Lightroom preset applied uniformly across a shoot fails on exposure, temperature, and tint because real-world lighting varies dramatically across 3 primary lighting archetypes:
      1. **Scene 1 (High-Key / Backlit by Water)**: Bright sky/water background tests subject face exposure versus highlight clipping.
      2. **Scene 2 (Low-Key / Dim Shadow Indoor)**: Deep shadows and ambient interior light test shadow recovery versus rich moody blacks.
      3. **Scene 3 (Harsh Midday Sun with Saturated Foliage)**: Direct sunlight on bright clothing (bridal gowns) tests highlight blowout and green reflection (chlorophyll bounce) on skin.
    - Testing Exposure, Temperature, and Tint across these 3 contrasting benchmark scenes prevents overfitting to a single lighting situation and captures the photographer's generalized aesthetic bias vector $(\Delta E, \Delta K, \Delta T)$.
  - **Offset Mapping**:
    - Each scene offers 5 graded choices (-2, -1, 0, +1, +2).
    - The average of the 3 scene answers for each slider sets the profile's baseline bias:
      - Exposure bias: `avg(Q1, Q2, Q3) * 0.25 EV`
      - Temperature bias: `avg(Q4, Q5, Q6) * 250K`
      - Tint bias: `avg(Q7, Q8, Q9) * 4 units`
- **Extracted 45 Benchmark Photo Cards into Saha**:
  - Extracted all 45 individual photo cards from Imagen reference screenshots `11_imagen_lite_survey_...` through `19_imagen_lite_survey_...` into [saha-app/public/survey/](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/public/survey/) and [saha-app/src/public/survey/](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/public/survey/).
  - Configured `publicDir: resolve(__dirname, 'public')` in [saha-app/vite.config.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/vite.config.js) ensuring Vite bundles all 45 benchmark images into `dist/survey/`.
- **Full 9-Step Style Survey Implementation in Saha**:
  - In [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx):
    - Replaced the 3 legacy SVG questions with the complete 9-step benchmark survey (`LITE_SURVEY_STEPS`):
      - Q1: Exposure (Backlit lake bride)
      - Q2: Exposure (Dim indoor girl in sweater)
      - Q3: Exposure (Direct sun bride in garden)
      - Q4: Temperature (Backlit lake bride)
      - Q5: Temperature (Dim indoor girl in sweater)
      - Q6: Temperature (Direct sun bride in garden)
      - Q7: Tint (Backlit lake bride)
      - Q8: Tint (Dim indoor girl in sweater)
      - Q9: Tint (Direct sun bride in garden)
    - Gradient preference bars matching Imagen:
      - Exposure: Dark-to-white gradient line with `Darker` / `Brighter` labels.
      - Temperature: Blue-to-yellow gradient line with `Cooler` / `Warmer` labels.
      - Tint: Green-to-magenta gradient line with `Green` / `Magenta` labels.
    - Card selection:
      - Outline in signature lime-yellow `#D4F49C` with soft glow.
      - Clicking immediately selects the option and auto-advances to the next question after 250ms for steps 1-8.
      - Step 9 reveals the pink/coral `[Build profile]` CTA button.
    - 9-dash progress tracker:
      - Active step highlighted in lime-yellow `#D4F49C`.
      - Allows direct navigation to any previously answered step.
    - Navigation: `[Back]` button from Q2 to Q9 (or back to Setup from Q1).
    - Aggregated survey payload computed on `Build profile` and sent to `POST /api/profiles/lite`.
- **Choose a Source Modal Footer Fix**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Ensured no footer (`Cancel`, `Next`, or photo count) is displayed on the initial "Choose a source" modal for either catalog or folder mode, perfectly matching Imagen Image 2.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 608ms with 0 errors (bundle: 357.88 kB, 45 survey images bundled).
  - Pytest API suite: 37 passed in `tests/api/test_profiles.py` and `tests/api/test_folders.py`.
  - Ruff linter: all checks passed with 0 errors.
  - Packaged distribution archive: `dist/saha-v1.48.zip` (235 files, 212.2 MB zip / 230.3 MB uncompressed).
  - Cleaned up prior archive `saha-v1.47.zip`.
  - No automatic git commit executed per user rules.

### Task 68: Extract Original Ultra-High-Resolution (960x1440) Master Assets for Style Survey
- **Discovered Native Master Assets in Imagen's Application Bundle**:
  - Inspected local Imagen installation on machine (`C:\Program Files\ImagenAI\resources\app.asar`).
  - Parsed the asar binary archive and discovered all 45 uncompressed, ultra-high-definition original WebP master image assets (`Exposure2012_question_*`, `Temperature_question_*`, `Tint_question_*`).
  - Resolution: 960x1440 for Scene 1 (lake bride) and Scene 2 (doorway child), 1440x961 for Scene 3 (garden bride). This provides an 8x resolution increase over the earlier 120x148 screenshot crops (64x more pixels), eliminating all blurriness and visual artifacts.
- **Extracted and Bundled All 45 Master Images**:
  - Extracted all 45 master `.webp` files and generated high-quality 95% `.jpg` fallbacks into both [saha-app/public/survey/](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/public/survey/) and [saha-app/src/public/survey/](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/public/survey/).
  - Updated [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx) to load `survey/${currentSurveyStep.imagePrefix}_opt${optIdx}.webp` natively with automatic `.jpg` fallback.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 673ms with 0 errors (bundle: 358.00 kB).
  - Packaged distribution archive: `dist/saha-v1.49.zip` (325 files, 249.3 MB zip / 267.5 MB uncompressed).
  - Cleaned up prior archive `saha-v1.48.zip`.
  - No automatic git commit executed per user rules.

### Task 69: Deep Cleanup of Unused Dead Code, Duplicate Images, and Stale Artifacts
- **Dead Component and Hook Removal in Frontend**:
  - Deleted `saha-app/src/components/genre-grid.jsx`: early emoji prototype superseded by the modern 10-genre SVG selector in `new-project-modal.jsx`.
  - Deleted `saha-app/src/hooks/useCaptures.js` and `saha-app/src/hooks/useRecentFolders.js`: legacy prototype hooks never imported anywhere in `saha-app/`.
- **Eliminated Duplicate Image Storage and Fallback Redundancies**:
  - Deleted duplicate directory `saha-app/src/public/`: redundant copy created prior to Vite `publicDir` configuration, saving 90 duplicate image files (~11.6 MB).
  - Deleted 45 redundant `.jpg` files from `saha-app/public/survey/`: kept the 45 uncompressed master `.webp` files (33KB-190KB) natively supported by Chromium/Electron.
  - Simplified [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx) to load `survey/${currentSurveyStep.imagePrefix}_opt${optIdx}.webp` directly without unused `.jpg` fallback logic.
- **Backend & Cache Cleanup**:
  - Deleted empty package `src/sonna_editor/profiles/` (contained only an unreferenced 0-byte `__init__.py`).
  - Deleted stale PID file `.saha/jobs/.serve.pid` left over from prior local server runs.
  - Deleted 93 MB stale partial download file `C:\Users\ASUS\.cache\torch\hub\checkpoints\*.partial`.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 602ms with 0 errors (bundle: 357.88 kB, dist survey contains exactly 45 master `.webp` assets).
  - Ruff linter: all checks passed with 0 errors across `src`, `scripts`, `tests`.
  - Packaged clean distribution archive: `dist/saha-v1.50.zip` (186 files, 215.6 MB zip / 233.7 MB uncompressed). Cleaned up 139 redundant files and reduced package size by ~34 MB.
  - Cleaned up prior archive `saha-v1.49.zip`.
  - No automatic git commit executed per user rules.

### Task 70: End-to-End Lite Profile Creation and Style Survey Validation & Pipeline Fix
- **Diagnosed and Resolved Exposure Cancellation Bug in Mode B Inference**:
  - Diagnosed critical bug: In [src/sonna_editor/inference/pipeline.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/inference/pipeline.py), `_mode_b_adjusted_values_for_photo` previously passed `photo_preset` directly to `compute_adjustment`. In `compute_adjustment`, `net_delta = target_delta - base_exp` subtracted out the base exposure. Because `photo_preset["Exposure2012"]` held the preset's exposure plus the survey's offset, adding `base_exp + (target_delta - base_exp)` produced `target_delta`, mathematically stripping the user's preset exposure AND their style survey offset on every single photo!
  - Fixed in [src/sonna_editor/inference/pipeline.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/inference/pipeline.py): `_mode_b_adjusted_values_for_photo` now evaluates scene adaptation against a neutral baseline (`neutral_preset["Exposure2012"] = 0.0`). The resulting adaptive delta `_exposure_delta(image)` is combined with `photo_preset["Exposure2012"]`, preserving 100% of the preset's look and the user's survey choice while adaptively lifting dark frames and pulling back bright frames. Added upper-tone percentile headroom protection (`p98`) so positive exposure style does not blow highlights.
- **Diagnosed and Resolved White Balance Style Survey Truncation**:
  - Diagnosed truncation bug: `_MODE_B_MAX_PRESET_TEMP_STYLE_DELTA` was hardcoded to 300.0K and `_MODE_B_MAX_PRESET_TINT_STYLE_DELTA` was hardcoded to 5.0. This clamped survey answers ±1 (500K) and ±2 (1000K) to the identical 300K, and Tint answers ±1 (5.0) and ±2 (10.0) to the identical 5.0.
  - Refined in [src/sonna_editor/inference/pipeline.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/inference/pipeline.py): `_survey_adjusted_preset` records `_survey_offsets`, and `_anchor_mode_b_wb_to_as_shot` separates accidental preset WB from the user's explicit Style Survey offset. The preset's raw WB is bounded by `±300K` and `±5 Tint` to prevent accidental preset camera WB from creating casts, while the user's intentional survey calibration offset (up to `±1000K` and `±10 Tint`) is fully applied on top of the camera's As-Shot WB.
- **Comprehensive End-to-End Validation Suite**:
  - Authored [tests/test_lite_profile_survey_validation.py](file:///c:/Users/ASUS/Projects/SonnaEditor/tests/test_lite_profile_survey_validation.py):
    - Tests survey math across all answer tiers (-2..+2).
    - Creates 3 complete Lite profiles end-to-end (Bright/Warm/Green, Neutral, Dark/Cool/Magenta).
    - Validates checkpoint creation, sidecar metadata, and bias vector initialization.
    - Validates per-photo inference on test images: Exposure differential is exactly +1.0 EV vs -1.0 EV, Temperature differential is exactly +1000K vs -1000K, Tint differential is exactly -10 vs +10 units.
    - Confirms all 144 non-survey preset sliders (Tone Curves, HSL, Color Grading, Clarity, Vibrance, Shadows, Highlights) are 100% faithful to the preset.
    - Confirms generated XMP sidecars on disk contain the exact values.
- **Verification & Packaging**:
  - Full pytest suite: **797 passed, 45 skipped in 240.59s (4 min)** with 0 failures and 0 errors.
  - Ruff linter: all checks passed with 0 errors across `src`, `scripts`, `tests`.
  - Vite build: passed in 578ms.
  - Packaged clean distribution archive: `dist/saha-v1.51.zip` (186 files, 215.6 MB zip / 233.7 MB uncompressed).
  - Cleaned up prior archive `saha-v1.50.zip`.
  - No automatic git commit executed per user rules.

### Task 71: Centralized Prominent Terminal Error Reflection Across Frontend & Backend
- **Electron Main Process Centralized Error Streaming**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Added process-level traps for `uncaughtException` and `unhandledRejection`, formatting them in bold red with `[electron ERROR]`.
    - Added line-buffered stream parser `pipeBackendStream`: all lines from backend stderr or lines matching `ERROR`, `CRITICAL`, `Traceback`, `Exception`, or `Failed` are prominently formatted with bold red `[backend ERROR]`, while warnings are formatted with yellow `[backend WARN]`.
    - Updated `console-message` forwarding: inspects message severity level and text. Any console error or unhandled exception is formatted with bold red `[renderer ERROR] <message> (<source>:<line>)`.
    - Added webContents crash and failure handlers (`render-process-gone`, `did-fail-load`, `unresponsive`) tagging terminal output with `[renderer CRASH]` and `[renderer LOAD ERROR]`.
- **Frontend Global Error Capture**:
  - In [saha-app/src/index.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.jsx):
    - Added global `window.addEventListener('error')` and `window.addEventListener('unhandledrejection')` listeners so no uncaught JavaScript exceptions or rejected promises are swallowed silently.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Updated `ErrorBoundary` to log with `[React ErrorBoundary Caught Error]`.
  - In [saha-app/src/api/client.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/api/client.js):
    - Added prominent error logging: logs `[API NETWORK ERROR]` on network/backend connectivity loss and `[API ERROR]` on non-2xx HTTP responses with status and detail.
- **Launcher Error Detection**:
  - In [scripts/run_app.py](file:///c:/Users/ASUS/Projects/SonnaEditor/scripts/run_app.py):
    - Added process error logging: emits bold red `[saha ERROR] Application process exited with code <exit_code>` upon non-zero exit, and cleanly traps launcher exceptions.
- **Verification & Packaging**:
  - `npm --prefix saha-app run build:vite`: 47 modules transformed in 607ms with 0 errors (bundle: 358.56 kB).
  - Pytest API & validation suites: 92 passed in 7.33s.
  - Ruff linter: all checks passed with 0 errors across `src`, `scripts`, `tests`.
  - Packaged distribution bundle: `dist/saha-v1.52.zip` (186 files, 215.6 MB zip / 233.7 MB uncompressed).
  - Cleaned up prior archive `saha-v1.51.zip`.
  - No automatic git commit executed per user rules.

### Task 72: Fractional Precision Style Survey Calibration for Lite Profile Creation
- **Backend Continuous Survey Answer Normalization**:
  - In [src/sonna_editor/mode_b/survey.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/mode_b/survey.py):
    - Updated `compute_offset` to accept `float | int` answers in the range `[-2.0, 2.0]`. Computes continuous, linear slider offsets (e.g. answer `1.333` on Exposure yields `+0.6665` stops EV offset).
    - Updated `normalise_lite_answers` to accept and validate floating-point answers. Retains exact integer answers (e.g. `0`, `1`, `-2`) for backwards compatibility while preserving fractional precision up to 4 decimal places.
    - Updated `build_survey_payload` type signature to `answers: dict[str, float | int]`.
  - In [src/sonna_editor/api/models.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/models.py):
    - Updated `LiteProfileRequest.survey_answers` type annotation to `dict[str, float | int]` so fractional averages from the frontend are parsed without integer truncation.
  - In [src/sonna_editor/api/routes/profiles.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/routes/profiles.py):
    - Updated `_validate_survey_answers` to handle `dict[str, float | int]`.
- **Frontend High-Precision Answer Aggregation**:
  - In [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx):
    - Replaced `Math.round(mean)` with `Number((mean).toFixed(3))`.
    - If a user chooses mixed preferences across the 3 benchmark scenes (e.g. +1, +2, +1), the survey calculates `1.333` instead of truncating down to `1`.
    - Zero UI impact: preserves the identical 9-step visual cards and layout without adding any extra buttons, clutter, or visual complexity.
- **Test Suite Updates**:
  - In [tests/test_style_survey.py](file:///c:/Users/ASUS/Projects/SonnaEditor/tests/test_style_survey.py):
    - Added `test_compute_offset_fractional` verifying linear scaling for +1.333 and -0.667.
    - Added `test_compute_offset_rejects_out_of_bounds` verifying boundary checking for values outside `[-2, +2]`.
    - Added `test_normalise_lite_answers_fractional_precision` and `test_normalise_lite_answers_rejects_out_of_bounds`.
  - In [tests/test_lite_profile_survey_validation.py](file:///c:/Users/ASUS/Projects/SonnaEditor/tests/test_lite_profile_survey_validation.py):
    - Added `test_lite_profile_fractional_survey_precision` verifying end-to-end profile creation and shoot inference with fractional answers (`exposure=1.333`, `temperature=-0.667`, `tint=0.5`).
- **Packaging & Verification**:
  - Full pytest suite: **812 passed, 45 skipped in 195.71s (3 min 15s)** with 0 failures and 0 errors.
  - Ruff linter: all checks passed with 0 errors across `src`, `scripts`, `tests`.
  - Vite build: 47 modules transformed in 560ms with 0 errors.
  - Packaged distribution bundle: `dist/saha-v1.53.zip` (194 files, 426.1 MB zip / 460.8 MB uncompressed).
  - Cleaned up prior archive `saha-v1.52.zip`.
  - No automatic git commit executed per user rules.

### Task 73: End-to-End Validation of Lite Profile Completion and Physical File Generation
- **Live Creation, Artifact Generation, and API Flow Validation**:
  - Validated real profile creation via `POST /api/profiles/lite`:
    - Generated physical checkpoint file `model-v0.4.0.ckpt` (113.5 MB) on disk in `v1_learning/`.
    - Generated metadata sidecar `model-v0.4.0.json` with `profile_type: "mode_b_initial"` and stable slug ID.
    - Verified physical backup copy of source `.xmp` preset and survey `.json` in `v1_learning/`.
    - Confirmed survey JSON accurately serializes continuous fractional offsets (`exposure_offset = 0.6665`, `temperature_offset = -333.5`, `tint_offset = 2.5`).
  - Validated PyTorch model integrity:
    - Successfully instantiated `SonnaEditor` model from the generated checkpoint file.
    - Confirmed output head zero weights and bias vector injection for slider set `v2`, architecture `3`.
  - Validated profile discovery and active status:
    - Confirmed `GET /api/profiles` enumerates the new Lite profile with `Ready to use` status, non-draft state, and correct version `v0.4.0`.
    - Confirmed activation via `POST /api/profiles/{id}/activate` updates `config.ACTIVE_PROFILE_PATH`.
  - Validated shoot inference execution:
    - Verified per-photo scene adaptation with the generated Lite profile: Exposure 0.90 EV, Temperature 4966.5 K, Tint 10.5.
  - Validated clean physical cleanup:
    - `DELETE /api/profiles/{id}` cleanly unlinks all 4 related files (`.ckpt`, `.json`, `-preset.xmp`, `-survey.json`).
- **UI Completion Experience**:
  - Confirmed `LiteProfileWizard` footer button transitions from "Next" to "Build profile" on Step 9, displays "Building…" during API submission, closes cleanly upon success, and triggers `onProfilesChanged` and `setRefreshKey`.
  - Confirmed profile card renders with green `✓ Ready to use` badge and coral `Use to edit` CTA button.
  - Confirmed clicking `Use to edit` activates the profile and launches `NewProjectModal` with the profile pre-selected.
- **Verification**:
  - Ran validation script `validate_lite_profile_flow.py`: all 8 steps passed 100%.
  - No automatic git commit executed per user rules.

### Task 74: End-to-End Validation of Project Creation (Catalog & Folder), Shoot Editing, and Lightroom Classic Before/After
- **Catalog & Folder Project Ingestion Validation**:
  - Validated real Lightroom Classic catalog scanning on `C:\Users\ASUS\OneDrive\Pictures\Lightroom\Lightroom Catalog.lrcat` (1.9 MB):
    - Opened read-only with SQLite `PRAGMA query_only = ON;`.
    - Discovered referenced folder `C:\Users\ASUS\Downloads\Train _sample` with 5 CR3 RAW files.
    - Verified `POST /api/folders/scan` returns 200 with `is_valid: true`, `raw_count: 5`, `source_type: "catalog"`.
  - Validated direct folder scanning on `C:\Users\ASUS\Downloads\Train _sample`:
    - Verified `POST /api/folders/scan` returns 200 with `is_valid: true`, `raw_count: 5`, `source_type: "folder"`.
- **Shoot Processing & XMP Sidecar Generation**:
  - Ran processing pipeline on real Canon `.CR3` RAW files using an active profile:
    - Previews extracted, per-photo adaptive exposure computed, and As-Shot camera WB anchored.
    - Verified generation of Adobe Lightroom compliant XMP sidecars with `ProcessVersion="15.4"`.
- **Lightroom Classic Before/After (`\`) Compatibility**:
  - Embedded "Pre-Saha" snapshot:
    - Every generated XMP includes a `<crs:Snapshots>` block with `crs:Name="Pre-Saha"` and `crs:Type="Develop"`, capturing the original unedited RAW identity state with exact camera As-Shot White Balance.
    - Enables instantaneous before/after comparison directly in Lightroom Classic's Snapshots panel and via Develop History.
  - Lightroom Classic Bridge package (`SahaBridge.lrplugin`):
    - Verified automatic generation of `sonna_lightroom_edits.lua` alongside edits.
    - Allows direct catalog develop settings import via `Library -> Plug-in Extras -> Import Saha Edits...` without requiring catalog restarts.
    - Develop history step `Apply Saha edits` enables native Before/After comparison (`\` key) in Lightroom Classic.
- **Verification**:
  - Executed automated validation test `validate_lrcat_and_folder_project_flow.py`: 100% passed across all 5 verification parts.
  - No automatic git commit executed per user rules.

### Task 75: Fix Lite Profile Creation Failure in Running Session & Unblock Profile Listing
- **Diagnosed Exact Root Causes**:
  - **Stale Backend Process**: The desktop app's background backend process was started at 09:42 AM before the fractional float precision schema was updated. When the updated frontend submitted `{exposure: 1.333, ...}`, the stale backend process rejected the request with HTTP 422: `Input should be a valid integer, got a number with a fractional part`, preventing the profile from ever being created.
  - **Frontend Deleted Name Collision**: In [saha-app/src/components/profile-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-view.jsx), `deletedIds` was storing `profile.name` instead of strictly `profile.id`. Any newly created profile whose display name matched a previously deleted test name (such as "test") was being filtered out of `processedServer`.
- **Backend Auto-Reload & Process Refresh**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Added `--reload` flag to `scripts/serve.py` so Uvicorn dev server monitors Python code changes and automatically reloads without requiring manual restarts.
  - Terminated the stale 09:42 AM backend process and started a fresh daemon backend with `--reload` active on port 8765.
- **Frontend Filter & Error Visibility Enhancements**:
  - In [saha-app/src/components/profile-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-view.jsx):
    - Fixed `processedServer` and `executeDelete` to filter and delete strictly by unique `profile.id`, unblocking any re-used human names.
    - Updated `LiteProfileWizard` and `PersonalProfileWizard` `onCreated` callbacks to await `onProfilesChanged` before incrementing `refreshKey`.
  - In [saha-app/src/hooks/useProfiles.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/hooks/useProfiles.js):
    - Updated `refetch()` to return the fetched profile array so caller handlers (`handleProfileCreated`) can immediately detect newly created profiles without receiving `undefined`.
  - In [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx):
    - Added prominent error banner in the modal footer right next to the "Build profile" button so any validation or network issues are immediately visible.
    - Added explicit pre-submission validation confirming the preset path is absolute.
  - Rebuilt Vite bundle: 47 modules transformed in 527ms.
- **Verification**:
  - Live tested `POST http://127.0.0.1:8765/api/profiles/lite` with fractional payload: returned HTTP 200 OK.
  - Tested profile activation and clean deletion through API endpoints.
  - Ran pytest suite: 89 passed in 4.93s (`tests/api/test_profiles.py`, `tests/test_style_survey.py`, `tests/test_lite_profile_survey_validation.py`).
  - Packaged distribution bundle: `dist/saha-v1.54.zip` (194 files, 460.8 MB uncompressed, 426.1 MB zip).
  - No automatic git commit executed per user rules.

### Task 76: Remove AI Culling Card, Adjust Action & Crop Tool from UI & Validate Tool Toggles
- **Removed Unused AI Culling Card & Adjust Action**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Removed top "AI culling" card and toggle switch from Step 3 ("Your workflow").
    - Cleaned up unused `aiCullingEnabled` state.
    - Removed non-functional "Adjust" button from AI Profile card row, preserving the "Change ▾" dropdown.
- **Removed Crop from UI While Preserving Code**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Hidden the Crop card in both `recommended` and `composition` tabs from being rendered in the UI, keeping full component and preview JSX in the repository for future activation.
    - Updated active count to reflect Straighten (`compActiveCount = autoStraighten ? 1 : 0`).
    - Focused Preview column on Straighten with horizon rotation graphic and descriptive subtitle.
- **Wired Real Backend Inference to Project Creation**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Imported `startProcess` from `./api/client.js`.
    - Updated `handleCreateProject` to forward `auto_straighten: Boolean(projectData.autoStraighten)` and `denoise_enabled: Boolean(projectData.denoiseEnabled)` to `POST /api/process`.
- **Validation of Tool Toggles**:
  - Authored comprehensive test suite [tests/test_ai_tools_toggle_validation.py](file:///c:/Users/ASUS/Projects/SonnaEditor/tests/test_ai_tools_toggle_validation.py):
    - Validated that when `auto_straighten=True`: `crs:CropAngle`, `crs:HasCrop="True"`, and Lightroom bridge `sonna_lightroom_edits.lua` CropAngle are strictly applied.
    - Validated that when `auto_straighten=False`: NO `CropAngle`, `HasCrop`, `CropTop`, or `CropLeft` attributes are applied to XMP, predictions JSON, or lua bridge.
    - Verified test suite: 25 passed in 0.33s.

### Task 77: Temporarily Remove Project Detail Workspace Page
- **Removed Project Detail Screen from Navigation**:
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - Disabled rendering of the `ProjectDetailView` workspace screen (`screen === 'project-detail'`), keeping the file [saha-app/src/components/project-detail-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/project-detail-view.jsx) preserved in the repository.
    - Updated `handleOpenProject` to reveal the project's folder in the host file manager via `window.saha?.revealPath` instead of transitioning to the photo inspection screen.
- **Verification**:
  - Rebuilt Vite bundle: 47 modules transformed in 498ms with 0 errors (bundle: 343.64 kB).
  - Packaged distribution bundle: `dist/saha-v1.54.zip` (186 files, 233.7 MB uncompressed, 215.6 MB zip).
  - No automatic git commit executed per user rules.

### Task 78: Enforce Unique Profile Names and Ensure Reliable UI Visibility
- **Backend Unique Name Enforcement**:
  - In [src/sonna_editor/api/routes/profiles.py](file:///c:/Users/ASUS/Projects/SonnaEditor/src/sonna_editor/api/routes/profiles.py):
    - Added `_assert_unique_profile_name(name)` check in both `create_lite_profile` (`POST /api/profiles/lite`) and `_validate_personal_profile_request` (`POST /api/profiles/personal`).
    - Validates against both `display_name` and `name` across all discovered profiles (case-insensitive and whitespace-trimmed).
    - Returns HTTP 400 with detail: `"A profile named '{name}' already exists. Every profile must have a unique name."`
- **Frontend Real-Time Duplicate Name Validation**:
  - In [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx):
    - Passed `profiles` prop into `LiteProfileWizard`.
    - Added reactive `isDuplicateName` check matching existing profile display names or IDs.
    - Renders red border on name input and displays an inline warning banner: `⚠️ A profile named "${trimmedName}" already exists. Every profile must have a unique name.`
    - Blocks `canAdvance`, Enter key submission, and API dispatch when a duplicate name is entered.
  - In [saha-app/src/components/personal-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/personal-wizard.jsx):
    - Passed `profiles` prop into `PersonalProfileWizard`.
    - Added duplicate name detection in `StepName` with red input border and inline warning banner, preventing step advancement.
- **Ensured Created Profiles Always Appear in UI**:
  - In [saha-app/src/components/profile-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-view.jsx):
    - Removed `deletedIds` filtering from `processedServer`. Backend profiles returned by `GET /api/profiles` represent actual checkpoints on disk and are never suppressed by stale client-side `localStorage`.
    - Kept `deletedIds` strictly for local mock drafts.
    - Added duplicate validation to `executeRename` and auto-incrementing unique copy naming in `handleDuplicate`.
  - In [saha-app/src/App.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/App.jsx):
    - In `handleProfileCreated`, purged any stale suppression entry from `saha_deleted_profile_ids` in `localStorage` so the new profile is immediately visible.
    - Automatically activates the newly created profile and syncs `newlyCreatedProfileId` to auto-select it in `NewProjectModal`.
- **Verification**:
  - Authored tests `test_lite_create_rejects_duplicate_profile_name` and `test_personal_create_rejects_duplicate_profile_name` in [tests/api/test_profiles.py](file:///c:/Users/ASUS/Projects/SonnaEditor/tests/api/test_profiles.py).
  - All 25 profile API tests passed in 0.37s.
  - Rebuilt Vite bundle: 47 modules transformed in 508ms (bundle: 345.56 kB).
  - Packaged distribution bundle: `dist/saha-v1.54.zip` (186 files, 233.7 MB uncompressed, 215.6 MB zip).
  - No automatic git commit executed per user rules.

### Task 79: Backend Auto-Recovery Watchdog & Persistent Terminal Reliability
- **Diagnosed Long-Running Connection Drop (`Failed to fetch`)**:
  - When running via terminal (`scripts/run_app.py`) over several hours, if the Python backend on port `8765` exited or was interrupted (e.g. sleep/wake, external server stopped, crash), Electron stayed open with `backend = null` and never restarted it, causing `GET /api/profiles` and `POST /api/profiles/lite` to fail with `Failed to connect to backend: Failed to fetch`.
- **Electron Main Process Self-Healing Watchdog**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Added single-flight `ensureBackendRunning()` recovery routine that probes `/api/health`, clears any stale listener on `:8765`, and respawns `scripts/serve.py`.
    - Added unexpected-exit auto-restart in `spawnBackend()` (`!quitting && !externalBackend`).
    - Added periodic 10s `startBackendWatchdog()` health monitor (recovers even if an external backend dies after startup).
    - Restricted `freeStalePort()` on Windows to `-State Listen` so closing/TIME_WAIT client sockets are never touched.
    - Registered `saha:ensure-backend` IPC handler and set `quitting = true` in `before-quit` to prevent restart during app shutdown.
  - In [saha-app/electron/preload.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/preload.js):
    - Exposed `window.saha.ensureBackend()` to the renderer.
  - In [scripts/run_app.py](file:///c:/Users/ASUS/Projects/SonnaEditor/scripts/run_app.py):
    - Restricted `_free_stale_ports()` on Windows to `-State Listen`.
- **Renderer Transparent Retry**:
  - In [saha-app/src/api/client.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/api/client.js):
    - When `fetch` throws a network error, `request()` invokes `window.saha.ensureBackend()` and retries the request once transparently before raising `ApiError`.
- **Verification**:
  - Verified `node --check` on `main.js` and `preload.js`, `vite build` (537ms), `ruff check`, and `pytest tests/api/test_profiles.py` (25 passed).
  - No automatic git commit executed per user rules.

### Task 80: Re-apply AI Culling Card & Adjust Removal in NewProjectModal
- **Restored Step 3 & Step 4 UI Cleanup**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx) (which had been reverted by an open IDE buffer):
    - Removed the `AI culling` card and `aiCullingEnabled` state from Step 3 ("Your workflow").
    - Removed the `Adjust` button from the AI Profile card row, keeping only `Change ▾`.
    - Re-applied `{false && (...)}` guard around the `Crop` cards in Step 4 (`recommended` and `composition` tabs) while preserving all crop code in the repository, set `compActiveCount = autoStraighten ? 1 : 0`, and defaulted `previewTool` to `'straighten'`.
- **Verification**:
  - Rebuilt Vite bundle cleanly (`npx vite build`).
  - No automatic git commit executed per user rules.

### Task 81: Fix Clipped 3-Dots Action Menu in Projects Table
- **Fixed Table Dropdown Clipping**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Changed the Projects list-view table container from `overflow: 'hidden'` to `overflow: 'visible'` and added `borderTopLeftRadius: 7` / `borderTopRightRadius: 7` to the table header so the `⋮` action menu (`Archive project`) and `Review edits ▾` dropdown on the bottom row are never clipped.
- **Verification**:
  - Rebuilt Vite bundle cleanly (`npx vite build`).
  - No automatic git commit executed per user rules.

### Task 82: Remove Tune Profile Button & Delete References Images Folder
- **Removed Tune Profile Icon Button & Tune Overrides**:
  - In [saha-app/src/components/profile-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-view.jsx):
    - Removed the tuning sliders icon button next to `Use to edit` on each profile row, the `Tune Overrides` option from the 3-dots context menu, and the unused `ProfileTuningModal` / `tuningProfile` state.
- **Removed `references/` Folder**:
  - Deleted the `references/` directory (40 reference screenshots + `README.md`) from the repository per user request.
- **Verification**:
  - `npx vite build` in `saha-app`: 47 modules transformed in 538ms (`dist/assets/index-DOBLor_N.js`, 340.56 kB).
  - No automatic git commit executed per user rules.

### Task 83: Remove RAW and Color Tags from Step 3 AI Profile Row
- **Removed `RAW` and `Color` Badges**:
  - In [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx):
    - Removed the `RAW` and `Color` tags next to the selected AI Profile name in Step 3 ("Your workflow"), keeping the `Lite` / `Talent` badge.
- **Verification**:
  - `npx vite build` in `saha-app`: 46 modules transformed in 565ms (`dist/assets/index-C26AkrYm.js`, 334.93 kB).
  - No automatic git commit executed per user rules.

### Task 84: Restrict Project Folder Reveal to "Review in Saha" Button Only
- **Removed Row-Click `onOpenProject` Handler**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Removed `onClick={() => onOpenProject?.(p.raw || p)}` and changed `cursor: 'pointer'` to `cursor: 'default'` on the Projects table row container, so clicking anywhere on the row no longer opens the file manager.
    - Preserved `onOpenProject?.(p.raw || p)` on the **Review in Saha** menu button.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 88: Classify Backend `INFO:` / `WARNING:` Logs Properly in Electron
- **Updated `pipeBackendStream` in [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js)**:
  - Extracted `logBackendLine(rawLine, isStderr)` so Uvicorn/Python `INFO:` and `DEBUG:` lines on `stderr` log as `[backend]`, `WARNING:`/`WARN:` lines log as `[backend WARN]`, and actual errors/tracebacks log as `[backend ERROR]`.
- **Verification**:
  - `node --check saha-app/electron/main.js`: passed.
  - No automatic git commit executed per user rules.

### Task 89: Keep Status "Ready for review" & "Review edits ▾" Button (Remove "Train profile" Override)
- **Updated Projects Table Status & Next Step in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Treated any existing `fine_tune` status projects as `isReady` so they display **Ready for review** and keep the **Review edits ▾** button instead of switching to **Train profile**.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 90: Make "Review edits ▾" and `⋮` Action Menus Mutually Exclusive
- **Updated Menu Toggle Handlers in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Clicking **Review edits ▾** now closes `cardMenuOpenId` (`setCardMenuOpenId(null)`), and clicking `⋮` now closes `reviewDropdownOpenId` (`setReviewDropdownOpenId(null)`), so only one menu can be open at a time.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 91: Chain Download Edits Flow into Direct Lightroom Classic (LrC) Launch
- **Updated Review in Adobe Workflow in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Clicking **Review in Adobe** starts with the **Download edit metadata** modal (`handleStartDownloadEdits`) featuring Project name, Lightroom catalog / Photos folder with Browse, and collapsible **Selection** (`Folders` / `Collections` table with live `scanFolder` data, excluding the Filters section and trial upsells).
  - Clicking **Download edits** displays the **Note** overwrite confirmation dialog (`"Downloading the edits will overwrite any previous edit of the photos submitted for editing."`) with **"Don't show this again"**, **Cancel**, and **Continue**.
  - Clicking **Continue** runs the **Downloading edit metadata** progress modal and finishes with **Download Complete!** featuring **Close** and **Open LR catalog** (which directly launches Lightroom Classic via `window.saha.openInLightroom` while keeping the project status as **Ready for review**).

### Task 92: Auto-Discover Catalog Folders in Download Modal & Launch `Lightroom.exe` Directly
- **Fixed Multi-Folder Selection & Photo Counts in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Imported `listCatalogs` and updated `handleStartDownloadEdits` to automatically resolve the user's Lightroom catalog (`.lrcat`) via `GET /api/catalogs` when a project was created from a standalone folder, so **Selection** displays **all** catalog folders (e.g., `Sample_Raw` and `Train _sample`) and pre-checks the matching project folder.
  - Mapped `f.total_photos` and `f.filtered_photos` from `scanFolder` so photo counts render accurately instead of `0`.
- **Direct `Lightroom.exe` Launch in [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js)**:
  - Updated `saha:open-lightroom` to prefer launching `Lightroom.exe` (Windows) / `Adobe Lightroom Classic.app` (macOS) directly with the `.lrcat` catalog or folder path instead of relying on `shell.openPath` first.
- **Verification**:
  - `node --check saha-app/electron/main.js` and `npx vite build` in `saha-app`: passed cleanly (`dist/assets/index-BQRQ2A6a.js`, 344.89 kB).
  - No automatic git commit executed per user rules.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 85: Make Electron Window & All App Views Responsive Across Window Sizes
- **Electron Minimum Window Constraints**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js):
    - Lowered `BrowserWindow` `minWidth` from `1100` to `640` and `minHeight` from `720` to `500` so users can resize the app to smaller desktop/split-screen sizes.
- **Global Responsive Grids & Media Queries**:
  - In [saha-app/src/index.html](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.html):
    - Added `.saha-table-scroll`, `.saha-lite-setup-grid`, `.saha-ai-tools-grid`, `.saha-2col-grid`, `.saha-3col-grid`, and `.saha-login-form` / `.saha-login-brand` responsive breakpoints (`820px`, `760px`, `680px`).
- **Component Responsiveness**:
  - [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx): Wrapped header toolbar (`flexWrap: 'wrap'`), made search input flexible, wrapped table in `.saha-table-scroll` (`minWidth: 820`) while keeping `overflow: 'visible'` on the table box, and constrained modal `maxWidth: '94vw'` / `maxHeight: '90vh'`.
  - [saha-app/src/components/profile-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-view.jsx): Wrapped profiles table in `.saha-table-scroll` (`minWidth: 740`).
  - [saha-app/src/components/new-project-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/new-project-modal.jsx): Replaced fixed `minHeight: 600` with `min(520px, 85vh)`, converted Step 2 genres grid to `repeat(auto-fill, minmax(135px, 1fr))`, and applied `.saha-ai-tools-grid` to Step 4.
  - [saha-app/src/components/lite-wizard.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/lite-wizard.jsx), [saha-app/src/components/profile-type-modal.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/profile-type-modal.jsx), [saha-app/src/components/login.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/login.jsx): Applied responsive stacking grids and fluid survey/brand panels.
- **Verification**:
  - `node --check saha-app/electron/main.js` and `npx vite build` in `saha-app`: passed cleanly.
  - No automatic git commit executed per user rules.

### Task 86: Direct Lightroom Classic (LrC) Launch from "Review in Adobe"
- **Replaced Download Edits Flow with Direct LrC Review Modal**:
  - In [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx):
    - Clicking **Review in Adobe** now opens the **"Where do you want to review the photos?"** modal with a single **Lightroom Classic (`LrC`)** option (no unnecessary download step or status switch to `Fine-tune profile`).
    - Clicking **Lightroom Classic** invokes `window.saha.openInLightroom(project.path)`.
- **Electron Lightroom Classic Launcher**:
  - In [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js) and [saha-app/electron/preload.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/preload.js):
    - Added `saha:open-lightroom` IPC handler (`window.saha.openInLightroom`) that opens `.lrcat` catalogs directly with Lightroom Classic via `shell.openPath`, or launches `Lightroom.exe` / `Adobe Lightroom Classic.app` directly for shoot folders.
- **Verification**:
  - `node --check` on `main.js` and `preload.js`, and `npx vite build` in `saha-app`: passed cleanly.
  - No automatic git commit executed per user rules.

### Task 87: Fix Vertical Scrollbar on Table Dropdowns Caused by CSS `overflow-x: auto`
- **Fixed `.saha-table-scroll` Overflow Spec Interaction**:
  - In [saha-app/src/index.html](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/index.html):
    - Changed `.saha-table-scroll` to `overflow: visible` by default (preventing CSS from coercing `overflow-y: visible` into `overflow-y: auto`), and scoped `overflow-x: auto` with `padding-bottom: 56px; margin-bottom: -56px;` to `@media (max-width: 960px)` so both normal and narrow windows show the `Archive project` dropdown cleanly without a vertical scrollbar.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 88: Classify Backend `INFO:` / `WARNING:` Logs Properly in Electron
- **Updated `pipeBackendStream` in [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js)**:
  - Extracted `logBackendLine(rawLine, isStderr)` so Uvicorn/Python `INFO:` and `DEBUG:` lines on `stderr` log as `[backend]`, `WARNING:`/`WARN:` lines log as `[backend WARN]`, and actual errors/tracebacks log as `[backend ERROR]`.
- **Verification**:
  - `node --check saha-app/electron/main.js`: passed.
  - No automatic git commit executed per user rules.

### Task 89: Keep Status "Ready for review" & "Review edits ▾" Button (Remove "Train profile" Override)
- **Updated Projects Table Status & Next Step in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Treated any existing `fine_tune` status projects as `isReady` so they display **Ready for review** and keep the **Review edits ▾** button instead of switching to **Train profile**.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 90: Make "Review edits ▾" and `⋮` Action Menus Mutually Exclusive
- **Updated Menu Toggle Handlers in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Clicking **Review edits ▾** now closes `cardMenuOpenId` (`setCardMenuOpenId(null)`), and clicking `⋮` now closes `reviewDropdownOpenId` (`setReviewDropdownOpenId(null)`), so only one menu can be open at a time.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.

### Task 91: Chain Download Edits Flow into Direct Lightroom Classic (LrC) Launch
- **Updated Review in Adobe Workflow in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Clicking **Review in Adobe** starts with the **Download edit metadata** modal (`handleStartDownloadEdits`) featuring Project name, Lightroom catalog / Photos folder with Browse, and collapsible **Selection** (`Folders` / `Collections` table with live `scanFolder` data, excluding the Filters section and trial upsells).
  - Clicking **Download edits** displays the **Note** overwrite confirmation dialog (`"Downloading the edits will overwrite any previous edit of the photos submitted for editing."`) with **"Don't show this again"**, **Cancel**, and **Continue**.
  - Clicking **Continue** runs the **Downloading edit metadata** progress modal and finishes with **Download Complete!** featuring **Close** and **Open LR catalog** (which directly launches Lightroom Classic via `window.saha.openInLightroom` while keeping the project status as **Ready for review**).

### Task 92: Auto-Discover Catalog Folders in Download Modal & Launch `Lightroom.exe` Directly
- **Fixed Multi-Folder Selection & Photo Counts in [saha-app/src/components/projects-view.jsx](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/src/components/projects-view.jsx)**:
  - Imported `listCatalogs` and updated `handleStartDownloadEdits` to automatically resolve the user's Lightroom catalog (`.lrcat`) via `GET /api/catalogs` when a project was created from a standalone folder, so **Selection** displays **all** catalog folders (e.g., `Sample_Raw` and `Train _sample`) and pre-checks the matching project folder.
  - Mapped `f.total_photos` and `f.filtered_photos` from `scanFolder` so photo counts render accurately instead of `0`.
- **Direct `Lightroom.exe` Launch in [saha-app/electron/main.js](file:///c:/Users/ASUS/Projects/SonnaEditor/saha-app/electron/main.js)**:
  - Updated `saha:open-lightroom` to prefer launching `Lightroom.exe` (Windows) / `Adobe Lightroom Classic.app` (macOS) directly with the `.lrcat` catalog or folder path instead of relying on `shell.openPath` first.
- **Verification**:
  - `node --check saha-app/electron/main.js` and `npx vite build` in `saha-app`: passed cleanly (`dist/assets/index-BQRQ2A6a.js`, 344.89 kB).
  - No automatic git commit executed per user rules.
- **Verification**:
  - `npx vite build` in `saha-app`: rebuilt cleanly.
  - No automatic git commit executed per user rules.
