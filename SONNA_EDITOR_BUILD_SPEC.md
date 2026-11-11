# Sonna Editor — Build Specification

**Version:** 1.0
**Target platforms:** macOS, Windows, and Linux
**Reference hardware:** M1 Pro MacBook Pro, 32GB RAM
**Goal:** Internal AI photo editing tool that learns Sonna's editing style and applies it to new event work, outputting Lightroom-compatible XMP sidecars.

**Implementation status note (2026-06-19):** This spec remains the historical build plan and workflow reference. The current branch implementation state lives in `HANDOVER.md`, `SESSION_STATE.md`, and `project_knowledge.md`. Current Windows training verification uses Python 3.11.15, uv 0.11.17, PyTorch `2.11.0+cu128`, and CUDA on an NVIDIA GeForce RTX 3050. The `Auto_Straighten` branch also tracks the hidden foundation checkpoint `foundation-sonna-raw-xmp-004-visual`. Do not rewrite completed task specs for minor implementation changes; update the status docs instead.

---

## How to use this document

This spec is designed for use with Claude Code, session by session. Each phase is broken into discrete tasks. Each task has:

1. **Workflow** — model to use (Sonnet/Opus), whether to use `/plan`, whether to invoke multi-agent review
2. **Goal** — what we're building
3. **Inputs** — what needs to exist before starting
4. **Outputs** — what should exist when finished
5. **Success criteria** — how we know it works
6. **Claude Code prompt** — the prompt to start that session

Work through phases sequentially. Don't skip ahead — later phases assume earlier ones work.

**For the Claude Code workflow patterns** (universal session opener, multi-agent invocation template, full model-selection matrix), see `HANDOVER.md` Part 5. This spec embeds the workflow per-task; the handover has the patterns themselves.

---

## Architecture overview

```
                    ┌─────────────────────────────────────────┐
                    │           SONNA EDITOR                  │
                    │       (Electron Desktop App)             │
                    └─────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  TRAINING    │          │  INFERENCE   │          │ FINE-TUNING  │
    │  PIPELINE    │          │  PIPELINE    │          │   PIPELINE   │
    └──────────────┘          └──────────────┘          └──────────────┘
            │                         │                         │
            ▼                         ▼                         ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │ Lightroom    │          │ New shoot    │          │ User-tweaked │
    │ catalog +    │          │ folder of    │          │ XMPs from    │
    │ XMP sidecars │          │ RAW files    │          │ recent jobs  │
    └──────────────┘          └──────────────┘          └──────────────┘
            │                         │                         │
            ▼                         ▼                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  DNG normalisation (Adobe DNG Converter)                     │
    └──────────────────────────────────────────────────────────────┘
            │                         │                         │
            ▼                         ▼                         ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │ Parquet      │          │  PyTorch     │          │  Combined    │
    │ training set │ ────────▶│  model       │◀──────── │  retrain set │
    └──────────────┘          └──────────────┘          └──────────────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │ XMP sidecars │
                              │ next to RAWs │
                              └──────────────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │  Lightroom   │
                              │   Classic    │
                              └──────────────┘
```

**Key principles:**
- Local-only. No cloud dependencies in v1.
- Non-destructive. Never modify original RAW files.
- DNG as internal format. Normalise at ingestion.
- XMP sidecars as output. Lightroom-native integration.
- Continuous learning. Every project tweaks the model.

---

## Project structure

```
sonna-editor/
├── pyproject.toml              # uv-managed dependencies
├── README.md                   # project readme
├── .gitignore                  # excludes data/, models/, *.lrcat
├── .python-version             # pins Python 3.11
│
├── src/
│   └── sonna_editor/
│       ├── __init__.py
│       ├── config.py           # paths, constants, slider definitions
│       │
│       ├── data/               # Phase 1
│       │   ├── __init__.py
│       │   ├── catalog.py      # Lightroom .lrcat reader
│       │   ├── dng.py          # Adobe DNG Converter wrapper
│       │   ├── xmp.py          # XMP read/write
│       │   ├── extract.py      # extract preview + metadata + sliders
│       │   ├── dataset.py      # Parquet dataset builder
│       │   └── audit.py        # dataset quality auditor
│       │
│       ├── preset/             # Phase 2 — Mode B
│       │   ├── __init__.py
│       │   ├── parser.py       # Lightroom preset parser
│       │   ├── adjuster.py     # content-aware delta calculator
│       │   └── pipeline.py     # end-to-end preset application
│       │
│       ├── model/              # Phase 3
│       │   ├── __init__.py
│       │   ├── architecture.py # ConvNeXt + metadata fusion
│       │   ├── losses.py       # weighted MSE per parameter
│       │   ├── augmentation.py # input augmentation, target preservation
│       │   └── postprocess.py  # output clamping
│       │
│       ├── training/           # Phase 3
│       │   ├── __init__.py
│       │   ├── datamodule.py   # PyTorch Lightning data module
│       │   ├── module.py       # Lightning training module
│       │   └── train.py        # CLI training entrypoint
│       │
│       ├── inference/          # Phase 4
│       │   ├── __init__.py
│       │   ├── engine.py       # batch inference on M1 GPU
│       │   ├── confidence.py   # MC dropout uncertainty
│       │   └── pipeline.py     # end-to-end inference loop
│       │
│       ├── finetune/           # Phase 5
│       │   ├── __init__.py
│       │   ├── capture.py      # detect user-tweaked photos
│       │   ├── delta.py        # compute prediction-vs-final deltas
│       │   └── retrain.py      # fine-tuning entrypoint
│       │
│       ├── profiles/           # Phase 6
│       │   ├── __init__.py
│       │   ├── registry.py     # profile metadata & versioning
│       │   └── manager.py      # profile CRUD operations
│       │
│       └── ui/                 # Phase 7
│           ├── __init__.py
│           ├── main.py         # legacy PyQt6 entrypoint (superseded by Electron)
│           ├── windows/        # main window, dialogs
│           ├── widgets/        # custom widgets
│           └── workers/        # QThread workers for long ops
│
├── scripts/                    # one-off CLI scripts
│   ├── verify_environment.py
│   ├── audit_catalog.py
│   ├── train_profile.py
│   └── process_shoot.py
│
├── tests/                      # pytest tests
│   ├── conftest.py
│   ├── test_xmp.py
│   ├── test_extract.py
│   ├── test_preset.py
│   └── fixtures/               # tiny test files
│
├── notebooks/                  # exploration / prototyping
│   └── README.md
│
├── data/                       # gitignored
│   ├── raw/                    # source RAW + XMP for training
│   ├── dng/                    # normalised DNGs
│   ├── parquet/                # training datasets
│   └── audits/                 # auditor output reports
│
└── models/                     # gitignored
    ├── sonna_events_v1.ckpt
    ├── sonna_events_v2.ckpt
    └── manifest.json           # profile registry
```

---

## Phase 0 — Environment setup

**Estimated time:** 2-3 hours
**Prerequisites:** None

### Task 0.1: Install foundational tools

**Goal:** Get the development environment ready on macOS, Windows, or Linux.

**Manual steps (do these yourself before Claude Code session):**

1. Install Homebrew if not already installed: https://brew.sh
2. Install `uv` (fast Python package manager): `brew install uv`, `pipx install uv`, or `python -m pip install --user uv`
3. Install `git` if not already.
4. Optional for DNG workflows: Download Adobe DNG Converter: https://helpx.adobe.com/camera-raw/digital-negative.html
   Install it at the OS default path or set `SONNA_DNG_CONVERTER` to the executable path.
5. Create a GitHub account if you don't have one (free tier is fine).
6. Create a new private repository called `sonna-editor`. Don't initialise it with anything.

**Success criteria:** All of the above installed. You can run `uv --version`, `git --version`, and the DNG Converter binary exists.

---

### Task 0.2: Initialise project & verify M1 GPU

**Workflow:** Sonnet, no `/plan`, no multi-agent. Routine setup task.

**Claude Code prompt:**

```
I'm starting a new Python project called sonna-editor. It's a local AI photo editing tool that should run on macOS, Windows, and Linux. I want to use uv for package management and Python 3.11.

Please:

1. Initialise a uv-managed Python 3.11 project in the current directory
2. Set up the full directory structure as specified in SONNA_EDITOR_BUILD_SPEC.md (the structure under "Project structure")
3. Create a pyproject.toml with these initial dependencies:
   - torch (latest stable; CUDA/MPS/CPU runtime selected automatically)
   - torchvision
   - pytorch-lightning
   - rawpy
   - pillow
   - lxml
   - pandas
   - pyarrow (for Parquet)
   - numpy
   - tqdm
   - pyqt6
   - pytest (dev dependency)
   - ruff (dev dependency)
   - mypy (dev dependency)
4. Create a .gitignore that excludes data/, models/, .venv/, __pycache__, .DS_Store, *.lrcat, *.lrcat-journal, .pytest_cache
5. Create a scripts/verify_environment.py that:
   - Imports torch and prints torch.__version__
   - Checks torch.backends.mps.is_available() and prints result
   - Runs a small tensor operation on the best available device (CUDA, MPS, or CPU) and reports success
   - Reports Adobe DNG Converter if installed or configured via `SONNA_DNG_CONVERTER`
   - Prints Python version and platform info
6. Create an empty README.md with the project name and one-line description
7. Initialise a git repo, make the initial commit, and tell me the command to push to my GitHub repo (which I've already created at github.com/[my-username]/sonna-editor)

After everything is created, run scripts/verify_environment.py and confirm all checks pass.
```

**Success criteria:**
- `uv run python scripts/verify_environment.py` outputs all green checks
- `torch.backends.mps.is_available()` returns `True`
- Project structure matches the spec
- Git repo initialised with first commit

---

## Phase 1 — Lightroom data extraction pipeline

**Estimated time:** 12-18 hours across multiple sessions
**Prerequisites:** Phase 0 complete, access to your Lightroom catalog

### Task 1.1: XMP sidecar reader/writer

**Workflow:** Sonnet, no `/plan`, no multi-agent. Library-style implementation, well-defined.

**Claude Code prompt:**

```
I need a robust XMP sidecar reader and writer for Lightroom edit metadata. Please implement src/sonna_editor/data/xmp.py with:

1. A function `read_xmp(path: Path) -> dict` that reads a Lightroom XMP sidecar and returns a dict of all develop settings. It should handle:
   - Standalone .xmp files next to RAWs
   - XMP embedded inside DNG files (extract via piexif or by parsing the file directly)
   - Both crs: namespace and the older xap: namespace
   - Missing fields gracefully (return None or default for absent sliders)

2. A function `write_xmp(path: Path, settings: dict, source_raw_path: Path | None = None) -> None` that writes a Lightroom-compatible XMP sidecar. It must:
   - Include proper namespace declarations (xmlns:crs, xmlns:xmp, xmlns:dc, xmlns:tiff, xmlns:exif)
   - Set crs:HasSettings="True"
   - Set crs:ProcessVersion to the latest (15.4 as of 2024)
   - Use correct XMP packet wrapping (<?xpacket begin="..." id="..."?> ... <?xpacket end="w"?>)
   - Be byte-identical-on-reread (round-trip safe)

3. A constant SLIDER_FIELDS in src/sonna_editor/config.py listing the 82 target fields we care about:
   Tone (8): Exposure2012, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012, Clarity2012, Dehaze
   Presence (3): Texture, Vibrance, Saturation
   White balance (2): Temperature, Tint
   HSL (24): HueAdjustmentRed/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta, plus same for SaturationAdjustment* and LuminanceAdjustment*
   Parametric (7): ParametricHighlights/Lights/Darks/Shadows + HighlightSplit/MidtoneSplit/ShadowSplit
   Color Grading (13): ColorGrade{Shadow,Midtone,Highlight,Global}{Hue,Sat,Lum} + ColorGradeBlending
   Calibration (6): CameraCalibration{Red,Green,Blue}{Hue,Saturation}
   Detail (8): Sharpness, SharpenRadius, SharpenDetail, SharpenEdgeMasking + LuminanceSmoothing, LuminanceNoiseReductionDetail/Contrast, ColorNoiseReduction
   Effects (4): PostCropVignetteAmount/Midpoint/Roundness, GrainAmount
   Lens (2): LensManualDistortionAmount, VignetteAmount
   Transform (5): PerspectiveVertical/Horizontal/Rotate/Scale/Aspect
   — DONE as of Phase 3 Task 3.1 side quest (37→82 expansion)

4. Comprehensive tests in tests/test_xmp.py:
   - Round-trip test (write then read returns same values)
   - Real Lightroom XMP fixture parses correctly (I'll provide a fixture file)
   - Missing fields return None
   - Output XMP opens correctly in Lightroom (we'll verify manually)

Reference the Adobe XMP Specification and CRS namespace docs for correctness. Don't invent field names — they need to match exactly what Lightroom writes.
```

**Manual step before this session:** Export one XMP from Lightroom (any edited photo, right-click → Metadata → Save Metadata to File) and put it in `tests/fixtures/sample_edit.xmp`.

**Success criteria:**
- All tests pass
- A written XMP, opened in Lightroom alongside a RAW, applies the expected edits

---

### Task 1.2: Adobe DNG Converter wrapper

**Workflow:** Sonnet, no `/plan`, no multi-agent. Subprocess wrapper, routine.

**Claude Code prompt:**

```
Implement src/sonna_editor/data/dng.py — a cross-platform Python wrapper around Adobe DNG Converter.

Requirements:

1. Function `convert_to_dng(input_path: Path, output_dir: Path, embed_original: bool = False, compress: bool = True) -> Path`:
   - Resolves the converter from `SONNA_DNG_CONVERTER`, OS default paths, or PATH, then calls it via subprocess
   - Uses appropriate CLI flags (-c for compressed, -e for embed original, -d for output directory)
   - Returns the path to the resulting DNG file
   - Handles errors (binary not found, conversion failure, unsupported format)

2. Function `batch_convert(input_paths: list[Path], output_dir: Path, max_workers: int = 4) -> list[Path]`:
   - Processes multiple files in parallel using multiprocessing.Pool
   - Shows tqdm progress bar
   - Returns list of output paths (None for failures)
   - Logs failures to a file in output_dir

3. Function `get_dng_converter_version() -> str` that returns the installed version string

4. Tests in tests/test_dng.py that mock subprocess calls and verify command construction. Skip live conversion tests (mark with @pytest.mark.integration).

Make the binary path configurable via src/sonna_editor/config.py so it works if the user has DNG Converter installed elsewhere.
```

**Success criteria:**
- Unit tests pass
- Manual test: convert a single CR3 or NEF file to DNG, file opens correctly in Lightroom

---

### Task 1.3: RAW preview & metadata extraction

**Workflow:** Sonnet, no `/plan`, no multi-agent. Library wrapper work.

**Claude Code prompt:**

```
Implement src/sonna_editor/data/extract.py — extracts everything we need from a RAW or DNG file for training.

Requirements:

1. Function `extract_preview(path: Path, target_size: int = 384) -> PIL.Image`:
   (target_size matches v1 model resolution; will be updated to 512 for v2 and 768 for v3 — driven by config.IMAGE_RESOLUTION)
   - Reads embedded JPEG preview from the RAW/DNG
   - Resizes to target_size on the long edge, preserving aspect ratio
   - Returns RGB PIL image
   - Uses rawpy or exifread to get the embedded preview without decoding the full RAW
   - Falls back to decoding the RAW with rawpy.imread() at half-size if no embedded preview exists

2. Function `extract_metadata(path: Path) -> dict` that returns a dict with:
   - iso (int)
   - shutter_speed (float, in seconds)
   - aperture (float, f-number)
   - focal_length (float, mm)
   - lens_model (str)
   - camera_body (str)
   - capture_datetime (datetime)
   - exposure_compensation (float, EV)
   - white_balance_preset (str: "Auto", "Daylight", "Tungsten", etc.)
   - camera_profile (str: "Adobe Color", "Camera Standard", etc. — from XMP if present)
   - width, height (int)

3. Function `compute_histogram(image: PIL.Image, bins: int = 32) -> np.ndarray`:
   - Returns a (3, bins) numpy array of normalised RGB histograms
   - Used as auxiliary input to the model

4. Function `extract_all(raw_path: Path, xmp_path: Path | None = None) -> dict`:
   - Combines preview, metadata, histogram, and slider values from XMP
   - Returns a dict ready to go into the training dataset
   - If xmp_path is None, looks for a sidecar next to raw_path automatically

5. Tests in tests/test_extract.py with a fixture RAW file (small CR2 or DNG, I'll provide).

Use rawpy for RAW handling. For metadata, prefer reading from the file's EXIF over re-extracting from the RAW data.
```

**Manual step:** Provide a single test RAW file at `tests/fixtures/sample.dng` or `tests/fixtures/sample.cr3`.

**Success criteria:**
- Extracts a thumbnail quickly on the reference machine; exact timing varies by OS and storage
- Metadata fields populate correctly for at least 3 different camera bodies you've used
- Tests pass

---

### Task 1.4: Lightroom catalog reader

**Workflow:** **OPUS** + **`/plan`** + **multi-agent (architect → engineer → reviewer)**. Highest-risk task in Phase 1. Schema is partially undocumented and version-dependent.

Use the multi-agent invocation pattern from HANDOVER.md Part 5. After `/plan` and architect review, the reviewer agent must specifically check: read-only enforcement (catalog can never be modified), missing photo handling, schema version compatibility, error message clarity.

**Fallback if this task gets ugly:** abandon catalog reader for v1. Use Lightroom's "Save Metadata to File" feature to dump XMPs alongside RAWs, then read XMPs directly via Task 1.1's reader. Catalog reader becomes a v2 productivity tool.

**Claude Code prompt:**

```
Implement src/sonna_editor/data/catalog.py — reads a Lightroom Classic .lrcat file (which is a SQLite database) to discover edited photos.

Requirements:

1. Function `connect_catalog(lrcat_path: Path) -> sqlite3.Connection`:
   - Opens the .lrcat file READ-ONLY (this is critical — we never want to risk corrupting the user's catalog)
   - Returns the connection

2. Function `find_edited_photos(conn, min_color_label: str | None = None, min_rating: int = 0, min_flag: str | None = None) -> list[dict]`:
   - Queries Adobe_images, Adobe_imageDevelopSettings, AgLibraryFile, AgLibraryFolder, AgLibraryRootFolder
   - Returns list of dicts with: image_id, file_path (resolved absolute path), capture_time, rating, color_label, pick_flag, has_develop_settings (bool)
   - Filters by the optional criteria

3. Function `get_develop_settings(conn, image_id: int) -> dict`:
   - Reads the developSettings text blob from Adobe_imageDevelopSettings
   - Parses the Lua-like format (it's actually serialized XMP)
   - Returns a dict matching what xmp.read_xmp() returns

4. Build the SQL queries carefully — Lightroom's schema has gone through versions. Target schema versions for Lightroom Classic 11 through 14. Document any version-specific handling in comments.

5. Function `export_xmps_for_photos(conn, photos: list[dict], output_method: str = "manual") -> None`:
   - For "manual" method: prints instructions for the user to do "Save Metadata to File" in Lightroom on the matching photos
   - For "auto" method: writes XMP sidecars next to the RAW files using our xmp.write_xmp(), based on the develop settings from the catalog
   - The "auto" method is faster but the "manual" method is safer for v1

6. Tests with a tiny synthetic .lrcat fixture (we'll create one by setting up Lightroom with a single photo and copying the catalog).

Important caveats to handle:
- Catalog might be locked if Lightroom is open — fail with clear error
- File paths in the catalog are relative; we have to join with the root folder path
- Some photos might be "missing" (offline) — flag these and skip
- Smart Previews exist in a separate .lrdata package; we don't need them but log if present
```

**Manual step:** Make a *copy* of one of your real Lightroom catalogs and put it somewhere safe for testing. Never run any tool against the live working catalog.

**Success criteria:**
- Reads a real Sonna catalog read-only without errors
- Finds the expected number of edited photos
- Successfully extracts develop settings for a sample photo and they match what `xmp.read_xmp()` returns when Lightroom exports the same photo's metadata

**Known v1 limitation — tone curves not extracted:**
Tone curve data (`ToneCurvePV2012`, `ToneCurvePV2012Red/Green/Blue`, `ToneCurveName2012`) is present in the Lua blob but stored as a table value (`{ 0, 0, 255, 255, ... }`). The Lua parser only handles scalar fields; table values are intentionally skipped. Tone curves are also absent from `SLIDER_FIELDS` and the model output vector. This means the v1 model cannot replicate or predict S-curve, luminosity-curve, or per-channel curve adjustments.

v2 enhancement: add `ToneCurvePV2012` parsing (4-point control point extraction) and include a flattened representation in the feature vector. Schedule after first model is trained and quality is assessed.

---

### Task 1.5: Dataset builder

**Workflow:** Sonnet, no `/plan`, no multi-agent. Mechanical data assembly.

**Claude Code prompt:**

```
Implement src/sonna_editor/data/dataset.py — turns a folder of RAW + XMP pairs (or DNGs with embedded XMP) into a Parquet training dataset.

Requirements:

1. Function `build_dataset(input_dir: Path, output_path: Path, profile_name: str, thumbnail_dir: Path) -> pd.DataFrame`:
   - Walks input_dir recursively, finds all RAW+XMP pairs
   - For each pair, calls extract.extract_all()
   - Saves the thumbnail JPEG to thumbnail_dir keyed by content hash
   - Builds a row with: id (hash), profile, raw_path, thumbnail_path, all metadata fields, histogram (as bytes blob), all 82 slider values
   - Writes the result as Parquet to output_path
   - Returns the DataFrame

2. Function `load_dataset(parquet_path: Path) -> pd.DataFrame` that loads the dataset back

3. Function `split_dataset(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, group_col: str = "shoot_id") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`:
   - Splits by SHOOT, not by photo (shoot_id derived from capture date + camera body — photos within ~12 hours of each other from the same body are the same shoot)
   - Returns train, val, test DataFrames
   - Uses sklearn's GroupShuffleSplit

4. Function `save_split(train, val, test, output_dir: Path)` that saves each split as separate Parquet files

5. CLI script scripts/build_dataset.py that takes --input-dir, --output-dir, --profile-name and runs the full pipeline with progress bars

6. Tests with synthetic data

Performance target: processes 1000 RAW+XMP pairs in under 5 minutes on the reference machine using multiprocessing for the per-file work.
```

**Success criteria:**
- Build a dataset from at least 100 real Sonna photos
- DataFrame has expected columns and reasonable values
- Split is by shoot, no leakage between train and val

---

### Task 1.6: Dataset auditor

**Workflow:** Sonnet, **light `/plan`** for the audit-check design and report structure, no multi-agent. The thinking task is "what should we audit and how do we present findings actionably" — `/plan` surfaces this before implementation.

**Claude Code prompt:**

```
Implement src/sonna_editor/data/audit.py — analyses a built dataset and produces a quality report.

Requirements:

1. Function `audit_dataset(parquet_path: Path, output_dir: Path) -> dict`:
   - Loads the dataset
   - Computes statistics
   - Generates plots (matplotlib) saved as PNGs
   - Writes a markdown report to output_dir/audit_report.md
   - Returns a summary dict

2. Statistics to compute and report:
   - Total photo count, photo count per shoot
   - Camera body distribution (warn if more than 3 bodies dominate)
   - Camera profile distribution (warn if mixed — recommend separating into multiple profiles)
   - ISO distribution histogram
   - Distribution of every slider value (82 histograms)
   - Detection of "likely unedited" photos: rows where 75+ sliders are exactly 0.0 (warn and list)
   - Outlier detection: photos with slider values > 3 std devs from the mean (list for manual review)
   - Consistency score: standard deviation of each slider, lower is better; flag if certain sliders have very high variance
   - Date range of captures
   - Estimated training time based on photo count and reference hardware

3. The markdown report should have clear sections:
   - Summary (photo count, recommendation: GO / WARN / STOP)
   - Hardware estimate (training time, memory needs)
   - Data composition (cameras, profiles, ISO)
   - Slider analysis (distributions, outliers)
   - Quality flags (unedited photos, outliers, inconsistency)
   - Recommendations (what to fix before training)

4. CLI script scripts/audit_catalog.py: takes --parquet-path and --output-dir

Make the report readable and actionable. The output should tell the user "you're ready to train" or "fix X, Y, Z first."
```

**Success criteria:**
- Run on a real Sonna dataset of 1000+ photos
- Report is readable and gives clear actions
- We use the report to decide whether to proceed to Phase 3

---

## Phase 2 — Mode B: Lite Preset (the fast win)

**Estimated time:** 8-12 hours
**Prerequisites:** Phase 1 tasks 1.1, 1.2, 1.3 complete

### Task 2.1: Lightroom preset parser

**Workflow:** Sonnet, no `/plan`, no multi-agent. File parser, routine.

**Claude Code prompt:**

```
Implement src/sonna_editor/preset/parser.py — reads Lightroom preset files.

Requirements:

1. Function `parse_preset(path: Path) -> dict`:
   - Handles three formats:
     - .xmp preset files (Lightroom Classic CC and later)
     - .lrtemplate files (legacy Lua-based, plain text key=value)
     - .xmpsettings (older XMP format)
   - Returns a dict of slider values matching SLIDER_FIELDS in config.py
   - Missing fields default to 0.0 (no adjustment)

2. Function `validate_preset(preset: dict) -> list[str]`:
   - Returns list of warnings about the preset
   - Flag if preset has local adjustments (we ignore those — only globals supported)
   - Flag if values are extreme

3. Tests with at least 3 real Lightroom presets in different formats

The parser should be robust to extra fields it doesn't understand (just ignore them).
```

**Manual step:** Provide 2-3 Sonna presets at `tests/fixtures/preset_*.xmp` and `tests/fixtures/preset_*.lrtemplate`.

**Success criteria:** Parses real Sonna presets, values match what Lightroom shows when you load the preset.

---

### Task 2.2: Content-aware adjuster

**Workflow:** Sonnet, no `/plan`, no multi-agent. Logic is well-specified in the prompt.

**Claude Code prompt:**

```
Implement src/sonna_editor/preset/adjuster.py — computes intelligent per-photo adjustments to apply on top of a base preset.

This is what makes "Mode B" smarter than just slapping a preset on every photo. For each new photo, we analyse the photo and compute a small delta that adjusts exposure, white balance, and shadow/highlight recovery to handle that specific photo's conditions.

Requirements:

1. Function `compute_adjustment(image: PIL.Image, metadata: dict, base_preset: dict, options: dict) -> dict`:
   - Returns a delta dict (only the fields we want to override)
   - Options control which adjustments are enabled:
     - auto_exposure: bool
     - auto_white_balance: bool
     - auto_shadow_recovery: bool
     - auto_highlight_recovery: bool

2. Auto-exposure logic:
   - Compute mean luminance of the image
   - Target middle-grey luminance ~118 (out of 255)
   - Compute exposure delta in stops needed to hit target
   - Clamp to ±0.7 stops max (we don't want to override the preset's intent dramatically)
   - Apply on top of base_preset["Exposure2012"]

3. Auto-white-balance logic:
   - Use grey-world assumption: average R/G and B/G ratios across the image
   - Compute temperature/tint correction
   - Clamp deltas: ±300K temperature, ±5 tint
   - Off by default — many shoots have intentional warm/cool grades

4. Shadow recovery:
   - If the bottom 10% of histogram has >25% of pixels (clipped shadows), add +10 to Shadows2012 on top of base
   - Clamp total Shadows2012 to +60

5. Highlight recovery:
   - If the top 10% of histogram has >5% of pixels (clipped highlights), subtract 10 from Highlights2012 on top of base
   - Clamp total Highlights2012 to -50

6. Function `apply_adjustment(base_preset: dict, delta: dict) -> dict`:
   - Combines base + delta with proper clamping to Lightroom slider ranges
   - Returns the final settings dict

7. Tests with synthetic images (e.g., all-black, all-white, normal histogram) verifying expected adjustments

The deltas should be subtle. The preset is doing the heavy lifting; the adjustments just handle outlier photos.
```

**Success criteria:**
- Underexposed test image gets +exposure delta
- Overexposed test image gets -exposure delta
- Normal test image gets near-zero delta

---

### Task 2.3: Mode B end-to-end pipeline

**Workflow:** Sonnet, no `/plan`, no multi-agent. Wiring task.

**Claude Code prompt:**

```
Implement src/sonna_editor/preset/pipeline.py — wires everything together for Mode B.

Requirements:

1. Function `process_shoot(input_dir: Path, output_dir: Path | None, preset_path: Path, options: dict, max_workers: int = 4) -> dict`:
   - Walks input_dir for RAW files (extensions in config.SUPPORTED_RAW_EXTENSIONS)
   - For each RAW:
     - Optionally converts to DNG first (option "convert_to_dng": False by default — we usually want to leave originals alone)
     - Extracts preview + metadata
     - Computes content-aware adjustment
     - Combines with base preset
     - Writes XMP sidecar next to original RAW (or to output_dir if provided)
   - Uses multiprocessing for parallel processing
   - Returns summary dict: {processed: int, failed: int, failures: list, output_paths: list}

2. CLI script scripts/process_shoot_preset.py with arguments:
   --input-dir
   --output-dir (optional, default: write next to RAWs)
   --preset
   --auto-exposure / --no-auto-exposure (default: on)
   --auto-white-balance / --no-auto-white-balance (default: off)
   --auto-shadow-recovery / --no-auto-shadow-recovery (default: on)
   --auto-highlight-recovery / --no-auto-highlight-recovery (default: on)
   --max-workers
   --dry-run (analyse but don't write XMPs)

3. Progress bar showing photos processed
4. Summary report printed at end

5. Integration test: process a folder of 10 real RAWs end-to-end, verify XMPs are written, manually open in Lightroom to confirm edits applied correctly.
```

**Success criteria:**
- Run on a real recent Sonna shoot
- Open in Lightroom, see the edits applied
- Edits look reasonable across varied lighting (this validates the auto-adjustment logic)

**MILESTONE:** At this point you have a working tool you can use day-to-day. Mode A (the trained model) replaces the preset values with model-predicted values, but the rest of the pipeline is identical and proven.

---

## Phase 3 — Model architecture & training

**Estimated time:** 20-30 hours across multiple sessions
**Prerequisites:** Phases 1 and 2 complete, dataset built and audited

### Task 3.1: Model architecture

**Workflow:** **OPUS** + **`/plan`** + **full multi-agent (architect → engineer → reviewer → QA)**. Highest-stakes task in the project. Architecture decisions propagate through everything downstream and are hard to reverse.

Use the multi-agent invocation pattern from HANDOVER.md Part 5 with explicit pause-between-roles. Architect proposes and justifies the architecture. Engineer implements after architect approval. Reviewer specifically checks: parameter count reasonableness (~30M target), CUDA/MPS/CPU compatibility, batch dimension handling, save/load correctness, embedding registry update logic for new camera bodies. QA writes tests including edge cases: single-sample batch, missing metadata fields, embedding overflow when a new camera appears.

**Critical:** also configure for **resolution flexibility** so 384/512/768 input sizes are a config flag, not hardcoded. This is required for the staged resolution roadmap (v1 384px → v2 512px → v3 768px).

**Claude Code prompt:**

```
Implement src/sonna_editor/model/architecture.py — the neural network architecture for predicting Lightroom slider values.

Architecture spec:

1. Class SonnaEditor(nn.Module):
   - **Image input resolution is config-driven** (default 384, supports 512 and 768). The model must work at any of these resolutions without code changes — only the config value changes.
   - Image branch:
     - Backbone: torchvision.models.convnext_tiny(weights="DEFAULT")
       (Choose ConvNeXt-Tiny — good accuracy/speed trade-off, ~28M params, runs well on M1 GPU at 384/512/768)
     - Remove the classification head, keep features (output: 768-d after global pooling — note this is feature dim, not resolution)
     - Allow freezing of stage 0-1 during initial training (config flag)

   - Metadata branch (input: dict of metadata + histogram):
     - ISO: log-scaled, then linear layer to 16-d
     - Shutter speed: log-scaled, then linear to 8-d
     - Aperture: linear to 8-d
     - Focal length: bucketed into 8 ranges (one-hot), then linear to 8-d
     - Camera body: nn.Embedding(num_bodies, 16) — body_id maintained in a registry
     - Lens: nn.Embedding(num_lenses, 8)
     - Camera profile: nn.Embedding(num_profiles, 8)
     - Capture WB: nn.Embedding(num_wb_presets, 8)
     - Histogram: 96-d input → MLP → 32-d
     - Concatenate all → MLP (128 → 64 → 64)
     - Output: 64-d metadata feature

   - Fusion: concatenate image (768) + metadata (64) → 832-d

   - Output heads (separate MLPs for each group):
     - Tone head: 832 → 256 → 128 → 8
     - Presence head: 832 → 128 → 64 → 3
     - WB head: 832 → 128 → 64 → 2
     - HSL head: 832 → 256 → 128 → 24
     - Parametric head: 832 → 128 → 64 → 7  [ADDED]
     - Color Grading head: 832 → 128 → 64 → 13  [ADDED]
     - Calibration head: 832 → 128 → 64 → 6  [ADDED]
     - Detail head: 832 → 64 → 4  [ADDED]
     - Noise head: 832 → 64 → 4  [ADDED]
     - Effects head: 832 → 64 → 4  [ADDED]
     - Lens head: 832 → 64 → 2  [ADDED]
     - Transform head: 832 → 64 → 5  [ADDED]

   - Total output: 82-d vector (~29.2M params total)
   — DONE as of Phase 3 Task 3.1

2. Class MetadataEncoder for the metadata branch (separate class for testability)

3. Forward signature:
   forward(image: Tensor[B, 3, H, W], metadata: dict[str, Tensor]) -> Tensor[B, 82]
   where H == W == config.IMAGE_RESOLUTION (one of 384, 512, 768)

4. Output ordering must match the SLIDER_FIELDS order in config.py exactly

5. Class registry for camera bodies / lenses / profiles / WB presets — the model holds an embedding table that grows as we encounter new bodies. Save/load with the model.

6. Forward should support a dropout flag for MC-dropout uncertainty estimation later

7. Tests:
   - Forward pass with dummy input produces output of correct shape
   - Total parameter count (should be ~30M, including backbone)
   - Save/load works
   - Embedding registry update works (adding a new camera body)
```

**Success criteria:**
- Model instantiates without error on M1 GPU
- Forward pass on batch of 8 images takes <500ms
- Parameter count documented

---

### Task 3.2: Loss function & augmentation

**Workflow:** **OPUS** + **`/plan`** + **multi-agent (architect → engineer → reviewer)**. High stakes. Wrong loss weights or buggy augmentation silently degrades training quality. Reviewer must specifically check: gradient balance across parameter categories, target preservation through augmentation pipeline (input augments, target stays fixed), deterministic seeding for reproducibility.

**Claude Code prompt:**

```
Implement src/sonna_editor/model/losses.py and src/sonna_editor/model/augmentation.py.

losses.py:

1. Class WeightedSliderLoss(nn.Module):
   - Per-parameter weighted MSE
   - Weights configurable via config.py (defaults below)
   - Each parameter normalised to its typical range before MSE (so a 0.5 stop exposure error contributes similarly to a 25 unit shadows error)

2. Loss weights — NEUTRAL LEARNER PRINCIPLE (locked decision, see HANDOVER.md Decision 8):
   ALL weights = 1.0 for all 82 sliders: SLIDER_LOSS_WEIGHTS = {field: 1.0 for field in SLIDER_FIELDS}
   Do not encode style opinions via weight inflation.

3. Range normalisation — done INSIDE WeightedSliderLoss, not via weights:
   normalized = (value - lo) / (hi - lo) using SLIDER_RANGES bounds for each slider.
   Temperature: log-space bounds lo=log(2000), hi=log(50000) — model predicts log(Kelvin).
   This gives equal penalty for equal fractional-range errors across all 82 sliders.
   Without this normalisation, large-range sliders (ColorGrade Hue 0-360) dominate gradients.

augmentation.py:

1. Class TrainingAugmentation:
   - Applied to IMAGE only, not to TARGET sliders
   - Random brightness shift: ±0.4 (this teaches the model that the SAME photo with different brightness should still get edited toward the same final look — critical)
   - Random contrast jitter: ±0.2
   - Random colour jitter: hue ±0.05, saturation ±0.2
   - Random horizontal flip: 50% (HSL hue won't flip with image, but at our training scale the model can learn invariance)
   - Random crop: 90-100% area, then resize to config.IMAGE_RESOLUTION
   - All applied as torchvision.transforms.v2 pipeline

2. Class ValidationAugmentation:
   - Just resize + center crop to config.IMAGE_RESOLUTION
   - No augmentation

3. Important: the augmentation must be deterministic given a seed for reproducibility, and the SAME image-target pair always gets fresh augmentation each epoch.

Tests verifying loss values are computed correctly and augmentation doesn't break tensor shapes.
```

**Success criteria:**
- Loss weights produce balanced gradients across parameters
- Augmentation visually looks reasonable (save examples to disk for inspection)
- No tensor shape errors

---

### Task 3.3: PyTorch Lightning training pipeline

**Workflow:** Mixed. `datamodule.py`: Sonnet, no `/plan`. `module.py`: **Opus** + **`/plan`** + light multi-agent (architect → engineer). The Lightning module's optimizer/scheduler choices and metric logging benefit from upfront thinking. Train script CLI: Sonnet.

**Claude Code prompt:**

```
Implement src/sonna_editor/training/datamodule.py and src/sonna_editor/training/module.py.

datamodule.py:

1. Class SonnaDataset(torch.utils.data.Dataset):
   - Loads from a Parquet file
   - __getitem__ returns (image_tensor, metadata_dict, target_tensor)
   - Image loaded from thumbnail JPEG, augmented per the augmentation module
   - Metadata constructed as dict of tensors
   - Target is the 82-d slider vector

2. Class SonnaDataModule(pytorch_lightning.LightningDataModule):
   - setup() loads train/val/test parquet files
   - DataLoaders with batch_size 16 default, num_workers 4, persistent_workers True
   - prepare_data() validates files exist

module.py:

1. Class SonnaLightningModule(pytorch_lightning.LightningModule):
   - Wraps SonnaEditor model
   - Configures AdamW optimizer (lr=3e-4, weight_decay=1e-4)
   - Cosine annealing scheduler with warm restarts
   - training_step / validation_step / test_step return loss + per-parameter MAE for logging
   - Logs to TensorBoard (M1-compatible)
   - configure_optimizers with optional differential LRs for backbone vs heads (heads 10x higher lr)

2. Custom metrics logged:
   - val_loss (overall weighted MSE)
   - val_mae_exposure, val_mae_temperature, val_mae_shadows, etc. (raw MAE per important parameter, in slider units)
   - val_mae_hsl_avg (averaged across 24 HSL params)

3. Callbacks configured:
   - ModelCheckpoint (save best by val_loss, keep top 3)
   - EarlyStopping (patience 10 epochs)
   - LearningRateMonitor

scripts/train_profile.py:

CLI training entrypoint:
- --train-parquet, --val-parquet, --test-parquet
- --output-dir for checkpoints
- --max-epochs (default 100, early stopping usually kicks in earlier)
- --batch-size (default 16)
- --resume-from-checkpoint
- --freeze-backbone (default: freeze for first 10 epochs, unfreeze after)
- Saves a final model.ckpt + a summary report (final metrics, training curves as PNGs)

Use runtime accelerator selection (`cuda`, `mps`, or `cpu`) and precision="32-true" for cross-platform reliability.
```

**Success criteria:**
- Training launches on M1 GPU without error
- One epoch on 1000 photos completes in <10 minutes
- Loss decreases meaningfully after a few epochs on a small subset

---

### Task 3.4: Training run on real data

**This is not a Claude Code coding task — it's a manual run.**

1. Build the full dataset from your Lightroom catalog (Phase 1)
2. Audit the dataset (Task 1.6), address any flagged issues
3. Run `scripts/train_profile.py` with your real data
4. Monitor TensorBoard during training
5. Evaluate the final model on the held-out test set

**Success criteria for v1 model:**
- Median exposure prediction error < 0.20 stops on test set
- Median temperature error < 250K
- Median tint error < 5
- Median HSL parameter error < 6 units
- Visual spot check: on 20 random test photos, the predicted edits look plausibly Sonna-style

If these aren't met: adjust loss weights, train longer, look at worst predictions to find data issues, possibly add more training data.

---

## Phase 4 — Inference engine

**Estimated time:** 8-12 hours
**Prerequisites:** Phase 3 complete with a trained model

### Task 4.1: Inference engine

**Workflow:** Sonnet initially. If performance falls short of the 5-minute-per-1000-photos target, escalate to **Opus + `/plan`** for an optimisation pass. Most of this is wiring the trained model into the existing Mode B pipeline structure.

**Claude Code prompt:**

```
Implement src/sonna_editor/inference/engine.py and src/sonna_editor/inference/pipeline.py.

engine.py:

1. Class InferenceEngine:
   - __init__(model_path: Path, device: str = "mps")
   - Loads checkpoint, sets model to eval mode
   - predict(images: list[Tensor], metadata: list[dict]) -> Tensor[N, 82]
   - predict_with_uncertainty(images, metadata, n_samples: int = 10) -> tuple[Tensor[N, 82], Tensor[N, 82]] (mean, std via MC dropout)
   - Batch processing internally for efficiency
   - Output post-processing: clamp to Lightroom slider valid ranges from config

2. Method warmup() that runs a single dummy forward pass so CUDA/MPS/CPU backend setup happens before timed inference

pipeline.py:

1. Function `process_shoot_with_model(input_dir: Path, output_dir: Path | None, model_path: Path, options: dict) -> dict`:
   - Mirrors process_shoot in preset/pipeline.py but uses the model instead of preset
   - Steps:
     a. Walk input_dir for RAW files
     b. (optional) Convert to DNG
     c. Extract previews + metadata in parallel
     d. Batch through inference engine
     e. (optional) Compute uncertainty for confidence flagging
     f. Write XMPs
   - Returns summary including low-confidence photo list (if uncertainty enabled)

2. Performance target: process a 1000-photo shoot in under 5 minutes on the reference machine
   - Bottleneck is RAW preview extraction (parallelisable)
   - Inference itself should be ~30-50 photos/sec batched

3. CLI script scripts/process_shoot_model.py with similar args to process_shoot_preset.py but --model-path instead of --preset
```

**Success criteria:**
- Process a real shoot end-to-end with the trained model
- Open in Lightroom, edits look like Sonna's style
- Speed target met

---

## Phase 5 — Continuous learning loop

**Estimated time:** 10-15 hours
**Prerequisites:** Phase 4 complete, model in production use for at least one project

### Task 5.1: Edit capture & delta tracking

**Workflow:** Sonnet, no `/plan`, no multi-agent. Mechanical comparison logic.

**Claude Code prompt:**

```
Implement src/sonna_editor/finetune/capture.py and src/sonna_editor/finetune/delta.py.

capture.py:

1. Function `capture_user_edits(shoot_dir: Path, model_predictions_path: Path, output_dir: Path) -> pd.DataFrame`:
   - Walks shoot_dir for current XMPs (which represent the user's final values after their tweaks)
   - Loads the model's original predictions from model_predictions_path (saved during inference)
   - For each photo, computes the delta between predicted and final
   - Returns DataFrame: id, raw_path, predicted_values, final_values, deltas, was_tweaked (bool: any delta > threshold)

2. The inference pipeline needs to be updated to save predictions alongside the XMP output (e.g., in a sidecar JSON file). Update pipeline.py from Phase 4 accordingly.

delta.py:

1. Function `analyse_deltas(captures: pd.DataFrame) -> dict`:
   - Aggregate stats: which parameters get tweaked most often, by how much
   - Identifies systematic biases (e.g., "model predicts exposure too high by 0.1 stops on average for ISO > 3200")
   - Returns analysis dict

2. Function `prepare_finetune_dataset(captures: pd.DataFrame, original_train_parquet: Path, output_path: Path, weight_recent: float = 2.0) -> pd.DataFrame`:
   - Combines original training data + captured user edits
   - Captured edits get higher sample weights (sampled more frequently during training)
   - Resulting DataFrame ready for the same training pipeline
```

**Success criteria:**
- After processing a shoot and tweaking in Lightroom, the system correctly identifies which photos were tweaked and by how much

---

### Task 5.2: Fine-tuning script

**Workflow:** **OPUS** + **`/plan`** + **full multi-agent (architect → engineer → reviewer → QA)**. Most dangerous code in the project. A bug here can corrupt a trained profile or silently degrade quality over time.

**Critical reviewer focus:** original model file must NEVER be overwritten. New version on every fine-tune run. Rollback must work. Validation set must be the same as original training so improvements are comparable. Registry update must be atomic (no half-written state).

QA tests must specifically cover: rollback scenarios, val-loss-worse-than-baseline scenarios (system should warn before promoting v2), interrupted training (no half-written checkpoint files), registry consistency under failure.

**Claude Code prompt:**

```
Implement src/sonna_editor/finetune/retrain.py and scripts/finetune_profile.py.

retrain.py:

1. Function `finetune_model(base_model_path: Path, finetune_parquet: Path, output_path: Path, options: dict) -> dict`:
   - Loads base model checkpoint
   - Reduces learning rate (default 1e-4, ~3x lower than original training)
   - Trains for fewer epochs (default 30 max with early stopping patience 5)
   - Validates against the SAME held-out validation set as the original training (so improvements are comparable)
   - Returns metrics: original_val_loss, finetuned_val_loss, improvement, per-parameter improvements

2. Critical: do NOT replace the original model file. Save fine-tuned model as a new version (e.g., sonna_events_v2.ckpt). Update the profile registry.

scripts/finetune_profile.py:

CLI:
- --profile-name
- --captures-dir (where capture data lives)
- --auto-accept (if val loss improves by > X%, auto-promote new version; otherwise prompt user)
- --dry-run (analyse only, don't actually train)

The script should produce a clear before/after report so the user can decide whether to adopt the new version.
```

**Success criteria:**
- Fine-tuning run completes in <2 hours on the reference machine
- New version metrics are reported clearly
- Old version preserved

---

## Phase 6 — Profile management

**Estimated time:** 4-6 hours

### Task 6.1: Profile registry

**Workflow:** Sonnet, no `/plan`, no multi-agent. JSON CRUD operations.

**Claude Code prompt:**

```
Implement src/sonna_editor/profiles/registry.py and src/sonna_editor/profiles/manager.py.

registry.py:

1. Class ProfileRegistry:
   - Stored as JSON at models/manifest.json
   - Each profile has: name, description, created_at, updated_at, base_model_version, current_version, training_data_summary (count, date range, cameras, etc.), version_history (list of {version, created_at, val_metrics, training_set_hash})
   - Methods: list_profiles(), get_profile(name), create_profile(name, ...), update_profile(name, ...), delete_profile(name)
   - Validates that referenced model files exist on disk

manager.py:

1. Higher-level operations:
   - create_new_profile(name, description, training_parquet) → triggers training
   - finetune_profile(name, captures_dir) → triggers fine-tuning
   - rollback_profile(name, target_version) → reverts current_version
   - export_profile(name, output_path) → bundles model + metadata for sharing (Phase 8)

2. Tests
```

**Success criteria:** Registry tracks at least one profile, supports versioning, persists across runs.

---

## Phase 7 — Desktop UI (Electron)

**Estimated time:** 25-35 hours
**Prerequisites:** All previous phases complete

### Task 7.1: Main window scaffolding

**Workflow:** Sonnet, **light `/plan`** for the layout and view structure decisions, no multi-agent. UI design benefits from upfront thinking.

**Claude Code prompt:**

```
Implement the Electron + React desktop UI under `saha-app/`.

Layout:
- Top toolbar: Profile selector dropdown, "Train New" button, "Manage Profiles" button
- Left sidebar: navigation (Process Shoot, Train Profile, Fine-tune, Settings)
- Main content area: changes based on sidebar selection
- Bottom status bar: current state, progress info

Initial views:
- "Process Shoot" view: folder picker, profile selector, options (auto-adjustments), Process button, progress area
- "Train Profile" view: catalog/folder picker, profile name input, options, Start Training button
- "Fine-tune" view: profile selector, captures folder picker, Start Fine-tune button
- "Settings" view: paths, options, DNG Converter location

Use the existing Electron + React stack and FastAPI backend bridge.

Implement basic windowing, navigation, and stub views. No actual functionality wired yet.
```

**Success criteria:** App opens on M1, all views accessible, no functionality but UI looks clean.

---

### Task 7.2: Process Shoot view (wire up Mode B)

**Workflow:** Sonnet, no `/plan`, no multi-agent. Wiring UI to existing backend.

**Claude Code prompt:**

```
Wire up the Process Shoot view to actually call the preset pipeline (Mode B for now — Mode A wired in 7.3).

Requirements:
- Folder picker actually browses
- Preset/profile picker lists available presets and trained profiles
- Options checkboxes wired to actual options
- Process button kicks off processing in a QThread (so UI doesn't freeze)
- Progress bar updates in real-time
- Result panel shows: photos processed, errors, time taken, "Open in Lightroom" hint
- Cancel button works (graceful shutdown of worker thread)
- Recent shoots list (last 5 processed)

Use Electron/React async flows properly. Don't block the renderer thread.
```

**Success criteria:** User can process a real shoot end-to-end through the UI without touching a terminal.

---

### Task 7.3: Wire up Mode A (trained model inference)

**Workflow:** Sonnet, no `/plan`, no multi-agent.

**Claude Code prompt:**

```
Extend the Process Shoot view to also support Mode A (trained model). Allow user to pick either a preset OR a profile from a dropdown.

Add confidence flag display: low-confidence photos in a list with "open in Lightroom for review" link.

Show predicted vs preset distinction visually (icon or label).
```

---

### Task 7.4: Train Profile view

**Workflow:** Sonnet, no `/plan`, no multi-agent.

**Claude Code prompt:**

```
Wire up Train Profile view to:
- Pick a Lightroom catalog OR a folder of edited photos
- Run the dataset builder
- Show audit results in a panel
- If audit passes (or user overrides), kick off training in a worker thread
- Show training progress: epoch, current loss, estimated time remaining
- On completion, register the new profile in the registry
- Provide cancel functionality

This is a long-running operation. UI must remain responsive.
```

---

### Task 7.5: Fine-tune view & profile manager

**Workflow:** Sonnet, no `/plan`, no multi-agent.

**Claude Code prompt:**

```
Implement Fine-tune view (similar to Train but uses fine-tuning pipeline) and Profile Manager dialog (list profiles, view metadata, delete, rollback versions).
```

---

### Task 7.6: Polish & packaging

**Workflow:** Sonnet, escalate to **Opus** if PyInstaller-specific issues arise (often the case for Python apps that bundle PyTorch). No `/plan` or multi-agent unless debugging.

**Claude Code prompt:**

```
Final polish:
- Application icon
- About dialog
- Error handling: friendly messages, "Send to Darshil" button that copies error log to clipboard
- Settings persistence (QSettings)
- Recent items
- Keyboard shortcuts
- Help menu with link to internal docs

For v1 packaging (just for Darshil's M1):
- PyInstaller spec to bundle everything as a .app
- Test that the .app launches on a fresh M1 with no Python installed
- Document the build process

Don't bother with code signing yet — that's Phase 8 for team distribution.
```

**Success criteria:** Standalone .app launches on M1, all features work, no terminal needed.

---

## Phase 8 — Team distribution (deferred)

Not specified in detail. Plan when Phase 7 is solid for Darshil personally. Will involve:

- Code signing certificates (Apple Developer Program ~NZD $165/year)
- Notarisation for macOS Gatekeeper
- DMG installer
- Shared profile storage (initially: Dropbox/Google Drive folder containing model files)
- Auto-update mechanism (Sparkle for macOS)
- Onboarding doc for Chad/Sean/Erin

---

## Costs summary

**Development (one-time):**
- Time: 80-100 hours over 8-10 weeks at 8-10 hrs/week
- Cloud GPU: NZD $0 by default (train locally when hardware is sufficient)
- Code signing (Phase 8 only, deferred): NZD $165/year if/when distributing to team

**Operational (ongoing, Phase 7 onward):**
- Compute: NZD $0 (local M1)
- Storage: NZD $0 (local disk)
- Subscriptions: NZD $0
- **Total: NZD $0/month**

Estimated savings vs. off-the-shelf AI editing subscriptions at typical Sonna volume: ~NZD $300-500/year, plus full IP ownership and no upload time.

---

## Glossary

- **DNG (Digital Negative)** — Adobe's normalised RAW format. We use it as our internal standard.
- **XMP** — eXtensible Metadata Platform. Adobe's metadata sidecar format. Our output.
- **Smart Preview** — Lightroom's lightweight DNG-based preview. We use embedded RAW previews instead.
- **Personal AI Profile** — a trained model profile specific to one photographer or studio's editing style.
- **Slider values** — the 37 numerical edit parameters Lightroom exposes.
- **Lite Profile / Mode B** — preset-based with content-aware adjustments. No training required.
- **Personal Profile / Mode A** — trained model that predicts slider values from image content.
- **Fine-tuning** — incrementally retraining an existing profile on new user-tweaked data.

---

## Final notes

**This spec is the source of truth.** When working with Claude Code, paste the relevant phase + task into the chat. Don't ask Claude Code to "implement everything" — work task by task, verify each one before moving on.

**Test on real data early.** Phase 1 should run against your actual Lightroom catalog as soon as the dataset builder works. Don't wait until Phase 3 to discover your training data has issues.

**The Mode B path is the fast win.** By end of Phase 2 you have a tool you actually use. Don't lose momentum chasing Mode A perfection before Mode B is solid.

**Ask for help.** When something doesn't work in Claude Code, come back here and ask. Some tasks (catalog reverse-engineering, MPS quirks, Lightroom XMP edge cases) will need iteration.
