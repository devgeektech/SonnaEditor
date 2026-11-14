#!/usr/bin/env python
"""Create a clean, self-contained production distribution zip of Saha.

Excludes all developer caches, unit/integration tests, one-time testing/audit scripts,
and dev-only files while keeping everything required to run the full application
locally on Windows and macOS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Paths to include in the bundle (relative to repo root)
INCLUDE_PATHS = [
    "src",
    "saha-app",
    "SonnaEditorFoundation",
    "v1_learning",
    "lightroom",
    "doc",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "start_saha_mac.command",
    "start_saha_windows.bat",
    "package.bat",
    "package.sh",
]

# Essential runtime scripts in scripts/ required for app execution, inference, and profile training
ESSENTIAL_SCRIPTS = {
    "run_app.py",
    "serve.py",
    "process_shoot_model.py",
    "process_shoot_preset.py",
    "train_profile.py",
    "finetune_profile.py",
    "build_mode_b_checkpoint.py",
    "build_dataset.py",
    "build_dataset_from_catalog.py",
    "train_foundation_model.py",
    "verify_environment.py",
    "run_style_survey.py",
    "package_app.py",
    "package_for_client.py",
}

# Directories completely excluded from the bundle
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    "__pycache__",
    "node_modules",
    "dist",
    "dist-app",
    ".system_generated",
    "training_workspace",
    "scratch",
    "tests",
    "references",
}

# File extensions excluded from the bundle
EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}

# Explicit files excluded from the bundle
EXCLUDE_FILES = {
    ".cursorrules",
    "powershell.cmd",
}

EXECUTABLE_EXTENSIONS = {".command", ".sh"}
EXECUTABLE_NAMES = {"start_saha_mac.command", "package.sh"}

README_CONTENT = """===================================================
 Saha - Internal AI Photo Editing Tool
 Sonna Studios
===================================================

Welcome! This bundle contains the complete, local Saha desktop editor
with the foundation AI model, trained Lite / Personal AI profiles, and
the Lightroom Classic bridge plugin.

Everything runs 100% locally on your computer. All files, model weights,
and edits are stored on your local disk.

---------------------------------------------------
QUICK START (macOS):
---------------------------------------------------
1. Double-click `start_saha_mac.command` in this folder.
   - If macOS Gatekeeper prompts on first launch: Right-click -> Open.
   - Or open Terminal in this folder and run: bash start_saha_mac.command
2. The launcher will automatically configure local dependencies,
   enable Apple Silicon MPS GPU acceleration, and open the Saha desktop application.

---------------------------------------------------
QUICK START (Windows):
---------------------------------------------------
1. Double-click `start_saha_windows.bat` in this folder.
2. The launcher will automatically configure local dependencies,
   enable CUDA GPU acceleration if an NVIDIA GPU is present, and open the Saha desktop application.

---------------------------------------------------
LIGHTROOM CLASSIC PLUGIN:
---------------------------------------------------
1. Open Lightroom Classic -> File -> Plug-in Manager -> Add.
2. Select the `lightroom/SahaBridge.lrplugin` folder.

---------------------------------------------------
FEATURES READY TO USE:
---------------------------------------------------
- Lite Profiles: Preset-anchored profiles calibrated to your camera WB.
- Personal AI Profiles: Train a personalized model from your RAW + XMP edits.
- Batch Inference: Process shoot folders and output XMP sidecars.
- Lightroom Bridge: One-click export and import of predicted settings.
===================================================
"""


def clean_local_caches(repo_root: Path) -> int:
    """Remove cache directories and files across the repository."""
    removed_count = 0

    # Cache directories
    top_caches = [
        repo_root / ".mypy_cache",
        repo_root / ".pytest_cache",
        repo_root / ".ruff_cache",
        repo_root / ".cache",
        repo_root / "saha-app" / "node_modules" / ".vite",
    ]
    for cdir in top_caches:
        if cdir.exists():
            shutil.rmtree(cdir, ignore_errors=True)
            removed_count += 1

    # __pycache__ directories outside .venv
    for p in repo_root.glob("**/__pycache__"):
        if ".venv" not in p.parts:
            shutil.rmtree(p, ignore_errors=True)
            removed_count += 1

    # Stray cache files
    for ext in ("*.pyc", "*.pyo", "*.DS_Store"):
        for f in repo_root.glob(f"**/{ext}"):
            if ".venv" not in f.parts:
                try:
                    f.unlink()
                    removed_count += 1
                except Exception:
                    pass

    return removed_count


def is_path_excluded(path: Path, root: Path) -> bool:
    """Determine whether a path should be omitted from the production bundle."""
    rel = path.relative_to(root)
    parts = rel.parts

    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith(".") and part not in {".gitattributes", ".gitignore", ".python-version"}:
            return True

    if path.is_file():
        if path.name in EXCLUDE_FILES:
            return True
        if path.suffix in EXCLUDE_EXTS:
            return True
        # If file is inside scripts/, only include essential runtime scripts
        if len(parts) >= 2 and parts[0] == "scripts" and path.name not in ESSENTIAL_SCRIPTS:
            return True

    return False


def _write_zip_entry(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    """Write file to zip, preserving POSIX executable permissions (0755) for scripts."""
    if file_path.suffix in EXECUTABLE_EXTENSIONS or file_path.name in EXECUTABLE_NAMES:
        zinfo = zipfile.ZipInfo.from_file(file_path, arcname=arcname)
        zinfo.external_attr = 0o100755 << 16
        with open(file_path, "rb") as f:
            zf.writestr(zinfo, f.read())
    else:
        zf.write(file_path, arcname=arcname)


def build_package(output_zip: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print("[clean] Cleaning local caches before packaging...")
    cleaned = clean_local_caches(repo_root)
    print(f"[clean] Removed {cleaned} cache directories/files.")

    print(f"[package] Building clean production bundle: {output_zip}")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("QUICK_START.txt", README_CONTENT)

        # 1. Add top-level included entries
        for entry in INCLUDE_PATHS:
            entry_path = repo_root / entry
            if not entry_path.exists():
                print(f"[warning] Path not found: {entry_path}", file=sys.stderr)
                continue

            if entry_path.is_file():
                arcname = entry_path.relative_to(repo_root).as_posix()
                _write_zip_entry(zf, entry_path, arcname=arcname)
                file_count += 1
                total_bytes += entry_path.stat().st_size
            else:
                for root_dir, dirs, files in os.walk(entry_path):
                    r_path = Path(root_dir)
                    if is_path_excluded(r_path, repo_root):
                        dirs[:] = []
                        continue
                    for file in files:
                        f_path = r_path / file
                        if is_path_excluded(f_path, repo_root):
                            continue
                        arcname = f_path.relative_to(repo_root).as_posix()
                        _write_zip_entry(zf, f_path, arcname=arcname)
                        file_count += 1
                        total_bytes += f_path.stat().st_size

        # 2. Add essential scripts from scripts/
        scripts_dir = repo_root / "scripts"
        if scripts_dir.exists():
            for script_name in sorted(ESSENTIAL_SCRIPTS):
                script_path = scripts_dir / script_name
                if script_path.exists() and script_path.is_file():
                    arcname = f"scripts/{script_name}"
                    _write_zip_entry(zf, script_path, arcname=arcname)
                    file_count += 1
                    total_bytes += script_path.stat().st_size

    mb = total_bytes / (1024 * 1024)
    zip_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[package] Done! Packaged {file_count} files ({mb:.1f} MB uncompressed, {zip_mb:.1f} MB zip)")
    print(f"[package] Destination: {output_zip.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Saha for clean, minimal production distribution."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dist/saha-production-bundle.zip"),
        help="Destination path for the production zip archive",
    )
    args = parser.parse_args()
    build_package(args.output)


if __name__ == "__main__":
    main()
