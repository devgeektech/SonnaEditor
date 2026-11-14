#!/usr/bin/env python
"""Package Saha for cross-platform local distribution and testing.

Builds a self-contained client test bundle as a .zip file that can be
extracted on macOS or Windows and run with a single click via
`start_saha_mac.command` (macOS) or `start.bat` / `START_SAHA.bat` (Windows).
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Paths to include (relative to repo root)
INCLUDE_PATHS = [
    "src",
    "scripts",
    "saha-app",
    "SonnaEditorFoundation",
    "v1_learning",
    "lightroom",
    "doc",
    "references",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".cursorrules",
    "README.md",
    "start_saha_mac.command",
    "start_saha_windows.bat",
]

# Patterns or directory names to exclude
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "dist-app",
    ".system_generated",
    "training_workspace",
    "scratch",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}

EXECUTABLE_EXTENSIONS = {".command", ".sh"}
EXECUTABLE_NAMES = {"start_saha_mac.command"}

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


def is_excluded(path: Path, root: Path, include_node_modules: bool = False) -> bool:
    rel_parts = path.relative_to(root).parts
    for part in rel_parts:
        if part in EXCLUDE_DIRS:
            if part == "node_modules" and include_node_modules:
                continue
            return True
        if part.startswith(".") and part not in {".gitattributes", ".gitignore", ".python-version"}:
            return True
    if path.suffix in EXCLUDE_EXTS:
        return True
    return False


def _write_zip_entry(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    """Write file to zip, preserving POSIX executable permissions for scripts."""
    if file_path.suffix in EXECUTABLE_EXTENSIONS or file_path.name in EXECUTABLE_NAMES:
        zinfo = zipfile.ZipInfo.from_file(file_path, arcname=arcname)
        # Set POSIX permissions to 0755 (-rwxr-xr-x): (0o100755 << 16)
        zinfo.external_attr = 0o100755 << 16
        with open(file_path, "rb") as f:
            zf.writestr(zinfo, f.read())
    else:
        zf.write(file_path, arcname=arcname)


def build_package(output_zip: Path, include_node_modules: bool = False) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"[package] Creating client bundle: {output_zip}")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add quick start README
        zf.writestr("QUICK_START.txt", README_CONTENT)

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
                    if is_excluded(r_path, repo_root, include_node_modules):
                        dirs[:] = []
                        continue
                    for file in files:
                        f_path = r_path / file
                        if is_excluded(f_path, repo_root, include_node_modules):
                            continue
                        arcname = f_path.relative_to(repo_root).as_posix()
                        _write_zip_entry(zf, f_path, arcname=arcname)
                        file_count += 1
                        total_bytes += f_path.stat().st_size

    mb = total_bytes / (1024 * 1024)
    zip_mb = output_zip.stat().st_size / (1024 * 1024)
    print(
        f"[package] Done! Added {file_count} files ({mb:.1f} MB uncompressed, {zip_mb:.1f} MB zip)"
    )
    print(f"[package] Output file: {output_zip.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Saha for client distribution.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dist/saha-app-bundle.zip"),
        help="Destination path for the zip archive",
    )
    parser.add_argument(
        "--include-node-modules",
        action="store_true",
        help="Include saha-app/node_modules in the zip bundle",
    )
    args = parser.parse_args()
    build_package(args.output, include_node_modules=args.include_node_modules)


if __name__ == "__main__":
    main()
