#!/usr/bin/env python
"""Package Saha for client distribution and testing.

Builds a self-contained client test bundle as a .zip file that Darshil can
extract on macOS and run with a single click via `start_saha_mac.command`.
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
    "lightroom",
    "doc",
    "pyproject.toml",
    "uv.lock",
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

README_CONTENT = """===================================================
 Saha - Internal AI Photo Editing Tool (Client Test Bundle)
 Sonna Studios
===================================================

Welcome! This bundle contains the Saha desktop editor with the
foundation AI model, Lite profile calibration, and the Lightroom Classic bridge.

QUICK START (macOS):
1. Double-click `start_saha_mac.command` in this folder.
   (If macOS Gatekeeper asks on first launch: Right-click -> Open).
2. The launcher will automatically ensure `uv` and Node.js are available,
   sync dependencies, and open the Saha desktop application.

LIGHTROOM CLASSIC PLUGIN:
- Open Lightroom Classic -> File -> Plug-in Manager -> Add.
- Select the `lightroom/SahaBridge.lrplugin` folder.

FEATURES READY TO TEST:
- Lite Profiles: Create a preset-anchored profile calibrated to your camera WB.
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
        if part.startswith(".") and part not in {".gitattributes", ".gitignore"}:
            return True
    if path.suffix in EXCLUDE_EXTS:
        return True
    return False


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
                zf.write(entry_path, arcname=arcname)
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
                        zf.write(f_path, arcname=arcname)
                        file_count += 1
                        total_bytes += f_path.stat().st_size

    mb = total_bytes / (1024 * 1024)
    zip_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[package] Done! Added {file_count} files ({mb:.1f} MB uncompressed, {zip_mb:.1f} MB zip)")
    print(f"[package] Output file: {output_zip.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Saha for client distribution.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dist/saha-mac-client-bundle.zip"),
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
