#!/usr/bin/env python
"""Create a clean, versioned, self-contained production distribution zip of Saha.

Features:
- Versioning support: names output saha-vX.Y.zip (e.g. saha-v1.0.zip, saha-v1.1.zip).
- Auto-increment: automatically detects previous versions in dist/ and increments minor version.
- Cleanup: automatically deletes all previous zip packages from dist/ so only the latest remains.
- Exclusions: strips developer caches, tests, test fixtures, audit/diagnostic scripts, and git artifacts.
"""

from __future__ import annotations

import argparse
import os
import re
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
    "DOUBLE_CLICK_MAC_START_SAHA.command",
    "DOUBLE_CLICK_WIN_START_SAHA.bat",
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
EXECUTABLE_NAMES = {"DOUBLE_CLICK_MAC_START_SAHA.command", "package.sh"}


def get_readme_content(version: str) -> str:
    return f"""===================================================
 Saha v{version} - Internal AI Photo Editing Tool
 Sonna Studios
===================================================

Welcome! This bundle contains the complete, local Saha desktop editor (v{version})
with the foundation AI model, trained Lite / Personal AI profiles, and
the Lightroom Classic bridge plugin.

Everything runs 100% locally on your computer. All files, model weights,
and edits are stored on your local disk.

---------------------------------------------------
QUICK START (macOS):
---------------------------------------------------
1. Double-click `DOUBLE_CLICK_MAC_START_SAHA.command` in this folder.
   - If macOS Gatekeeper prompts on first launch: Right-click -> Open.
   - Or open Terminal in this folder and run: bash DOUBLE_CLICK_MAC_START_SAHA.command
2. The launcher will automatically configure local dependencies,
   enable Apple Silicon MPS GPU acceleration, and open the Saha desktop application.

---------------------------------------------------
QUICK START (Windows):
---------------------------------------------------
1. Double-click `DOUBLE_CLICK_WIN_START_SAHA.bat` in this folder.
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

    for p in repo_root.glob("**/__pycache__"):
        if ".venv" not in p.parts:
            shutil.rmtree(p, ignore_errors=True)
            removed_count += 1

    for ext in ("*.pyc", "*.pyo", "*.DS_Store"):
        for f in repo_root.glob(f"**/{ext}"):
            if ".venv" not in f.parts:
                try:
                    f.unlink()
                    removed_count += 1
                except Exception:
                    pass

    return removed_count


def resolve_next_version(dist_dir: Path, requested_version: str | None = None) -> str:
    """Resolve package version.

    If requested_version is provided, uses it.
    Otherwise, inspects dist/ for existing saha-vX.Y.zip files and increments minor version.
    If none found, defaults to 1.0.
    """
    if requested_version:
        clean = requested_version.strip().lstrip("v").lstrip("V")
        return clean

    version_pattern = re.compile(r"^saha-v(\d+)\.(\d+)\.zip$", re.IGNORECASE)
    existing_versions: list[tuple[int, int]] = []

    if dist_dir.exists():
        for item in dist_dir.glob("saha-v*.zip"):
            m = version_pattern.match(item.name)
            if m:
                existing_versions.append((int(m.group(1)), int(m.group(2))))

    if not existing_versions:
        repo_root = dist_dir.parent
        for candidate in [dist_dir / "version.txt", repo_root / ".saha" / "version.txt"]:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8").strip().lstrip("v").lstrip("V")
                if content:
                    return content
        return "1.0"

    existing_versions.sort()
    latest_major, latest_minor = existing_versions[-1]
    return f"{latest_major}.{latest_minor + 1}"


def delete_old_packages(dist_dir: Path, exclude_file: Path | None = None) -> int:
    """Delete all older package zip files in dist/."""
    deleted_count = 0
    if not dist_dir.exists():
        return 0

    for item in dist_dir.glob("*.zip"):
        if exclude_file and item.resolve() == exclude_file.resolve():
            continue
        try:
            item.unlink()
            print(f"[clean] Deleted old package: {item.name}")
            deleted_count += 1
        except Exception as e:
            print(f"[warning] Could not delete {item.name}: {e}")

    return deleted_count


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


def build_package(version: str | None = None, output_path: Path | None = None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean local caches across workspace
    print("[clean] Cleaning local caches before packaging...")
    cleaned = clean_local_caches(repo_root)
    print(f"[clean] Removed {cleaned} cache directories/files.")

    # 2. Resolve version and output filename
    v_str = resolve_next_version(dist_dir, requested_version=version)
    if output_path is None:
        target_zip = dist_dir / f"saha-v{v_str}.zip"
    else:
        target_zip = output_path

    # 3. Delete old zip packages from dist/
    print("[clean] Removing older packages from dist/...")
    removed_old = delete_old_packages(dist_dir, exclude_file=target_zip)
    if removed_old > 0:
        print(f"[clean] Successfully cleared {removed_old} older package(s).")

    # 4. Record current version in dist/version.txt
    version_file = dist_dir / "version.txt"
    try:
        version_file.write_text(v_str, encoding="utf-8")
    except Exception:
        pass

    # 5. Build clean zip package
    print(f"[package] Building Saha v{v_str}: {target_zip.name}...")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("QUICK_START.txt", get_readme_content(v_str))

        # Add top-level included entries
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

        # Add essential scripts from scripts/
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
    zip_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f"[package] Done! Packaged {file_count} files ({mb:.1f} MB uncompressed, {zip_mb:.1f} MB zip)")
    print(f"[package] Output file: {target_zip.resolve()}")
    return target_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Saha into a clean, versioned production zip."
    )
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="Optional explicit version string (e.g. 1.0, 1.1). If omitted, auto-increments.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional custom destination path for the zip archive",
    )
    args = parser.parse_args()
    build_package(version=args.version, output_path=args.output)


if __name__ == "__main__":
    main()
