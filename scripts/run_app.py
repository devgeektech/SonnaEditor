#!/usr/bin/env python
"""One-command development launcher for Saha.

This starts the Electron/React app from the repo root. The Electron main
process owns the backend lifecycle: it reuses an existing API on port 8765 or
spawns `scripts/serve.py` itself.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sonna_editor import config


def _ensure_windows_paths() -> None:
    """Add standard Windows installation directories to PATH if missing."""
    if sys.platform != "win32":
        return
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "nodejs",
        Path(r"C:\Program Files\nodejs"),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "nodejs",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs",
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin",
        Path(os.environ.get("APPDATA", "")) / "npm",
    ]
    current_path = os.environ.get("PATH", "")
    current_parts = [p.lower() for p in current_path.split(";") if p.strip()]
    to_add: list[str] = []
    for candidate in candidates:
        if candidate.exists() and str(candidate).lower() not in current_parts:
            to_add.append(str(candidate))
    if to_add:
        os.environ["PATH"] = ";".join(to_add) + ";" + current_path


def _npm_command() -> str:
    _ensure_windows_paths()
    command = "npm.cmd" if sys.platform == "win32" else "npm"
    resolved = shutil.which(command)
    if resolved is None and sys.platform == "win32":
        for fallback in [
            Path(r"C:\Program Files\nodejs\npm.cmd"),
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "nodejs" / "npm.cmd",
        ]:
            if fallback.exists():
                return str(fallback)
    if resolved is None:
        raise RuntimeError(
            "npm was not found on PATH. Install Node.js, then rerun this launcher."
        )
    return resolved


def _node_command() -> str:
    _ensure_windows_paths()
    command = "node.exe" if sys.platform == "win32" else "node"
    resolved = shutil.which(command)
    if resolved is None and sys.platform == "win32":
        for fallback in [
            Path(r"C:\Program Files\nodejs\node.exe"),
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "nodejs" / "node.exe",
        ]:
            if fallback.exists():
                return str(fallback)
    if resolved is None:
        raise RuntimeError(
            "Node.js was not found on PATH. Install Node.js LTS, then rerun this launcher."
        )
    return resolved


def _is_node_modules_valid_for_platform(app_dir: Path) -> bool:
    """Check if node_modules has binary assets matching the current OS."""
    node_modules = app_dir / "node_modules"
    if not node_modules.exists():
        return False
    electron_dist = node_modules / "electron" / "dist"
    if not electron_dist.exists():
        return False
    if sys.platform == "win32":
        return (electron_dist / "electron.exe").exists()
    if sys.platform == "darwin":
        return (electron_dist / "Electron.app").exists()
    if sys.platform.startswith("linux"):
        return (electron_dist / "electron").exists()
    return True


def _run(command: list[str], cwd: Path) -> int:
    print(f"[saha] {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the Saha Electron app and backend with one command."
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not run npm install even if frontend dependencies are missing.",
    )
    args = parser.parse_args()

    repo_root = config.PROJECT_ROOT
    app_dir = repo_root / "saha-app"
    node_modules = app_dir / "node_modules"

    if not (app_dir / "package.json").exists():
        raise SystemExit(f"Frontend package not found at {app_dir}")

    config.ensure_runtime_directories()
    _node_command()
    npm = _npm_command()

    if sys.platform == "darwin" and os.environ.get("ELECTRON_DISABLE_GPU") is None:
        os.environ["ELECTRON_DISABLE_GPU"] = "0"

    if not node_modules.exists() or not _is_node_modules_valid_for_platform(app_dir):
        if node_modules.exists():
            print(
                "[saha] Detected node_modules from a different operating system. Reinstalling...",
                flush=True,
            )
            shutil.rmtree(node_modules, ignore_errors=True)
        if args.skip_install:
            raise SystemExit(
                "saha-app/node_modules is missing or built for another OS. Run without --skip-install "
                "or install frontend dependencies manually."
            )
        install_code = _run([npm, "install"], app_dir)
        if install_code != 0:
            raise SystemExit(install_code)

    raise SystemExit(_run([npm, "run", "dev"], app_dir))


if __name__ == "__main__":
    main()
