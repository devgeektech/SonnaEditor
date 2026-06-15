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


def _npm_command() -> str:
    command = "npm.cmd" if sys.platform == "win32" else "npm"
    resolved = shutil.which(command)
    if resolved is None:
        raise RuntimeError(
            "npm was not found on PATH. Install Node.js, then rerun this launcher."
        )
    return resolved


def _node_command() -> str:
    command = "node.exe" if sys.platform == "win32" else "node"
    resolved = shutil.which(command)
    if resolved is None:
        raise RuntimeError(
            "Node.js was not found on PATH. Install Node.js LTS, then rerun this launcher."
        )
    return resolved


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

    if not node_modules.exists():
        if args.skip_install:
            
            raise SystemExit(
                "saha-app/node_modules is missing. Run without --skip-install "
                "or install frontend dependencies manually."
            )
        install_code = _run([npm, "install"], app_dir)
        if install_code != 0:
            raise SystemExit(install_code)

    raise SystemExit(_run([npm, "run", "dev"], app_dir))


if __name__ == "__main__":
    main()
