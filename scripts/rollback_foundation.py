#!/usr/bin/env python
"""Roll back the active foundation checkpoint by updating the manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sonna_editor import config
from sonna_editor.foundation import (

    
    foundation_manifest_path,
    list_foundation_versions,
    rollback_foundation_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set the active foundation checkpoint to a previous version."
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Foundation version to activate, e.g. foundation-v3.",
    )
    parser.add_argument(
        "--foundation-repo",
        type=Path,
        default=None,
        help="Foundation repo/folder to update. Defaults to SONNA_FOUNDATION_REPO or config.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List manifest versions and exit.",
    )
    return parser.parse_args()


def _configure_repo(path: Path | None) -> None:
    if path is None:
        return
    repo = path.expanduser()
    config.FOUNDATION_REPO_DIR = repo
    os.environ[config.FOUNDATION_REPO_ENV_VAR] = str(repo)


def _print_versions() -> None:
    manifest = {}
    manifest_path = foundation_manifest_path()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active = manifest.get("active_version")
    for entry in list_foundation_versions():
        marker = "*" if entry.get("version") == active else " "
        print(
            f"{marker} {entry.get('version')}  "
            f"{entry.get('foundation_type')}  "
            f"{entry.get('checkpoint')}"
        )


def main() -> None:
    args = _parse_args()
    _configure_repo(args.foundation_repo)
    if args.list:
        _print_versions()
        return
    if not args.version:
        raise SystemExit("version is required unless --list is supplied")
    checkpoint = rollback_foundation_checkpoint(args.version)
    print(f"Active foundation version: {args.version}")
    print(f"Active foundation checkpoint: {checkpoint}")
    print(f"Foundation manifest: {foundation_manifest_path()}")


if __name__ == "__main__":
    main()
