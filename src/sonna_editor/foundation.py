"""Foundation checkpoint discovery and manifest helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sonna_editor import config


def foundation_repo_dir() -> Path:
    """Return the configured foundation repo directory."""
    env = os.environ.get(config.FOUNDATION_REPO_ENV_VAR)
    return Path(env).expanduser() if env else Path(config.FOUNDATION_REPO_DIR).expanduser()


def foundation_manifest_path() -> Path:
    """Return the active foundation manifest path."""
    return foundation_repo_dir() / "foundation_manifest.json"


def ensure_foundation_repo_layout(*, initialise_git: bool = False) -> Path:
    """Create the local foundation repo layout and helper metadata files.

    The foundation repo is intentionally separate from the app repo. It can be
    pushed to its own private Git repository, and it is the only place where the
    long-lived base checkpoint should be versioned.
    """
    repo = foundation_repo_dir()
    (repo / "checkpoints").mkdir(parents=True, exist_ok=True)

    readme = repo / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Sonna Editor Foundation Model\n\n"
            "Private repository for the active Sonna Editor foundation checkpoint.\n"
            "Do not store RAW photos or generated training datasets here.\n\n"
            "Files:\n"
            "- `foundation_manifest.json`: points to the active checkpoint.\n"
            "- `checkpoints/*.ckpt`: versioned foundation checkpoints.\n"
            "- `checkpoints/*.json`: matching checkpoint sidecars.\n",
            encoding="utf-8",
        )

    gitignore = repo / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Keep this repo focused on the foundation model only.\n"
            "*.tmp\n"
            "__pycache__/\n"
            ".DS_Store\n"
            "training_runs/\n",
            encoding="utf-8",
        )

    gitattributes = repo / ".gitattributes"
    if not gitattributes.exists():
        gitattributes.write_text(
            "*.ckpt filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )

    if initialise_git and not (repo / ".git").exists():
        subprocess.run(["git", "init"], cwd=repo, check=True)

    return repo


def _resolve_relative_to_repo(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = foundation_repo_dir() / path
    return path


def resolve_foundation_checkpoint() -> Path:
    """Find the active foundation checkpoint.

    Resolution order:
    1. ``SONNA_FOUNDATION_CHECKPOINT`` absolute or relative path.
    2. ``foundation_manifest.json`` with ``active_checkpoint``.
    3. Legacy fallback ``foundation.ckpt`` in the foundation repo.
    """
    env_ckpt = os.environ.get(config.FOUNDATION_CHECKPOINT_ENV_VAR)
    if env_ckpt:
        path = Path(env_ckpt).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SONNA_FOUNDATION_CHECKPOINT does not exist: {path}")
        return path

    manifest_path = foundation_manifest_path()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid foundation manifest JSON: {manifest_path}") from exc
        active = manifest.get("active_checkpoint")
        if not active:
            raise ValueError(f"Foundation manifest missing active_checkpoint: {manifest_path}")
        path = _resolve_relative_to_repo(str(active))
        if not path.exists():
            raise FileNotFoundError(f"Active foundation checkpoint does not exist: {path}")
        return path

    fallback = foundation_repo_dir() / "foundation.ckpt"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Foundation checkpoint not found. Set SONNA_FOUNDATION_CHECKPOINT, "
        f"or create {manifest_path} with active_checkpoint, or place foundation.ckpt "
        f"in {foundation_repo_dir()}."
    )


def write_foundation_manifest(
    *,
    checkpoint_path: Path,
    sidecar_path: Path | None,
    display_name: str,
    source_run_dir: Path,
) -> Path:
    """Write the foundation repo manifest pointing at a versioned checkpoint."""
    repo = foundation_repo_dir()
    repo.mkdir(parents=True, exist_ok=True)

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    payload = {
        "active_checkpoint": rel(checkpoint_path),
        "active_sidecar": rel(sidecar_path) if sidecar_path is not None else None,
        "display_name": display_name,
        "source_run_dir": rel(source_run_dir),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    manifest_path = foundation_manifest_path()
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def promote_foundation_checkpoint(
    *,
    source_ckpt: Path,
    display_name: str,
    version_stem: str,
    source_run_dir: Path,
) -> Path:
    """Copy a trained checkpoint into the foundation repo and mark it active."""
    repo = ensure_foundation_repo_layout()
    checkpoints_dir = repo / "checkpoints"

    dest = checkpoints_dir / f"{version_stem}.ckpt"
    if dest.exists() or dest.with_suffix(".json").exists():
        raise FileExistsError(f"Foundation checkpoint already exists: {dest}")

    shutil.copy2(source_ckpt, dest)
    source_sidecar = source_ckpt.with_suffix(".json")
    dest_sidecar: Path | None = None
    if source_sidecar.exists():
        dest_sidecar = dest.with_suffix(".json")
        shutil.copy2(source_sidecar, dest_sidecar)

    write_foundation_manifest(
        checkpoint_path=dest,
        sidecar_path=dest_sidecar,
        display_name=display_name,
        source_run_dir=source_run_dir,
    )
    return dest
