"""Foundation checkpoint discovery and manifest helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sonna_editor import config
from sonna_editor.training.image_foundation import FOUNDATION_IMAGE_TYPE


def foundation_repo_dir() -> Path:
    """Return the configured foundation repo directory."""
    env = os.environ.get(config.FOUNDATION_REPO_ENV_VAR)
    return Path(env).expanduser() if env else Path(config.FOUNDATION_REPO_DIR).expanduser()


def foundation_manifest_path() -> Path:
    """Return the active foundation manifest path."""
    return foundation_repo_dir() / "foundation_manifest.json"


def ensure_foundation_repo_layout(*, initialise_git: bool = False) -> Path:
    """Create the local foundation repo layout and helper metadata files.

    By default the foundation repo lives under the project root so a fresh clone
    is self-contained. Operators can still point SONNA_FOUNDATION_REPO at a
    separate private Git/LFS repo when they want that workflow.
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


def _latest_existing_checkpoint() -> Path | None:
    """Return the newest existing versioned foundation checkpoint, if any."""
    checkpoints_dir = foundation_repo_dir() / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = [path for path in checkpoints_dir.glob("*.ckpt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


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
            fallback = _latest_existing_checkpoint()
            if fallback is not None:
                return fallback
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
    history: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing.get("history"), list):
                history = list(existing["history"])
            if existing.get("active_checkpoint"):
                history.append({
                    "checkpoint": existing.get("active_checkpoint"),
                    "sidecar": existing.get("active_sidecar"),
                    "display_name": existing.get("display_name"),
                    "source_run_dir": existing.get("source_run_dir"),
                    "updated_at": existing.get("updated_at"),
                    "foundation_type": existing.get("foundation_type"),
                })
        except json.JSONDecodeError:
            history = []
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = {}
    if isinstance(ckpt, dict):
        payload["foundation_type"] = ckpt.get("foundation_type") or (
            ckpt.get("arch_config") or {}
        ).get("foundation_type") or "sonna_editor_slider_regression"
    payload["history"] = history[-20:]
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


def is_image_foundation_checkpoint(path: Path) -> bool:
    """Return True when `path` is an image-to-image foundation checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False
    return (
        ckpt.get("foundation_type") == FOUNDATION_IMAGE_TYPE
        or (ckpt.get("arch_config") or {}).get("foundation_type") == FOUNDATION_IMAGE_TYPE
    )


def image_foundation_resolution(path: Path) -> int:
    """Return the training resolution recorded in an image foundation checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch_config = ckpt.get("arch_config") or {}
    return int(arch_config.get("image_resolution") or config.IMAGE_RESOLUTION)


def load_sonna_model_from_foundation_checkpoint(
    checkpoint_path: Path,
    *,
    registry: Any | None = None,
    slider_set_version: str = config.CURRENT_SLIDER_SET_VERSION,
):
    """Load a foundation checkpoint as a `SonnaEditor` instance.

    Full `SonnaEditor` checkpoints load normally. Image-to-image foundation
    checkpoints initialise a fresh `SonnaEditor` and copy only compatible
    ConvNeXt backbone tensors, preserving the RAW+XMP profile-training contract.
    """
    from sonna_editor.model.architecture import SonnaEditor

    checkpoint_path = Path(checkpoint_path)
    if not is_image_foundation_checkpoint(checkpoint_path):
        return SonnaEditor.from_checkpoint(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] = ckpt["model_state"]
    model = SonnaEditor(
        registry=registry,
        freeze_backbone=True,
        _pretrained_backbone=False,
        arch_version=3,
        slider_set_version=slider_set_version,
        use_wb_metadata_skip=True,
    )
    current_state = model.state_dict()
    backbone_state = {
        key: value
        for key, value in state.items()
        if key.startswith("backbone_features.")
        and key in current_state
        and current_state[key].shape == value.shape
    }
    model.load_state_dict(backbone_state, strict=False)
    return model
