"""Foundation checkpoint discovery and manifest helpers."""

from __future__ import annotations

import json
import os
import shutil
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sonna_editor import config

FOUNDATION_MANIFEST_SCHEMA_VERSION = 2
DEFAULT_FOUNDATION_TYPE = "sonna_editor_slider_regression"
_FOUNDATION_VERSION_RE = re.compile(r"^foundation-v(\d+)$")


def foundation_repo_dir() -> Path:
    """Return the configured foundation repo directory."""
    env = os.environ.get(config.FOUNDATION_REPO_ENV_VAR)
    return Path(env).expanduser() if env else Path(config.FOUNDATION_REPO_DIR).expanduser()


def foundation_manifest_path() -> Path:
    """Return the active foundation manifest path."""
    return foundation_repo_dir() / "foundation_manifest.json"


def ensure_foundation_repo_layout() -> Path:
    """Create the local foundation folder layout and helper metadata files.

    By default the foundation folder lives under the project root so a fresh
    clone is self-contained. The parent SonnaEditor repo tracks this folder,
    with checkpoint binaries routed through Git LFS.
    """
    repo = foundation_repo_dir()
    (repo / "checkpoints").mkdir(parents=True, exist_ok=True)

    readme = repo / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Sonna Editor Foundation Model\n\n"
            "Repo-local folder for the active Sonna Editor foundation checkpoint.\n"
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest() -> dict[str, Any]:
    manifest_path = foundation_manifest_path()
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid foundation manifest JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Foundation manifest must contain a JSON object: {manifest_path}")
    return payload


def _foundation_type_from_checkpoint(checkpoint_path: Path) -> str:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return DEFAULT_FOUNDATION_TYPE
    if not isinstance(ckpt, dict):
        return DEFAULT_FOUNDATION_TYPE
    return str(
        ckpt.get("foundation_type")
        or (ckpt.get("arch_config") or {}).get("foundation_type")
        or DEFAULT_FOUNDATION_TYPE
    )


def _capabilities_for_foundation_type(foundation_type: str) -> list[str]:
    return ["backbone_features", "metadata_encoder", "slider_regression"]


def _trained_on_for_foundation_type(foundation_type: str) -> list[str]:
    return ["lightroom_parameters"]


def _merge_training_sources(*sources: Any) -> list[str]:
    ordered: list[str] = []
    for source in sources:
        if isinstance(source, str):
            candidates = [source]
        elif isinstance(source, list):
            candidates = [str(item) for item in source if item]
        else:
            candidates = []
        for candidate in candidates:
            if candidate not in ordered:
                ordered.append(candidate)
    return ordered


def _normalise_version_stem(version_stem: str) -> str:
    stem = version_stem.removesuffix(".ckpt")
    if not stem:
        raise ValueError("Foundation version stem cannot be empty")
    return stem


def _version_number(version: str) -> int | None:
    match = _FOUNDATION_VERSION_RE.match(version)
    return int(match.group(1)) if match else None


def _next_foundation_version(manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest if manifest is not None else _read_manifest()
    numbers: list[int] = []
    for entry in manifest.get("versions") or []:
        if isinstance(entry, dict):
            version = entry.get("version")
            if isinstance(version, str):
                n = _version_number(version)
                if n is not None:
                    numbers.append(n)
    checkpoints_dir = foundation_repo_dir() / "checkpoints"
    if checkpoints_dir.exists():
        for path in checkpoints_dir.glob("foundation-v*.ckpt"):
            n = _version_number(path.stem)
            if n is not None:
                numbers.append(n)
    return f"foundation-v{max(numbers, default=0) + 1}"


def _legacy_versions_from_history(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    versions: list[dict[str, Any]] = []

    def add_entry(entry: dict[str, Any], *, fallback_display_name: str | None = None) -> None:
        checkpoint = entry.get("checkpoint")
        if not checkpoint:
            return
        checkpoint_str = str(checkpoint)
        version = Path(checkpoint_str).stem
        if any(existing.get("version") == version for existing in versions):
            return
        foundation_type = str(entry.get("foundation_type") or DEFAULT_FOUNDATION_TYPE)
        versions.append({
            "version": version,
            "checkpoint": checkpoint_str,
            "sidecar": entry.get("sidecar"),
            "display_name": entry.get("display_name") or fallback_display_name or version,
            "source_run_dir": entry.get("source_run_dir"),
            "created_at": entry.get("updated_at"),
            "foundation_type": foundation_type,
            "capabilities": _capabilities_for_foundation_type(foundation_type),
            "trained_on": _trained_on_for_foundation_type(foundation_type),
        })

    for entry in manifest.get("history") or []:
        if not isinstance(entry, dict):
            continue

        add_entry(entry)

    active_checkpoint = manifest.get("active_checkpoint")
    if active_checkpoint:
        add_entry(
            {
                "checkpoint": active_checkpoint,
                "sidecar": manifest.get("active_sidecar"),
                "display_name": manifest.get("display_name"),
                "source_run_dir": manifest.get("source_run_dir"),
                "updated_at": manifest.get("updated_at"),
                "foundation_type": manifest.get("foundation_type"),
            },
            fallback_display_name=str(manifest.get("active_version") or ""),
        )
    return versions


def list_foundation_versions() -> list[dict[str, Any]]:
    """Return manifest-tracked foundation versions, oldest to newest."""
    manifest = _read_manifest()
    versions = manifest.get("versions")
    if isinstance(versions, list):
        return [dict(v) for v in versions if isinstance(v, dict)]
    return _legacy_versions_from_history(manifest)


def _version_entry_by_name(version: str) -> dict[str, Any]:
    for entry in list_foundation_versions():
        if entry.get("version") == version:
            return entry
    raise ValueError(f"Foundation version not found in manifest: {version}")


def describe_foundation_checkpoint(path: Path) -> dict[str, Any]:
    """Return provenance metadata for a foundation checkpoint path."""
    path = Path(path).expanduser()
    resolved = path.resolve()
    version: str | None = None
    entry_match: dict[str, Any] | None = None
    for entry in list_foundation_versions():
        checkpoint = entry.get("checkpoint")
        if not checkpoint:
            continue
        try:
            candidate = _resolve_relative_to_repo(str(checkpoint)).resolve()
        except OSError:
            continue
        if candidate == resolved:
            version = str(entry.get("version") or Path(str(checkpoint)).stem)
            entry_match = entry
            break

    foundation_type = (
        str(entry_match.get("foundation_type"))
        if entry_match and entry_match.get("foundation_type")
        else _foundation_type_from_checkpoint(path)
    )
    return {
        "foundation_version": version or path.stem,
        "foundation_checkpoint": str(resolved),
        "foundation_sha256": _sha256(path) if path.exists() else None,
        "foundation_type": foundation_type,
        "foundation_capabilities": (
            entry_match.get("capabilities")
            if entry_match and entry_match.get("capabilities")
            else _capabilities_for_foundation_type(foundation_type)
        ),
        "foundation_trained_on": (
            entry_match.get("trained_on")
            if entry_match and entry_match.get("trained_on")
            else _trained_on_for_foundation_type(foundation_type)
        ),
    }


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

    if foundation_manifest_path().exists():
        manifest = _read_manifest()
        active = manifest.get("active_checkpoint")
        if not active and manifest.get("active_version"):
            active_entry = _version_entry_by_name(str(manifest["active_version"]))
            active = active_entry.get("checkpoint")
        if not active:
            raise ValueError(
                f"Foundation manifest missing active_checkpoint: {foundation_manifest_path()}"
            )
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
        f"or create {foundation_manifest_path()} with active_checkpoint, or place "
        f"foundation.ckpt in {foundation_repo_dir()}."
    )


def write_foundation_manifest(
    *,
    checkpoint_path: Path,
    sidecar_path: Path | None,
    display_name: str,
    source_run_dir: Path,
    version: str | None = None,
) -> Path:
    """Write the foundation repo manifest pointing at a versioned checkpoint."""
    repo = foundation_repo_dir()
    repo.mkdir(parents=True, exist_ok=True)

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    manifest_path = foundation_manifest_path()
    existing = _read_manifest() if manifest_path.exists() else {}
    version = version or Path(checkpoint_path).stem
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    foundation_type = _foundation_type_from_checkpoint(checkpoint_path)
    versions = existing.get("versions")
    if not isinstance(versions, list):
        versions = _legacy_versions_from_history(existing)
    versions = [dict(v) for v in versions if isinstance(v, dict)]
    current_trained_on = _trained_on_for_foundation_type(foundation_type)
    previous_active_trained_on: list[str] = []
    active_version = existing.get("active_version")
    if isinstance(active_version, str):
        for existing_entry in versions:
            if existing_entry.get("version") == active_version:
                previous_active_trained_on = _merge_training_sources(
                    existing_entry.get("trained_on")
                )
                break
    lineage_trained_on = _merge_training_sources(
        previous_active_trained_on,
        existing.get("trained_on"),
        current_trained_on,
    )
    versions = [v for v in versions if v.get("version") != version]
    entry: dict[str, Any] = {
        "version": version,
        "checkpoint": rel(checkpoint_path),
        "sidecar": rel(sidecar_path) if sidecar_path is not None else None,
        "display_name": display_name,
        "source_run_dir": rel(source_run_dir),
        "created_at": updated_at,
        "foundation_type": foundation_type,
        "capabilities": _capabilities_for_foundation_type(foundation_type),
        "trained_on": lineage_trained_on,
        "sha256": _sha256(checkpoint_path) if checkpoint_path.exists() else None,
    }
    versions.append(entry)
    history: list[dict[str, Any]] = [
        {
            "checkpoint": v.get("checkpoint"),
            "sidecar": v.get("sidecar"),
            "display_name": v.get("display_name"),
            "source_run_dir": v.get("source_run_dir"),
            "updated_at": v.get("created_at"),
            "foundation_type": v.get("foundation_type"),
        }
        for v in versions[:-1]
    ]
    payload: dict[str, Any] = {
        "schema_version": FOUNDATION_MANIFEST_SCHEMA_VERSION,
        "active_version": version,
        "active_checkpoint": rel(checkpoint_path),
        "active_sidecar": rel(sidecar_path) if sidecar_path is not None else None,
        "display_name": display_name,
        "source_run_dir": rel(source_run_dir),
        "updated_at": updated_at,
        "foundation_type": foundation_type,
        "capabilities": entry["capabilities"],
        "trained_on": lineage_trained_on,
        "versions": versions,
        # Kept for compatibility with existing docs/tests and quick manual reads.
        "history": history[-20:],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def promote_foundation_checkpoint(
    *,
    source_ckpt: Path,
    display_name: str,
    version_stem: str | None = None,
    source_run_dir: Path,
) -> Path:
    """Copy a trained checkpoint into the foundation repo and mark it active."""
    repo = ensure_foundation_repo_layout()
    checkpoints_dir = repo / "checkpoints"

    version_stem = (
        _normalise_version_stem(version_stem)
        if version_stem is not None
        else _next_foundation_version()
    )
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
        version=version_stem,
    )
    return dest


def rollback_foundation_checkpoint(version: str) -> Path:
    """Set the active foundation checkpoint to an existing manifest version."""
    repo = foundation_repo_dir()
    manifest = _read_manifest()
    entry = _version_entry_by_name(version)
    checkpoint_value = entry.get("checkpoint")
    if not checkpoint_value:
        raise ValueError(f"Foundation version missing checkpoint path: {version}")
    checkpoint_path = _resolve_relative_to_repo(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot roll back to {version}; checkpoint does not exist: {checkpoint_path}"
        )
    sidecar_value = entry.get("sidecar")
    sidecar_path = _resolve_relative_to_repo(str(sidecar_value)) if sidecar_value else None
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    manifest["schema_version"] = FOUNDATION_MANIFEST_SCHEMA_VERSION
    manifest["active_version"] = version
    manifest["active_checkpoint"] = rel(checkpoint_path)
    manifest["active_sidecar"] = rel(sidecar_path) if sidecar_path is not None else None
    manifest["display_name"] = entry.get("display_name") or version
    manifest["source_run_dir"] = entry.get("source_run_dir")
    manifest["updated_at"] = updated_at
    manifest["foundation_type"] = entry.get("foundation_type") or DEFAULT_FOUNDATION_TYPE
    manifest["capabilities"] = entry.get("capabilities") or _capabilities_for_foundation_type(
        str(manifest["foundation_type"])
    )
    manifest["trained_on"] = entry.get("trained_on") or _trained_on_for_foundation_type(
        str(manifest["foundation_type"])
    )
    foundation_manifest_path().write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


