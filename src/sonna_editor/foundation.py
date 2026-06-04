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
from sonna_editor.training.image_foundation import FOUNDATION_IMAGE_TYPE

FOUNDATION_MANIFEST_SCHEMA_VERSION = 2
DEFAULT_FOUNDATION_TYPE = "sonna_editor_slider_regression"
HYBRID_FOUNDATION_TYPE = "hybrid_multitask"
IMAGE_DECODER_STATE_KEY = "image_decoder_state"
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
        or (HYBRID_FOUNDATION_TYPE if IMAGE_DECODER_STATE_KEY in ckpt else None)
        or DEFAULT_FOUNDATION_TYPE
    )


def _capabilities_for_foundation_type(foundation_type: str) -> list[str]:
    if foundation_type == FOUNDATION_IMAGE_TYPE:
        return ["backbone_features", "image_decoder"]
    if foundation_type == HYBRID_FOUNDATION_TYPE:
        return [
            "backbone_features",
            "metadata_encoder",
            "slider_regression",
            "image_decoder",
        ]
    return ["backbone_features", "metadata_encoder", "slider_regression"]


def _trained_on_for_foundation_type(foundation_type: str) -> list[str]:
    if foundation_type == FOUNDATION_IMAGE_TYPE:
        return ["raw_dng_tiff"]
    if foundation_type == HYBRID_FOUNDATION_TYPE:
        return ["raw_xmp", "raw_dng_tiff"]
    return ["raw_xmp"]


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


def _extract_decoder_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    decoder_state = payload.get(IMAGE_DECODER_STATE_KEY)
    if isinstance(decoder_state, dict):
        return {
            str(key): value
            for key, value in decoder_state.items()
            if isinstance(value, torch.Tensor)
        }
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        return {}
    return {
        str(key): value
        for key, value in model_state.items()
        if isinstance(key, str)
        and isinstance(value, torch.Tensor)
        and key.startswith("decoder.")
    }


def _copy_compatible_state(
    *,
    destination: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in source.items()
        if key.startswith(prefixes)
        and key in destination
        and isinstance(value, torch.Tensor)
        and destination[key].shape == value.shape
    }


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
    for entry in manifest.get("history") or []:
        if not isinstance(entry, dict):
            continue
        checkpoint = entry.get("checkpoint")
        if not checkpoint:
            continue
        checkpoint_str = str(checkpoint)
        version = Path(checkpoint_str).stem
        versions.append({
            "version": version,
            "checkpoint": checkpoint_str,
            "sidecar": entry.get("sidecar"),
            "display_name": entry.get("display_name") or version,
            "source_run_dir": entry.get("source_run_dir"),
            "created_at": entry.get("updated_at"),
            "foundation_type": entry.get("foundation_type") or DEFAULT_FOUNDATION_TYPE,
            "capabilities": _capabilities_for_foundation_type(
                str(entry.get("foundation_type") or DEFAULT_FOUNDATION_TYPE)
            ),
            "trained_on": _trained_on_for_foundation_type(
                str(entry.get("foundation_type") or DEFAULT_FOUNDATION_TYPE)
            ),
        })
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


def carry_foundation_auxiliary_state(
    *,
    source_checkpoint: Path | None,
    destination_checkpoint: Path,
    trained_on: list[str] | None = None,
) -> bool:
    """Carry non-SonnaEditor foundation heads into a newly saved Sonna checkpoint.

    RAW+XMP training saves a native `SonnaEditor` checkpoint. When it warm-starts
    from a hybrid/image foundation checkpoint, this helper preserves the image
    decoder and marks the result as a hybrid foundation checkpoint so later TIFF
    training can keep using the same active file.
    """
    if source_checkpoint is None or not source_checkpoint.exists():
        return False
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        return False
    decoder_state = _extract_decoder_state(source)
    if not decoder_state:
        return False

    destination = torch.load(destination_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(destination, dict):
        return False
    destination[IMAGE_DECODER_STATE_KEY] = decoder_state
    destination["foundation_type"] = HYBRID_FOUNDATION_TYPE
    destination["slider_heads_trained"] = True
    arch_config = destination.setdefault("arch_config", {})
    if isinstance(arch_config, dict):
        arch_config["foundation_type"] = HYBRID_FOUNDATION_TYPE
        arch_config["slider_heads_trained"] = True
    destination["foundation_trained_on"] = _merge_training_sources(
        source.get("foundation_trained_on"),
        (source.get("arch_config") or {}).get("foundation_trained_on")
        if isinstance(source.get("arch_config"), dict)
        else None,
        _trained_on_for_foundation_type(_foundation_type_from_checkpoint(source_checkpoint)),
        trained_on,
    )
    if "image_metrics" in source:
        destination["image_metrics"] = source["image_metrics"]
    torch.save(destination, destination_checkpoint)
    return True


def save_hybrid_foundation_checkpoint(
    *,
    image_checkpoint: Path,
    output_checkpoint: Path,
    base_checkpoint: Path | None,
    image_resolution: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    metrics: dict[str, float],
) -> Path:
    """Save one foundation checkpoint containing slider model + image decoder.

    TIFF/image training updates the ConvNeXt backbone and decoder. This function
    writes those learned visual weights into a `SonnaEditor` checkpoint while
    preserving slider heads from the previous active foundation checkpoint when
    they exist.
    """
    from sonna_editor.model.architecture import SonnaEditor

    image_payload = torch.load(image_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(image_payload, dict) or "model_state" not in image_payload:
        raise ValueError(f"Invalid image foundation checkpoint: {image_checkpoint}")
    image_state = image_payload["model_state"]
    if not isinstance(image_state, dict):
        raise ValueError(f"Image foundation checkpoint has invalid model_state: {image_checkpoint}")

    base_payload: dict[str, Any] | None = None
    if base_checkpoint is not None and base_checkpoint.exists():
        loaded = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            base_payload = loaded

    if base_payload is not None and "registry" in base_payload:
        sonna_model = SonnaEditor.from_checkpoint(base_checkpoint)
    else:
        sonna_model = SonnaEditor(
            _pretrained_backbone=False,
            arch_version=3,
            slider_set_version=config.CURRENT_SLIDER_SET_VERSION,
            use_wb_metadata_skip=True,
        )

    current_state = sonna_model.state_dict()
    backbone_state = _copy_compatible_state(
        destination=current_state,
        source=image_state,
        prefixes=("backbone_features.",),
    )
    sonna_model.load_state_dict(backbone_state, strict=False)
    sonna_model.save_checkpoint(output_checkpoint)

    hybrid_payload = torch.load(output_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(hybrid_payload, dict):
        raise ValueError(f"Invalid Sonna checkpoint after save: {output_checkpoint}")

    decoder_state = _extract_decoder_state(image_payload)
    hybrid_payload[IMAGE_DECODER_STATE_KEY] = decoder_state
    hybrid_payload["foundation_type"] = HYBRID_FOUNDATION_TYPE
    hybrid_payload["image_metrics"] = metrics
    base_type = (
        _foundation_type_from_checkpoint(base_checkpoint)
        if base_checkpoint is not None and base_checkpoint.exists()
        else None
    )
    base_slider_heads_trained = True
    if base_payload is not None:
        base_arch = base_payload.get("arch_config") or {}
        if isinstance(base_arch, dict):
            base_slider_heads_trained = bool(
                base_payload.get("slider_heads_trained", base_arch.get("slider_heads_trained", True))
            )
    hybrid_payload["slider_heads_trained"] = bool(
        base_payload is not None
        and "registry" in base_payload
        and base_type != FOUNDATION_IMAGE_TYPE
        and base_slider_heads_trained
    )
    hybrid_payload["foundation_trained_on"] = _merge_training_sources(
        base_payload.get("foundation_trained_on") if base_payload else None,
        (base_payload.get("arch_config") or {}).get("foundation_trained_on")
        if base_payload and isinstance(base_payload.get("arch_config"), dict)
        else None,
        _trained_on_for_foundation_type(_foundation_type_from_checkpoint(base_checkpoint))
        if base_checkpoint is not None and base_checkpoint.exists()
        else None,
        ["raw_dng_tiff"],
    )
    arch_config = hybrid_payload.setdefault("arch_config", {})
    if isinstance(arch_config, dict):
        arch_config["foundation_type"] = HYBRID_FOUNDATION_TYPE
        arch_config["slider_heads_trained"] = hybrid_payload["slider_heads_trained"]
        arch_config["image_resolution"] = image_resolution
        arch_config["image_train_rows"] = train_rows
        arch_config["image_val_rows"] = val_rows
        arch_config["image_test_rows"] = test_rows
        arch_config["foundation_trained_on"] = hybrid_payload["foundation_trained_on"]
    torch.save(hybrid_payload, output_checkpoint)
    return output_checkpoint


def foundation_requires_slider_prior_initialisation(path: Path) -> bool:
    """Return True when a foundation warm start has untrained slider heads."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False
    foundation_type = _foundation_type_from_checkpoint(path)
    if foundation_type == FOUNDATION_IMAGE_TYPE:
        return True
    if foundation_type != HYBRID_FOUNDATION_TYPE:
        return False
    arch_config = ckpt.get("arch_config") or {}
    slider_heads_trained = ckpt.get("slider_heads_trained")
    if slider_heads_trained is None and isinstance(arch_config, dict):
        slider_heads_trained = arch_config.get("slider_heads_trained")
    return slider_heads_trained is False


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
) -> Any:
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
