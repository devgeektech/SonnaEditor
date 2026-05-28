"""Profile discovery + activation + Lite profile creation.

Reads .ckpt files directly from config.CHECKPOINTS_DIR. Phase 6 will replace this
with a proper registry while keeping the response shape stable.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Optional

from fastapi import APIRouter, HTTPException

from sonna_editor import config
from sonna_editor.api.models import (
    DeleteProfileResponse,
    LiteProfileCreated,
    LiteProfileRequest,
    Profile,
)
from sonna_editor.mode_b import checkpoint_builder as mode_b_builder
from sonna_editor.mode_b.survey import (
    QUESTION_ORDER,
    VALID_ANSWERS,
    build_survey_payload,
    write_survey,
)

router = APIRouter()

# Matches "model-v1.0.1.ckpt" → ("1", "0", "1"). Lightning intermediates from
# v1_learning/checkpoints/ (e.g. "epoch=017-val_loss=0.0010.ckpt") deliberately
# don't match — they are training artefacts, not user-facing profiles.
_VERSION_RE = re.compile(r"^model-v(\d+)\.(\d+)\.(\d+)(?:-(\w+))?\.ckpt$")

_ACTIVE_PROFILE_FILE = Path.home() / ".saha" / "active_profile.txt"

# Used as Profile.name when a ckpt's sidecar lacks display_name — covers the
# v1.x Mode A production ckpts whose sidecars predate the field. Mode B and
# future ckpts write display_name verbatim from the user's profile name.
LEGACY_PROFILE_NAME_FALLBACK: Final[str] = "DP Event"


def _checkpoint_id(version_label: str) -> str:
    # "v1.0.1" → "dp-event-v1.0.1". Legacy fallback for sidecars that don't
    # carry profile_id (i.e. Mode A v1.x production). Mode B sidecars write a
    # slug like "mode-b-wedding-lite-20260514-1102" and _build_profile prefers
    # that over this fallback.
    return f"dp-event-{version_label}"


def _profile_name(sidecar: dict) -> str:
    # Modern sidecars (Mode B + any future schema) carry display_name
    # verbatim from the user's profile name. Legacy Mode A v1.x sidecars
    # predate the field and fall back to the "DP Event" lineage label.
    return sidecar.get("display_name") or LEGACY_PROFILE_NAME_FALLBACK


def _read_active_id() -> Optional[str]:
    try:
        return _ACTIVE_PROFILE_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _write_active_id(profile_id: str) -> None:
    _ACTIVE_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_PROFILE_FILE.write_text(profile_id)


def _read_sidecar(ckpt_path: Path) -> dict:
    sidecar = ckpt_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _file_mtime_iso(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


def _build_profile(ckpt_path: Path, active_id: Optional[str]) -> Optional[Profile]:
    m = _VERSION_RE.match(ckpt_path.name)
    if not m:
        return None
    version_label = f"v{m.group(1)}.{m.group(2)}.{m.group(3)}"

    sidecar = _read_sidecar(ckpt_path)
    # Prefer sidecar-provided id (Mode B writes a slug like
    # "mode-b-wedding-lite-20260514-1102"; future schemas can carry any
    # stable id). Falls back to the legacy "dp-event-{version}" scheme so
    # Mode A v1.x profiles keep their existing ids and the active-profile
    # file written before this change still resolves.
    raw_profile_id = sidecar.get("profile_id")
    profile_id = str(raw_profile_id) if raw_profile_id else _checkpoint_id(version_label)

    trained_at = sidecar.get("date_iso") or sidecar.get("train_date") or _file_mtime_iso(ckpt_path)
    val_loss = sidecar.get("ft_val_loss") or sidecar.get("val_loss")
    photo_count = sidecar.get("n_capture_rows")
    if photo_count is not None and sidecar.get("n_original_rows") is not None:
        photo_count = int(photo_count) + int(sidecar["n_original_rows"])
    elif sidecar.get("train_rows") is not None:
        photo_count = int(sidecar["train_rows"])

    display_name = sidecar.get("display_name")
    resolution = sidecar.get("resolution")
    if resolution is not None:
        try:
            resolution = int(resolution)
        except (TypeError, ValueError):
            resolution = None

    raw_skip = sidecar.get("default_skip_fields") or []
    default_skip_fields = [str(f) for f in raw_skip] if isinstance(raw_skip, list) else []

    raw_profile_type = sidecar.get("profile_type")
    profile_type = str(raw_profile_type) if raw_profile_type is not None else None

    return Profile(
        id=profile_id,
        name=_profile_name(sidecar),
        version=version_label,
        checkpoint_path=str(ckpt_path.resolve()),
        trained_at=trained_at,
        photo_count=photo_count,
        val_loss=val_loss,
        is_active=(active_id == profile_id),
        display_name=display_name,
        resolution=resolution,
        default_skip_fields=default_skip_fields,
        profile_type=profile_type,
    )


def _discover_profiles() -> list[Profile]:
    """Scan CHECKPOINTS_DIR (non-recursive) for model-v*.ckpt files."""
    if not config.CHECKPOINTS_DIR.is_dir():
        return []

    active_id = _read_active_id()
    discovered: list[Profile] = []
    for ckpt_path in config.CHECKPOINTS_DIR.glob("model-v*.ckpt"):
        profile = _build_profile(ckpt_path, active_id)
        if profile is not None:
            discovered.append(profile)

    # Sort by version descending so newest is first.
    def version_tuple(p: Profile) -> tuple[int, int, int]:
        m = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", p.version)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)

    discovered.sort(key=version_tuple, reverse=True)

    # If no profile was marked active, default to the most recently trained.
    if discovered and not any(p.is_active for p in discovered):
        newest = max(
            discovered,
            key=lambda p: datetime.fromisoformat(
                (p.trained_at or "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")
            ),
        )
        newest.is_active = True

    return discovered


@router.get("/profiles", response_model=list[Profile])
def list_profiles() -> list[Profile]:
    return _discover_profiles()


@router.post("/profiles/{profile_id}/activate", response_model=Profile)
def activate_profile(profile_id: str) -> Profile:
    discovered = _discover_profiles()
    match = next((p for p in discovered if p.id == profile_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    _write_active_id(profile_id)
    # Recompute active flags so the returned object reflects the write.
    for p in discovered:
        p.is_active = (p.id == profile_id)
    return next(p for p in discovered if p.id == profile_id)


@router.delete("/profiles/{profile_id}", response_model=DeleteProfileResponse)
def delete_profile(profile_id: str) -> DeleteProfileResponse:
    discovered = _discover_profiles()
    match = next((p for p in discovered if p.id == profile_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    if match.is_active:
        raise HTTPException(
            status_code=409,
            detail="Deactivate this profile first before deleting it.",
        )

    ckpt_path = Path(match.checkpoint_path)
    deleted_paths: list[str] = []

    candidate_paths = [
        ckpt_path,
        ckpt_path.with_suffix(".json"),
        ckpt_path.with_name(f"{ckpt_path.stem}-survey.json"),
        ckpt_path.with_name(f"{ckpt_path.stem}-preset.xmp"),
    ]

    for path in candidate_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        if path.exists():
            continue
        deleted_paths.append(str(path))

    return DeleteProfileResponse(profile_id=profile_id, deleted_paths=deleted_paths)


# ── Lite profile creation (Mode B) ──────────────────────────────────────────

# Matches the Mode B output ckpt naming scheme committed in B1:
# model-v0.<seq>.0.ckpt where <seq> is a monotonic counter. The v0.x range
# carves out a clean separation from v1.x trained profiles.
_LITE_VERSION_RE = re.compile(r"^model-v0\.(\d+)\.0\.ckpt$")


def _next_lite_seq() -> int:
    """Return max(<seq>) + 1 across existing v0.<seq>.0 ckpts, or 1 if none."""
    if not config.CHECKPOINTS_DIR.is_dir():
        return 1
    seen: list[int] = []
    for p in config.CHECKPOINTS_DIR.glob("model-v0.*.0.ckpt"):
        m = _LITE_VERSION_RE.match(p.name)
        if m:
            seen.append(int(m.group(1)))
    return (max(seen) + 1) if seen else 1


def _validate_survey_answers(answers: dict[str, int]) -> None:
    """Raise HTTPException(400) if the survey payload is incomplete/invalid."""
    missing = set(QUESTION_ORDER) - set(answers.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Survey answers missing for: {sorted(missing)}",
        )
    extra = set(answers.keys()) - set(QUESTION_ORDER)
    if extra:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown survey keys: {sorted(extra)}",
        )
    for key, value in answers.items():
        if value not in VALID_ANSWERS:
            raise HTTPException(
                status_code=400,
                detail=f"Survey answer for {key!r} must be one of {list(VALID_ANSWERS)}, got {value}",
            )


def _find_active_mode_a(profiles: list[Profile]) -> Optional[Profile]:
    """Return the active profile if it's a Mode A (Personal AI) ckpt, else None."""
    for p in profiles:
        if p.is_active and p.profile_type != "mode_b_initial":
            return p
    return None


@router.post("/profiles/lite", response_model=LiteProfileCreated)
def create_lite_profile(req: LiteProfileRequest) -> LiteProfileCreated:
    """Build a Mode B initial checkpoint from a preset + style survey.

    Wraps mode_b.checkpoint_builder.build_mode_b_checkpoint(). Base ckpt is
    sourced from the currently-active Personal AI profile (Decision C1
    from the P5 design). The user's preset file is copied into
    CHECKPOINTS_DIR so the sidecar's source_preset path stays stable if
    the original is moved or deleted; the same is done for the survey JSON.
    """
    name = req.profile_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="profile_name must be non-empty")

    preset_path = Path(req.preset_path)
    if not preset_path.is_absolute():
        raise HTTPException(status_code=400, detail="preset_path must be absolute")
    if not preset_path.exists():
        raise HTTPException(status_code=400, detail=f"Preset not found: {preset_path}")
    if not preset_path.is_file():
        raise HTTPException(status_code=400, detail="preset_path is not a file")
    if preset_path.suffix.lower() != ".xmp":
        raise HTTPException(status_code=400, detail="Preset must be a .xmp file")

    _validate_survey_answers(req.survey_answers)

    profiles = _discover_profiles()
    active = _find_active_mode_a(profiles)
    if active is None:
        raise HTTPException(
            status_code=400,
            detail="Activate a Personal AI profile first — Lite profiles are derived from a trained base checkpoint.",
        )

    base_ckpt_path = Path(active.checkpoint_path)
    if not base_ckpt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Active profile's checkpoint missing on disk: {base_ckpt_path}",
        )

    if not config.CHECKPOINTS_DIR.is_dir():
        config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    seq = _next_lite_seq()
    stem = f"model-v0.{seq}.0"
    output_ckpt_path = config.CHECKPOINTS_DIR / f"{stem}.ckpt"
    survey_dest = config.CHECKPOINTS_DIR / f"{stem}-survey.json"
    preset_dest = config.CHECKPOINTS_DIR / f"{stem}-preset.xmp"

    # Persist survey + preset alongside the new ckpt before invoking the
    # builder. Both paths get recorded in the Mode B sidecar's
    # source_survey / source_preset fields.
    survey_payload = build_survey_payload(req.survey_answers)
    write_survey(survey_payload, survey_dest)
    shutil.copyfile(preset_path, preset_dest)

    try:
        sidecar_path = mode_b_builder.build_mode_b_checkpoint(
            preset_path=preset_dest,
            survey_path=survey_dest,
            base_ckpt_path=base_ckpt_path,
            output_ckpt_path=output_ckpt_path,
            profile_name=name,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Best-effort cleanup so a retry with the same seq doesn't trip over
        # half-written artefacts. The ckpt file may or may not exist
        # depending on where the builder failed.
        for partial in (survey_dest, preset_dest, output_ckpt_path):
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Lite profile build failed: {exc}")

    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    return LiteProfileCreated(
        profile_id=sidecar["profile_id"],
        ckpt_path=str(output_ckpt_path.resolve()),
        sidecar_path=str(Path(sidecar_path).resolve()),
    )
