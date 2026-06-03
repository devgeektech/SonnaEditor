"""Pydantic request/response models for the Sonna Editor API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


# ── Profiles ────────────────────────────────────────────────────────────────

class Profile(BaseModel):
    id: str
    name: str
    version: str
    checkpoint_path: str
    trained_at: Optional[str] = None
    photo_count: Optional[int] = None
    val_loss: Optional[float] = None
    is_active: bool = False
    # Frontend should prefer `display_name` when present; else fall back to
    # `f"{name} {version}"`. Lets sidecar JSONs override the hardcoded "DP Event"
    # default for variants (smoke / candidate / production) sharing one version.
    display_name: Optional[str] = None
    # Resolution the profile was trained at; informational only. The inference
    # engine reads the same value directly from the sidecar to size its
    # augmentation pipeline.
    resolution: Optional[int] = None
    # Default user skip set when this profile is selected. The frontend
    # initialises its skip-fields toggle to this when the user picks the
    # profile; the user can still override per-job.
    default_skip_fields: list[str] = Field(default_factory=list)
    # Sidecar-sourced marker for the profile's provenance. `None` for Mode A
    # trained ckpts (v1.2.3 production and earlier sidecars predate the field);
    # `"mode_b_initial"` for preset-derived Mode B ckpts (see
    # mode_b/checkpoint_builder.py:PROFILE_TYPE). Future categories
    # ("mode_b_finetuned", etc.) flow through without code changes. The UI in
    # P5 treats `None` as "Personal AI" by default.
    profile_type: Optional[str] = None


# ── Folder scan ─────────────────────────────────────────────────────────────

class FolderScanRequest(BaseModel):
    folder_path: str


class RawFileEntry(BaseModel):
    name: str
    size_bytes: int


class FolderScanResponse(BaseModel):
    folder_path: str
    raw_count: int
    files: list[RawFileEntry]
    is_valid: bool
    error: Optional[str] = None
    truncated: bool = False
    # Count of .xmp sidecars in the folder whose basename matches one of the
    # discovered RAW files. These are the sidecars that a Saha run would
    # overwrite — the frontend uses this to gate Process Folders behind a
    # confirmation dialog. Stray .xmp files unrelated to any RAW are not
    # counted (they're invisible to the inference pipeline's write path).
    xmp_conflict_count: int = 0


class RecentFolder(BaseModel):
    name: str
    path: str
    raw_count: int
    last_processed_at: Optional[str] = None


# ── Captures ────────────────────────────────────────────────────────────────

class CapturesResponse(BaseModel):
    """Wraps analyse_deltas() output with two extra fields."""

    captures_count: int
    since: Optional[str] = None
    n_photos: int = 0
    metadata_coverage: dict = Field(default_factory=dict)
    per_field: dict = Field(default_factory=dict)
    most_adjusted_fields: list = Field(default_factory=list)
    correlations: list = Field(default_factory=list)
    filtered_field_deltas: dict = Field(default_factory=dict)


# ── Process / Finetune jobs ────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    folder_path: str
    profile_id: str
    confidence_threshold: float = 0.65
    write_xmp_in_place: bool = True
    flag_low_confidence: bool = True
    # Deprecated compat shim. If True, the route adds ["Temperature", "Tint"]
    # to skip_fields. New callers should pass skip_fields directly.
    preserve_wb: bool = False
    # Slider fields the user has toggled off — XMP writer omits them so
    # Lightroom falls back to its defaults (AsShot for Temperature/Tint,
    # zero / no entry for HSL/etc). Model still predicts these; they remain
    # in sonna_predictions.json for finetune capture but don't reach the XMP.
    skip_fields: list[str] = Field(default_factory=list)


class FinetuneRequest(BaseModel):
    base_profile_id: str
    captures_dir: str
    weight_recent: float = 1.0


# ── Personal AI profile creation ────────────────────────────────────────────

class PersonalProfileRequest(BaseModel):
    """Request body for POST /api/profiles/personal."""

    profile_name: str
    input_dir: str
    max_epochs: int = 50
    batch_size: int = 16
    workers: int = 4


# ── Lite profile creation ───────────────────────────────────────────────────

class LiteProfileRequest(BaseModel):
    """Request body for POST /api/profiles/lite."""

    profile_name: str
    # Absolute path to the user's Lightroom preset .xmp. Electron's file
    # picker hands the renderer an absolute path, which is forwarded
    # verbatim. The route copies the file into CHECKPOINTS_DIR so the
    # Lite sidecar's source_preset reference stays stable if the user
    # later moves or deletes the original.
    preset_path: str
    # Answers keyed by survey question. Current UI asks exposure /
    # temperature / tint and fills legacy look-slider answers with zero so
    # older survey serialization remains compatible.
    survey_answers: dict[str, int] = Field(default_factory=dict)


class LiteProfileCreated(BaseModel):
    """Response body for POST /api/profiles/lite."""

    profile_id: str
    ckpt_path: str
    sidecar_path: str


class DeleteProfileResponse(BaseModel):
    """Response body for DELETE /api/profiles/{profile_id}."""

    profile_id: str
    deleted_paths: list[str]


class JobAck(BaseModel):
    """Returned by POST /api/process and POST /api/finetune."""

    job_id: str
    state: str


class JobSnapshot(BaseModel):
    """Unified snapshot for both kinds — fields not relevant to a kind are None.

    Discriminator is `kind` ("process" | "finetune"). v1 deliberately uses one
    flat model rather than a discriminated union: simpler client code and the
    field set is small.
    """

    job_id: str
    kind: str
    state: str
    started_at: str
    ended_at: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    uncertainty_enabled: bool = False

    # process-only
    folder_path: Optional[str] = None
    profile_id: Optional[str] = None
    photos_total: Optional[int] = None
    photos_processed: Optional[int] = 0
    photos_flagged: Optional[int] = 0
    photos_failed: Optional[int] = 0
    current_photo: Optional[str] = None
    photos_per_sec: Optional[float] = 0.0
    eta_seconds: Optional[int] = 0
    output_paths_so_far: Optional[list[str]] = None

    # finetune-only
    base_profile_id: Optional[str] = None
    captures_dir: Optional[str] = None
    profile_name: Optional[str] = None
    dataset_dir: Optional[str] = None
    epochs_total: Optional[int] = None
    epochs_completed: Optional[int] = 0
    current_epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    new_checkpoint_path: Optional[str] = None
