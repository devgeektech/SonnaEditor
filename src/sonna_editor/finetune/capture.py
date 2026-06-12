"""Edit capture — compare model predictions against user's final Lightroom edits."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import numpy as np
import pandas as pd
from PIL import Image

from sonna_editor import config
from sonna_editor.data.extract import (
    compute_histogram,
    compute_scene_statistics,
    extract_metadata,
    extract_preview,
)
from sonna_editor.data.xmp import read_xmp
from sonna_editor.inference.pipeline import RAW_EXTENSIONS

# Source labels for the deltas JSON — describes WHY a delta is or isn't computable.
_SOURCE_USER_FINAL = "user_final"      # field in final XMP; delta = final - predicted
_SOURCE_MODEL_FILTERED = "model_filtered"  # model predicted it but we didn't write it to XMP
_SOURCE_LR_DEFAULT = "lr_default"     # field absent from XMP; Lightroom used its own default


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _histogram_to_bytes(hist: np.ndarray) -> bytes:
    """Serialise (3, 32) float32 histogram to bytes (matching training Parquet format)."""
    buf = io.BytesIO()
    np.save(buf, hist)
    return buf.getvalue()


def _parse_iso_datetime(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _xmp_mtime(xmp_path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(xmp_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _compute_edit_lag(
    run_timestamp: str,
    xmp_mtime_dt: Optional[datetime],
) -> Optional[float]:
    run_dt = _parse_iso_datetime(run_timestamp)
    if run_dt is None or xmp_mtime_dt is None:
        return None
    run_dt = run_dt.astimezone(timezone.utc)
    return (xmp_mtime_dt - run_dt).total_seconds()


def _float_or_none(value: object) -> float | None:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _build_deltas_json(
    predicted: dict[str, float],
    final: Mapping[str, object],
    v1_skip_fields: frozenset[str],
) -> str:
    """
    Build the deltas JSON string for one photo.

    Iterates ``predicted.keys()`` — that's the canonical field set for the
    model that produced these predictions (135 for v1 ckpts, 147 for v2).
    Iterating config.SLIDER_FIELDS unconditionally was the bug: for a v1
    predictions dict the 12 v2-extension fields fell through the KeyError
    catch and got marked source="lr_default", silently mis-attributing the
    captured edits.

    For each field present in `predicted`:
    - source="user_final"     if field was in final XMP → delta = final - predicted
    - source="model_filtered" if field is in v1_skip_fields (we didn't write it)
    - source="lr_default"     if field absent from XMP and not skip-listed
    """
    result: dict[str, dict] = {}
    for field in predicted:
        final_val = final.get(field)
        if final_val is not None:
            final_float = _float_or_none(final_val)
            predicted_float = _float_or_none(predicted.get(field))
            delta = (
                final_float - predicted_float
                if final_float is not None and predicted_float is not None
                else None
            )
            result[field] = {"delta": delta, "source": _SOURCE_USER_FINAL}
        elif field in v1_skip_fields:
            result[field] = {"delta": None, "source": _SOURCE_MODEL_FILTERED}
        else:
            result[field] = {"delta": None, "source": _SOURCE_LR_DEFAULT}
    return json.dumps(result)


def capture_user_edits(
    shoot_dir: Path,
    model_predictions_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compare model predictions against the user's final Lightroom edits for a shoot.

    Reads `sonna_predictions.json` (written by process_shoot_with_model) and the
    current XMP sidecars in shoot_dir (which now represent the user's final values
    after any Lightroom tweaks). For each photo, captures all 135 slider deltas
    with provenance metadata.

    Args:
        shoot_dir:               Folder containing RAW files and final XMP sidecars.
        model_predictions_path:  Path to sonna_predictions.json from the inference run.
        output_dir:              Directory to write thumbnails/ and captures.parquet.

    Returns:
        DataFrame with training-compatible columns (metadata + 135 final slider values)
        plus provenance/delta columns. Also written to output_dir/captures.parquet.
    """
    # --- Load prediction sidecar ---
    sidecar = json.loads(model_predictions_path.read_text())
    predictions_by_file: dict[str, dict[str, float]] = sidecar["photos"]
    v1_skip_fields: frozenset[str] = frozenset(sidecar.get("v1_skip_fields", []))
    run_timestamp: str = sidecar.get("run_timestamp", "")
    model_version: str = sidecar.get("model_version", "unknown")
    model_path_str: str = sidecar.get("model_path", "")

    # --- Prepare output directories ---
    thumb_dir = output_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # --- Walk shoot_dir for matching RAW files ---
    rows: list[dict] = []

    for raw_path in sorted(shoot_dir.iterdir()):
        if not raw_path.is_file():
            continue
        if raw_path.suffix.lower() not in RAW_EXTENSIONS:
            continue
        if raw_path.name not in predictions_by_file:
            continue  # photo wasn't in this inference run

        # Find adjacent XMP (case-insensitive suffix)
        xmp_path = raw_path.with_suffix(".xmp")
        if not xmp_path.exists():
            xmp_path = raw_path.with_suffix(".XMP")
        if not xmp_path.exists():
            continue  # no final XMP — user hasn't processed this photo yet

        # --- Read final slider values ---
        final = read_xmp(xmp_path)

        # --- Timestamps and edit lag ---
        xmp_mtime_dt = _xmp_mtime(xmp_path)
        xmp_modified_time = xmp_mtime_dt.isoformat() if xmp_mtime_dt else None
        edit_lag_seconds = _compute_edit_lag(run_timestamp, xmp_mtime_dt)

        # --- Extract preview + metadata ---
        try:
            preview: Image.Image = extract_preview(raw_path)
        except Exception:
            preview = Image.new("RGB", (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION))

        thumb_path = thumb_dir / f"{raw_path.stem}.jpg"
        preview.save(thumb_path, format="JPEG", quality=90)

        meta = extract_metadata(raw_path)
        histogram = _histogram_to_bytes(compute_histogram(preview))
        scene_stats = compute_scene_statistics(preview)

        # --- Capture time ---
        capture_dt = meta.get("capture_datetime")
        capture_time = capture_dt.isoformat() if isinstance(capture_dt, datetime) else None

        # --- Build deltas JSON ---
        predicted = predictions_by_file[raw_path.name]
        deltas_json = _build_deltas_json(predicted, final, v1_skip_fields)

        # --- Build row ---
        row: dict = {
            # Core identity
            "id": _sha16(str(raw_path)),
            "file_path": str(raw_path),
            "thumbnail_path": str(thumb_path),
            "shoot_id": shoot_dir.name,
            # Timestamps
            "capture_time": capture_time,
            # Camera metadata
            "camera_body": meta.get("camera_body"),
            "lens_model": meta.get("lens_model"),
            "camera_profile": None,
            "white_balance_preset": meta.get("white_balance_preset"),
            "iso": meta.get("iso"),
            "shutter_speed": meta.get("shutter_speed"),
            "aperture": meta.get("aperture"),
            "focal_length": meta.get("focal_length"),
            # Image features
            "histogram": histogram,
            # Provenance (analysis only — not used for training)
            "model_version": model_version,
            "model_path": model_path_str,
            "prediction_timestamp": run_timestamp,
            "xmp_modified_time": xmp_modified_time,
            "edit_lag_seconds": edit_lag_seconds,
            "predicted_values": json.dumps(predicted),
            "deltas": deltas_json,
        }

        for field in config.SCENE_STAT_FIELDS:
            row[field] = scene_stats.get(field)

        # Final slider values (training targets) — 135 individual columns
        for field in config.SLIDER_FIELDS:
            val = final.get(field)
            row[field] = float(val) if val is not None else None

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "captures.parquet", index=False)

    return df
