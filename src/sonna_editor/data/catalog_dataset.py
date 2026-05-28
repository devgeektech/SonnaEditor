"""Catalog-based dataset builder for training on Lightroom-managed photos.

Reads slider values directly from the Lightroom catalog (Lua/XMP blobs),
so photos don't need XMP sidecars exported first. Selects the most recent
N edited photos by capture date, skips inaccessible RAW files gracefully.
"""
from __future__ import annotations

import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.catalog import (
    DevelopSettingsParseError,
    connect_catalog,
    find_edited_photos,
    get_develop_settings,
)
from sonna_editor.data.dataset import (
    _THUMB_QUALITY,
    _derive_shoot_id,
    _file_id,
    _histogram_to_bytes,
    save_split,
    split_dataset,
)
from sonna_editor.data.extract import compute_histogram, extract_metadata, extract_preview

logger = logging.getLogger(__name__)

# Mirror audit.py thresholds without importing private internals
_SCALAR_SLIDER_FIELDS: list[str] = [f for f in SLIDER_FIELDS if not f.startswith("ToneCurve")]
_UNEDITED_ZERO_THRESHOLD = 80  # ≥80 of 87 scalar sliders at 0.0 → likely unedited


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unedited_dict(sliders: dict) -> bool:
    """Return True if slider dict has ≥ threshold scalar sliders at 0.0 / None."""
    zero_count = sum(
        1 for f in _SCALAR_SLIDER_FIELDS
        if sliders.get(f) is None or sliders.get(f) == 0.0
    )
    return zero_count >= _UNEDITED_ZERO_THRESHOLD


def _parse_capture_time(capture_time: str | None) -> datetime | None:
    if not capture_time:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(capture_time, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(capture_time)
    except (ValueError, TypeError):
        return None


def _process_catalog_row(args: tuple) -> dict | None:
    """Worker: extract thumbnail + metadata from RAW, using pre-read slider values."""
    raw_path_str, capture_time, profile_name, thumbnail_dir_str, sliders = args
    raw_path = Path(raw_path_str)
    thumbnail_dir = Path(thumbnail_dir_str)

    try:
        preview = extract_preview(raw_path)
        metadata = extract_metadata(raw_path)
        histogram = compute_histogram(preview)

        file_id = _file_id(raw_path)
        thumb_path = thumbnail_dir / f"{file_id}.jpg"
        if not thumb_path.exists():
            preview.save(thumb_path, format="JPEG", quality=_THUMB_QUALITY)

        # Prefer EXIF datetime from the RAW; fall back to catalog capture_time
        exif_dt = metadata.get("capture_datetime")
        if isinstance(exif_dt, str):
            try:
                exif_dt = datetime.fromisoformat(exif_dt)
            except ValueError:
                exif_dt = None
        cap_dt = exif_dt or _parse_capture_time(capture_time)

        shoot_id = _derive_shoot_id(cap_dt, metadata.get("camera_body"))

        row: dict = {
            "id": file_id,
            "profile": profile_name,
            "raw_path": raw_path_str,
            "xmp_path": None,  # sliders sourced from catalog, not sidecar
            "thumbnail_path": str(thumb_path),
            "shoot_id": shoot_id,
            "iso": metadata.get("iso"),
            "shutter_speed": metadata.get("shutter_speed"),
            "aperture": metadata.get("aperture"),
            "focal_length": metadata.get("focal_length"),
            "lens_model": metadata.get("lens_model"),
            "camera_body": metadata.get("camera_body"),
            # v1.1.0 separate make/model alongside the legacy combined camera_body.
            "make": metadata.get("make"),
            "model": metadata.get("model"),
            "capture_datetime": cap_dt.isoformat() if cap_dt else None,
            "exposure_compensation": metadata.get("exposure_compensation"),
            "white_balance_preset": metadata.get("white_balance_preset"),
            "camera_profile": metadata.get("camera_profile"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "histogram": _histogram_to_bytes(histogram),
        }

        for field in SLIDER_FIELDS:
            row[field] = sliders.get(field)

        return row

    except Exception as e:
        logger.error("Failed to process %s: %s", raw_path_str, e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_from_catalog(
    catalog_path: Path,
    output_path: Path,
    profile_name: str,
    thumbnail_dir: Path,
    limit: int = 30_000,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a Parquet training dataset from a Lightroom catalog.

    Selects the most recent `limit` photos that have develop settings and pass
    the unedited filter. Missing RAW files are silently skipped and counted.

    Returns:
        (DataFrame with one row per photo, stats dict with skip counts)
    """
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: query catalog, select candidates (main process, no workers) ----
    conn = connect_catalog(catalog_path)
    try:
        logger.info("Querying catalog for all photos with develop settings...")
        all_photos = find_edited_photos(conn)
    except Exception:
        conn.close()
        raise

    edited = [p for p in all_photos if p["has_develop_settings"]]

    # Deduplicate virtual copies: Lightroom can create multiple catalog entries
    # (different image_id) pointing to the same RAW file_path. Keep the highest
    # image_id per path — the most recently created catalog entry.
    best_by_path: dict[str, dict] = {}
    for photo in edited:
        key = str(photo["file_path"])
        if key not in best_by_path or photo["image_id"] > best_by_path[key]["image_id"]:
            best_by_path[key] = photo
    n_virtual_copies_removed = len(edited) - len(best_by_path)
    edited_deduped = list(best_by_path.values())

    # Sort DESC — most recent capture first; None timestamps fall to end
    edited_desc = sorted(edited_deduped, key=lambda p: p["capture_time"] or "", reverse=True)

    logger.info(
        "Catalog: %d total photos, %d with develop settings, %d after virtual-copy dedup"
        " — scanning most recent first",
        len(all_photos), len(edited), len(edited_deduped),
    )

    selected: list[tuple[dict, dict]] = []
    skip_missing = 0
    skip_unedited = 0
    skip_parse_error = 0

    pbar = tqdm(edited_desc, desc="Scanning catalog", unit="photo", dynamic_ncols=True)
    for photo in pbar:
        if len(selected) >= limit:
            break

        if not photo["file_path"].exists():
            skip_missing += 1
            continue

        try:
            sliders = get_develop_settings(conn, photo["image_id"])
        except DevelopSettingsParseError as e:
            logger.debug("Parse error image_id=%d: %s", photo["image_id"], e)
            skip_parse_error += 1
            continue

        if _is_unedited_dict(sliders):
            skip_unedited += 1
            continue

        selected.append((photo, sliders))
        pbar.set_postfix(
            selected=len(selected),
            miss=skip_missing,
            uned=skip_unedited,
        )

    conn.close()

    logger.info(
        "Selected %d photos | Skipped: %d missing, %d unedited, %d parse errors",
        len(selected), skip_missing, skip_unedited, skip_parse_error,
    )

    if not selected:
        raise RuntimeError(
            "No photos selected — check catalog path, SSD mount, and edit coverage."
        )

    # ---- Phase 2: extract thumbnails in parallel ----
    worker_args = [
        (
            str(photo["file_path"]),
            photo["capture_time"],
            profile_name,
            str(thumbnail_dir),
            sliders,
        )
        for photo, sliders in selected
    ]

    rows: list[dict] = []
    failures = 0

    if max_workers == 1:
        iterator = (_process_catalog_row(a) for a in worker_args)
    else:
        pool = multiprocessing.Pool(processes=max_workers)
        iterator = pool.imap(_process_catalog_row, worker_args)

    try:
        for result in tqdm(
            iterator,
            total=len(worker_args),
            desc="Extracting thumbnails",
            unit="photo",
            dynamic_ncols=True,
        ):
            if result is None:
                failures += 1
            else:
                rows.append(result)
    finally:
        if max_workers != 1:
            pool.close()
            pool.join()

    if failures:
        logger.warning("%d/%d thumbnail extractions failed.", failures, len(selected))

    if not rows:
        raise RuntimeError("All thumbnail extractions failed — dataset is empty.")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)

    stats: dict[str, int] = {
        "total_in_catalog": len(all_photos),
        "total_with_develop_settings": len(edited),
        "skip_virtual_copy": n_virtual_copies_removed,
        "skip_missing": skip_missing,
        "skip_unedited": skip_unedited,
        "skip_parse_error": skip_parse_error,
        "skip_extraction_error": failures,
        "included": len(rows),
    }
    return df, stats
