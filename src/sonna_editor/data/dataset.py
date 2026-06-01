from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from sonna_editor.config import SCENE_STAT_FIELDS, SLIDER_FIELDS, SUPPORTED_RAW_EXTENSIONS
from sonna_editor.data.extract import extract_all

logger = logging.getLogger(__name__)

# Thumbnail save quality (JPEG)
_THUMB_QUALITY = 90


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _file_id(raw_path: Path) -> str:
    """Stable SHA-256 ID for a RAW file, keyed on its absolute path."""
    return hashlib.sha256(str(raw_path.resolve()).encode()).hexdigest()


def _find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Walk input_dir and return RAW/XMP pairs for supervised training.

    Images without a matching XMP sidecar are skipped so training only sees
    supervised examples with explicit target settings.
    """
    pairs: list[tuple[Path, Path]] = []
    for raw_path in sorted(input_dir.rglob("*")):
        if raw_path.suffix.lower() not in SUPPORTED_RAW_EXTENSIONS:
            continue
        xmp = raw_path.with_suffix(".xmp")
        if not xmp.exists():
            xmp = raw_path.with_suffix(".XMP")
        if not xmp.exists():
            logger.info("Skipping %s: no matching XMP sidecar found", raw_path)
            continue
        pairs.append((raw_path, xmp))
    return pairs


def _histogram_to_bytes(hist: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, hist)
    return buf.getvalue()


def _bytes_to_histogram(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def _derive_shoot_id(capture_datetime: datetime | None, camera_body: str | None) -> str:
    """Assign a shoot ID by bucketing capture time into 12-hour windows."""
    if capture_datetime is None:
        return f"unknown_{camera_body or 'unknown'}"
    epoch = datetime(2000, 1, 1)
    hours_since_epoch = (capture_datetime - epoch).total_seconds() / 3600
    bucket = int(hours_since_epoch // 12)
    body = (camera_body or "unknown").replace(" ", "_")
    return f"{bucket}_{body}"


def _process_pair(args: tuple[Path, Path | None, str, Path]) -> dict | None:
    """Worker function: extract one RAW+XMP pair and return a row dict."""
    raw_path, xmp_path, profile_name, thumbnail_dir = args
    try:
        data = extract_all(raw_path, xmp_path=xmp_path)
    except Exception as e:
        logger.error("Failed to process %s: %s", raw_path, e)
        return None

    file_id = _file_id(raw_path)

    # Save thumbnail
    thumb: Image.Image = data["preview"]
    thumb_path = thumbnail_dir / f"{file_id}.jpg"
    if not thumb_path.exists():
        thumb.save(thumb_path, format="JPEG", quality=_THUMB_QUALITY)

    # Capture datetime — may be string or datetime object
    cap_dt = data.get("capture_datetime")
    if isinstance(cap_dt, str):
        try:
            cap_dt = datetime.fromisoformat(cap_dt)
        except ValueError:
            cap_dt = None

    shoot_id = _derive_shoot_id(cap_dt, data.get("camera_body"))

    as_shot_temperature = None
    as_shot_tint = None
    if data.get("as_shot_wb") is not None:
        as_shot_temperature, as_shot_tint = data["as_shot_wb"]

    row: dict = {
        "id": file_id,
        "profile": profile_name,
        "raw_path": str(raw_path),
        "xmp_path": str(xmp_path) if xmp_path else None,
        "thumbnail_path": str(thumb_path),
        "shoot_id": shoot_id,
        # Metadata
        "iso": data.get("iso"),
        "shutter_speed": data.get("shutter_speed"),
        "aperture": data.get("aperture"),
        "focal_length": data.get("focal_length"),
        "lens_model": data.get("lens_model"),
        "camera_body": data.get("camera_body"),
        # v1.1.0 separate make/model alongside the legacy combined camera_body.
        "make": data.get("make"),
        "model": data.get("model"),
        "capture_datetime": cap_dt.isoformat() if cap_dt else None,
        "exposure_compensation": data.get("exposure_compensation"),
        "white_balance_preset": data.get("white_balance_preset"),
        "camera_profile": data.get("camera_profile"),
        "width": data.get("width"),
        "height": data.get("height"),
        "as_shot_temperature": as_shot_temperature,
        "as_shot_tint": as_shot_tint,
        # Histogram as bytes blob
        "histogram": _histogram_to_bytes(data["histogram"]),
    }

    scene_stats = data.get("scene_stats") or {}
    for field in SCENE_STAT_FIELDS:
        row[field] = scene_stats.get(field)

    # Slider values — 119 float columns
    sliders: dict = data.get("sliders") or {}
    for field in SLIDER_FIELDS:
        row[field] = sliders.get(field)

    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    input_dir: Path,
    output_path: Path,
    profile_name: str,
    thumbnail_dir: Path,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Build a Parquet training dataset from a folder of RAW+XMP pairs.

    Walks input_dir recursively, extracts previews and metadata for each pair,
    saves thumbnails to thumbnail_dir, and writes the result to output_path.
    Returns the resulting DataFrame.
    """
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = _find_pairs(input_dir)
    if not pairs:
        raise ValueError(
            f"No RAW files with matching XMP sidecars found in {input_dir}"
        )

    logger.info("Found %d supervised RAW/XMP pairs in %s", len(pairs), input_dir)

    args = [(raw, xmp, profile_name, thumbnail_dir) for raw, xmp in pairs]

    rows: list[dict] = []
    failures = 0

    if max_workers == 1:
        iterator = (_process_pair(a) for a in args)
    else:
        pool = multiprocessing.Pool(processes=max_workers)
        iterator = pool.imap(_process_pair, args)

    try:
        for result in tqdm(iterator, total=len(args), desc="Building dataset", unit="photo"):
            if result is None:
                failures += 1
            else:
                rows.append(result)
    finally:
        if max_workers != 1:
            pool.close()
            pool.join()

    if failures:
        logger.warning("%d/%d files failed to process.", failures, len(pairs))

    if not rows:
        raise RuntimeError("All files failed to process — dataset is empty.")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return df


def load_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load a dataset from Parquet."""
    return pd.read_parquet(parquet_path)


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    group_col: str = "shoot_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by shoot group (no photo-level leakage between splits).

    shoot_id is derived from capture time + camera body: photos within ~12 hours
    from the same body belong to the same shoot, so they always land in the same
    split.

    Original GroupShuffleSplit-based behaviour preserved when stratify_on=None.
    Default behaviour balances by shoot across Temperature correction,
    Exposure2012, and Tint correction; see stratified_group_split() for details.
    """
    return stratified_group_split(
        df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_col=group_col,
        stratify_on=("Temperature", "Exposure2012", "Tint"),
        n_strata=5,
        random_state=42,
    )


def stratified_group_split(
    df: pd.DataFrame,
    val_ratio: float = 0.107,
    test_ratio: float = 0.139,
    group_col: str = "shoot_id",
    stratify_on: str | Sequence[str] | None = "Temperature",
    asshot_col: str = "as_shot_temperature",
    n_strata: int = 5,
    random_state: int = 42,
    test_priority_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a photo dataset by group (shoot) with stratification on group-level
    editing characteristics.

    Why this exists: the original GroupShuffleSplit allocates groups uniformly at
    random. With small shoot counts, val/test can drift badly on the exact
    sliders we care about most. For example, a split that balances Temperature
    correction can still leave Exposure2012 much brighter in val/test than train.
    This function keeps shoot-level isolation while balancing split distributions
    across one or more edit characteristics.

    Algorithm:
      1. Per shoot, compute photo-weighted means for each stratification field.
         Temperature and Tint are represented as corrections from AsShot when
         their AsShot columns are available. Exposure is represented directly.
      2. Try deterministic random group assignments that hit the target
         photo-count ratios.
      3. Select the assignment with the lowest objective: size-ratio error plus
         train/val/test mean drift from global means across all stratification
         fields.

    Returns (train_df, val_df, test_df). Photo order within each split matches
    df's row order (no shuffle of photos themselves).

    Backward compatibility: setting stratify_on=None falls back to the original
    GroupShuffleSplit behaviour.
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in DataFrame columns")

    if stratify_on is None:
        # Legacy path — preserves the pre-stratified behaviour for callers that
        # explicitly opt out of stratification.
        groups = df[group_col].values
        s1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        train_val_idx, test_idx = next(s1.split(df, groups=groups))
        df_tv = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        s2 = GroupShuffleSplit(
            n_splits=1, test_size=val_ratio / (1 - test_ratio), random_state=random_state
        )
        train_idx, val_idx = next(s2.split(df_tv, groups=df_tv[group_col].values))
        df_train = df_tv.iloc[train_idx].reset_index(drop=True)
        df_val = df_tv.iloc[val_idx].reset_index(drop=True)
        logger.info(
            "Split (unstratified): train=%d, val=%d, test=%d", len(df_train), len(df_val), len(df_test)
        )
        return df_train, df_val, df_test

    # ── Balanced stratified path ──
    stratify_fields = [stratify_on] if isinstance(stratify_on, str) else list(stratify_on)
    missing = [field for field in stratify_fields if field not in df.columns]
    if missing:
        raise ValueError(f"stratify fields missing from DataFrame columns: {missing}")
    if "Temperature" in stratify_fields and asshot_col not in df.columns:
        raise ValueError(
            f"stratify_on includes 'Temperature' but asshot_col='{asshot_col}' "
            "is not in DataFrame columns"
        )

    import random as _random

    group_sizes = df.groupby(group_col).size()
    groups = group_sizes.index.tolist()
    n_total = len(df)
    target_counts = {
        "test": int(round(n_total * test_ratio)),
        "val": int(round(n_total * val_ratio)),
    }

    work = pd.DataFrame({group_col: df[group_col].values})
    feature_names: list[str] = []
    for field in stratify_fields:
        values = pd.to_numeric(df[field], errors="coerce")
        if field == "Temperature":
            asshot = pd.to_numeric(df[asshot_col], errors="coerce")
            work[field] = values - asshot
            feature_names.append(field)
        elif field == "Tint" and "as_shot_tint" in df.columns:
            ashot_tint = pd.to_numeric(df["as_shot_tint"], errors="coerce")
            work[field] = values - ashot_tint
            feature_names.append(field)
        else:
            work[field] = values
            feature_names.append(field)

    group_features = work.groupby(group_col)[feature_names].mean().fillna(0.0)
    global_means = pd.Series(index=feature_names, dtype=float)
    global_stds = pd.Series(index=feature_names, dtype=float)
    for field in feature_names:
        expanded = pd.to_numeric(work[field], errors="coerce").fillna(0.0)
        global_means[field] = float(expanded.mean())
        std = float(expanded.std(ddof=0))
        global_stds[field] = std if std > 1e-6 else 1.0

    group_to_pos = {sid: i for i, sid in enumerate(groups)}
    size_arr = group_sizes.reindex(groups).to_numpy(dtype=float)
    feature_arr = (
        (group_features.reindex(groups).fillna(0.0) - global_means) / global_stds
    ).to_numpy(dtype=float)
    tail_columns: list[np.ndarray] = []
    tail_quantile = 1.0 / max(n_strata, 2)
    for field in feature_names:
        group_col_values = group_features[field]
        low_cut = float(group_col_values.quantile(tail_quantile))
        high_cut = float(group_col_values.quantile(1.0 - tail_quantile))
        values = group_col_values.reindex(groups).to_numpy()
        tail_columns.append((values <= low_cut).astype(float))
        tail_columns.append((values >= high_cut).astype(float))
        min_value = float(group_col_values.min())
        max_value = float(group_col_values.max())
        tail_columns.append(np.isclose(values, min_value).astype(float))
        tail_columns.append(np.isclose(values, max_value).astype(float))
    tail_arr = np.stack(tail_columns, axis=1) if tail_columns else np.zeros((len(groups), 0))
    global_tail = (
        (tail_arr * size_arr[:, None]).sum(axis=0) / max(float(size_arr.sum()), 1.0)
        if tail_arr.size
        else np.zeros((0,), dtype=float)
    )

    def _photos_in(shoots: list) -> int:
        return int(sum(size_arr[group_to_pos[sid]] for sid in shoots))

    def _make_assignment(rng: _random.Random) -> tuple[list, list, list]:
        shuffled = groups[:]
        rng.shuffle(shuffled)
        test_shoots: list = []
        val_shoots: list = []
        train_shoots: list = []

        for sid in shuffled:
            test_count = _photos_in(test_shoots)
            val_count = _photos_in(val_shoots)
            if test_count < target_counts["test"]:
                test_shoots.append(sid)
            elif val_count < target_counts["val"]:
                val_shoots.append(sid)
            else:
                train_shoots.append(sid)

        if not train_shoots:
            train_shoots.append(test_shoots.pop())
        if not val_shoots:
            val_shoots.append(train_shoots.pop())
        if not test_shoots:
            test_shoots.append(train_shoots.pop())
        return train_shoots, val_shoots, test_shoots

    def _objective(train_shoots: list, val_shoots: list, test_shoots: list) -> float:
        split_map = {
            "train": train_shoots,
            "val": val_shoots,
            "test": test_shoots,
        }
        score = 0.0
        score += abs(_photos_in(test_shoots) - target_counts["test"]) / max(n_total, 1)
        score += abs(_photos_in(val_shoots) - target_counts["val"]) / max(n_total, 1)
        for shoots in split_map.values():
            if not shoots:
                score += 100.0
                continue
            idx = np.fromiter((group_to_pos[sid] for sid in shoots), dtype=int)
            weights = size_arr[idx]
            split_z = (feature_arr[idx] * weights[:, None]).sum(axis=0) / weights.sum()
            score += float(np.abs(split_z).sum())
            if tail_arr.size:
                split_tail = (tail_arr[idx] * weights[:, None]).sum(axis=0) / weights.sum()
                score += 2.0 * float(np.abs(split_tail - global_tail).sum())
        if tail_arr.size:
            for col_idx in range(tail_arr.shape[1]):
                total_tail_groups = int(tail_arr[:, col_idx].sum())
                if total_tail_groups < 2:
                    continue
                desired_splits = min(total_tail_groups, 3)
                occupied_splits = 0
                for shoots in split_map.values():
                    idx = np.fromiter((group_to_pos[sid] for sid in shoots), dtype=int)
                    occupied_splits += int(bool(tail_arr[idx, col_idx].sum()))
                score += 10.0 * float(desired_splits - occupied_splits)
        return score

    rng = _random.Random(random_state)
    best: tuple[float, list, list, list] | None = None
    n_candidates = max(500, min(5_000, len(groups) * 100))
    for _ in range(n_candidates):
        train_shoots, val_shoots, test_shoots = _make_assignment(rng)
        score = _objective(train_shoots, val_shoots, test_shoots)
        if best is None or score < best[0]:
            best = (score, train_shoots, val_shoots, test_shoots)

    assert best is not None
    _, train_shoots, val_shoots, test_shoots = best
    train_set, val_set, test_set = set(train_shoots), set(val_shoots), set(test_shoots)
    df_train = df[df[group_col].isin(train_set)].reset_index(drop=True)
    df_val = df[df[group_col].isin(val_set)].reset_index(drop=True)
    df_test = df[df[group_col].isin(test_set)].reset_index(drop=True)
    logger.info(
        "Balanced group split (fields=%s): train=%d photos / %d shoots, "
        "val=%d / %d, test=%d / %d",
        feature_names,
        len(df_train), len(train_set),
        len(df_val), len(val_set),
        len(df_test), len(test_set),
    )
    return df_train, df_val, df_test

def save_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save train/val/test splits as separate Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    logger.info("Saved splits to %s", output_dir)
