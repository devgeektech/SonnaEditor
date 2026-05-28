from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from sonna_editor.config import SLIDER_FIELDS, SUPPORTED_RAW_EXTENSIONS
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
        # Histogram as bytes blob
        "histogram": _histogram_to_bytes(data["histogram"]),
    }

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
    Default behaviour (stratify_on="Temperature") uses stratified-by-shoot
    allocation; see stratified_group_split() for details.
    """
    return stratified_group_split(
        df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_col=group_col,
        stratify_on="Temperature",
        n_strata=5,
        random_state=42,
    )


def stratified_group_split(
    df: pd.DataFrame,
    val_ratio: float = 0.107,
    test_ratio: float = 0.139,
    group_col: str = "shoot_id",
    stratify_on: str | None = "Temperature",
    asshot_col: str = "as_shot_temperature",
    n_strata: int = 5,
    random_state: int = 42,
    test_priority_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a photo dataset by group (shoot) with stratification on group-level
    editing characteristics.

    Why this exists: the original GroupShuffleSplit allocates groups uniformly at
    random. With ~120 shoots, 13-shoot val/test sets, and a small number of
    cooling-heavy shoots that happen to be large, random allocation produced
    severe direction-distribution drift (val 81% cooling vs train 56%). This
    function stratifies shoots into n_strata quantile buckets by their mean
    edit-delta, then allocates within each stratum so all splits draw from
    similar editing-characteristic distributions.

    Algorithm:
      1. Per shoot, compute mean(stratify_on - asshot_col) across photos.
         Shoots without AsShot data get mean_delta=0 (neutral stratum).
      2. Quantile-bucket shoots into n_strata.
      3. Within each stratum: shuffle (seeded), then integer-allocate
         test/val/train counts. Test gets max(1, round(n*test_ratio));
         val gets max(1, round(n*val_ratio)); train absorbs the remainder.
      4. Balancing post-pass: prioritises test_ratio. If test photo-share is
         under target by more than tolerance, move shoots from train to test
         (preferring shoots whose size best closes the gap, drawn from strata
         where train is most over-represented). Same for val.

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

    # ── Stratified path ──
    if stratify_on not in df.columns or asshot_col not in df.columns:
        raise ValueError(
            f"stratify_on='{stratify_on}' and asshot_col='{asshot_col}' must both be in df.columns"
        )

    # Step 1: shoot-level stratification key
    final = pd.to_numeric(df[stratify_on], errors="coerce")
    asshot = pd.to_numeric(df[asshot_col], errors="coerce")
    work = pd.DataFrame({group_col: df[group_col].values, "_delta": final - asshot})
    shoot_means = work.groupby(group_col)["_delta"].mean()
    # Shoots with no AsShot data → neutral stratum (mean_delta=0 falls in middle bucket).
    shoot_means = shoot_means.fillna(0.0)

    # Step 2: quantile-bucket
    try:
        strata = pd.qcut(shoot_means, n_strata, labels=False, duplicates="drop")
    except ValueError as e:
        raise ValueError(
            f"qcut failed at n_strata={n_strata}; dataset may be too small or too uniform: {e}"
        )

    # Photo counts per shoot — for the balancing post-pass.
    shoot_sizes = df.groupby(group_col).size()

    # Step 3: integer allocate within each stratum, deterministic
    import random as _random
    rng = _random.Random(random_state)
    train_shoots: list = []
    val_shoots: list = []
    test_shoots: list = []
    shoot_to_stratum: dict = {}
    for s in sorted(strata.unique()):
        stratum = strata[strata == s].index.tolist()
        rng.shuffle(stratum)
        n = len(stratum)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val
        # Integer arithmetic safety: shrink test/val if they crowd train out.
        while n_train < 1:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
            n_train = n - n_test - n_val
        train_shoots += stratum[:n_train]
        val_shoots += stratum[n_train:n_train + n_val]
        test_shoots += stratum[n_train + n_val:]
        for sid in stratum:
            shoot_to_stratum[sid] = s

    # Step 4: balancing post-pass — prioritise test photo-ratio, then val.
    # Move shoots from train to the under-represented split, picking shoots
    # whose photo-count best closes the gap, drawn preferentially from strata
    # where train is currently most over-represented vs target.
    n_total = len(df)
    target_test_photos = int(round(n_total * test_ratio))
    target_val_photos = int(round(n_total * val_ratio))
    tol_photos = int(round(n_total * test_priority_tolerance))

    def _photos_in(shoots: list) -> int:
        return int(shoot_sizes.reindex(shoots).fillna(0).sum())

    def _move_to(target_shoots: list, target_photo_goal: int) -> None:
        """Move shoots from train_shoots to target_shoots until photo count
        reaches goal (within tol_photos), or no good candidate remains."""
        cur = _photos_in(target_shoots)
        # Per-stratum train-count → identifies strata where train is over-represented.
        max_iters = len(train_shoots)  # safety bound
        for _ in range(max_iters):
            if cur >= target_photo_goal - tol_photos:
                return
            gap = target_photo_goal - cur
            train_set = set(train_shoots)
            # Candidate shoots from train, sorted by photo size (ascending)
            cand = sorted(train_set, key=lambda sid: shoot_sizes.get(sid, 0))
            # Best candidate = closest to gap without overshooting beyond +50%
            best = None
            best_diff = float("inf")
            for sid in cand:
                sz = int(shoot_sizes.get(sid, 0))
                if sz == 0:
                    continue
                # Prefer candidates whose size doesn't overshoot gap by >50%.
                if sz <= int(gap * 1.5):
                    diff = abs(gap - sz)
                    if diff < best_diff:
                        best, best_diff = sid, diff
            if best is None:
                # No non-overshooting candidate — take the smallest train shoot.
                best = cand[0] if cand else None
            if best is None:
                return  # train is empty; nothing to move
            train_shoots.remove(best)
            target_shoots.append(best)
            cur += int(shoot_sizes.get(best, 0))

    _move_to(test_shoots, target_test_photos)
    _move_to(val_shoots, target_val_photos)

    # Final assembly
    train_set, val_set, test_set = set(train_shoots), set(val_shoots), set(test_shoots)
    df_train = df[df[group_col].isin(train_set)].reset_index(drop=True)
    df_val = df[df[group_col].isin(val_set)].reset_index(drop=True)
    df_test = df[df[group_col].isin(test_set)].reset_index(drop=True)
    logger.info(
        "Stratified split (n_strata=%d, stratify_on=%s): "
        "train=%d photos / %d shoots, val=%d / %d, test=%d / %d",
        n_strata, stratify_on,
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
