#!/usr/bin/env python3
"""
Migrate v1 (135-label) parquet splits to v2 (147-label) for the locked-wide retrain.

Adds 12 new label columns (idx 135-146) per the v2 SLIDER_FIELDS expansion.
Primary path: re-extract each field from the source XMP (sidecar or embedded
in DNG) via sonna_editor.data.xmp.read_xmp. Fallback: config.SLIDER_DEFAULTS
when the XMP is missing/unreadable or the specific field is absent.

PREREQUISITE: the external drive holding the source RAWs must be mounted.
Otherwise every row will hit the SLIDER_DEFAULTS fallback and the migrated
parquet will have no real v2 signal — only defaults.

Usage:
  uv run python scripts/migrate_labels_to_v2.py
  uv run python scripts/migrate_labels_to_v2.py --dry-run --limit 100
  uv run python scripts/migrate_labels_to_v2.py \\
      --input-dir data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified \\
      --output-dir data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified.lockedwide147 \\
      --splits train,val,test \\
      --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sonna_editor import config
from sonna_editor.data.xmp import read_xmp

_logger = logging.getLogger("migrate_labels_to_v2")

V2_EXTENSION_FIELDS: list[str] = list(config.SLIDER_FIELDS[135:])  # 12 fields
DEFAULT_INPUT_DIR = Path("data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified")
DEFAULT_OUTPUT_DIR = Path("data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified.lockedwide147")
DEFAULT_SPLITS = ("train", "val", "test", "train_3k_stratified")


def _xmp_target_for_raw(raw_path: Path) -> Path | None:
    """Return the path read_xmp() should open for this RAW.

    Order of preference:
      1. Sidecar .xmp next to the RAW (any RAW format).
      2. The RAW itself (read_xmp parses embedded XMP from DNG/CR3/NEF/etc.
         via _extract_xmp_bytes_from_binary).
    Returns None if neither exists / drive not mounted.
    """
    sidecar = raw_path.with_suffix(".xmp")
    if sidecar.exists():
        return sidecar
    if raw_path.exists():
        return raw_path  # embedded XMP fallback inside the RAW file
    return None


def _extract_v2_fields_for_row(raw_path_str: str) -> tuple[dict[str, float], list[str], str | None]:
    """Extract the 12 v2 fields for one row.

    Returns:
        values: dict[field -> float], always 12 entries (defaults fill gaps).
        fallback_fields: list of v2 fields where SLIDER_DEFAULTS was used.
        error: short string if XMP read failed altogether; None otherwise.
    """
    raw_path = Path(raw_path_str)
    target = _xmp_target_for_raw(raw_path)

    if target is None:
        return (
            {f: config.SLIDER_DEFAULTS[f] for f in V2_EXTENSION_FIELDS},
            list(V2_EXTENSION_FIELDS),
            "no XMP source (RAW + sidecar both missing — drive unmounted?)",
        )

    try:
        xmp_data = read_xmp(target)
    except Exception as exc:
        return (
            {f: config.SLIDER_DEFAULTS[f] for f in V2_EXTENSION_FIELDS},
            list(V2_EXTENSION_FIELDS),
            f"read_xmp failed: {type(exc).__name__}: {exc}",
        )

    values: dict[str, float] = {}
    fallback: list[str] = []
    for field in V2_EXTENSION_FIELDS:
        v = xmp_data.get(field)
        if v is None:
            values[field] = config.SLIDER_DEFAULTS[field]
            fallback.append(field)
        else:
            try:
                values[field] = float(v)
            except (TypeError, ValueError):
                values[field] = config.SLIDER_DEFAULTS[field]
                fallback.append(field)
    return values, fallback, None


def migrate_split(
    input_path: Path,
    output_path: Path,
    workers: int,
    limit: int | None,
    dry_run: bool,
) -> dict:
    """Migrate one parquet split. Returns stats dict."""
    df = pd.read_parquet(input_path)

    # Idempotency: skip if all 12 v2 fields are already present
    if all(f in df.columns for f in V2_EXTENSION_FIELDS):
        _logger.info("%s already has all 12 v2 columns; skipping", input_path)
        return {"input": str(input_path), "skipped": True, "rows": len(df)}

    if "raw_path" not in df.columns:
        raise ValueError(f"{input_path}: missing 'raw_path' column, cannot re-extract")

    work_df = df.head(limit) if limit is not None else df
    raw_paths = work_df["raw_path"].tolist()

    fallback_counts = {f: 0 for f in V2_EXTENSION_FIELDS}
    error_count = 0
    new_columns: dict[str, list[float]] = {f: [] for f in V2_EXTENSION_FIELDS}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(
            pool.map(_extract_v2_fields_for_row, raw_paths),
            total=len(raw_paths),
            desc=input_path.name,
            unit="row",
        ))

    for values, fallback, error in results:
        for f in V2_EXTENSION_FIELDS:
            new_columns[f].append(values[f])
        for f in fallback:
            fallback_counts[f] += 1
        if error is not None:
            error_count += 1

    # If --limit was used, only the first N rows have extracted v2 values;
    # the rest of the dataframe is dropped to avoid writing partially-migrated data.
    if limit is not None:
        df = work_df.copy()
    for f, vals in new_columns.items():
        df[f] = vals

    if dry_run:
        _logger.info(
            "DRY-RUN: would write %d rows x %d cols to %s",
            len(df), len(df.columns), output_path,
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(df),
        "error_rows": error_count,
        "fallback_counts": fallback_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Input parquet dir (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output parquet dir (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--splits", default=",".join(DEFAULT_SPLITS),
        help=f"Comma-separated split names (default: {','.join(DEFAULT_SPLITS)})",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Thread pool size for XMP reads (default: 8)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only migrate the first N rows of each split (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run extraction and report stats but don't write output parquets",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input_dir.exists():
        _logger.error("Input dir does not exist: %s", args.input_dir)
        return 1
    if args.output_dir.resolve() == args.input_dir.resolve():
        _logger.error("Output dir must differ from input dir (would overwrite!)")
        return 1

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    summary = []
    for split in split_names:
        input_path = args.input_dir / f"{split}.parquet"
        output_path = args.output_dir / f"{split}.parquet"
        if not input_path.exists():
            _logger.warning("Split missing: %s -- skipping", input_path)
            continue
        result = migrate_split(input_path, output_path, args.workers, args.limit, args.dry_run)
        summary.append(result)

    print("\n=== Migration summary ===")
    if args.dry_run:
        print("  (DRY-RUN: no parquet files were written)")
    for r in summary:
        if r.get("skipped"):
            print(f"  {r['input']}: SKIPPED ({r['rows']} rows, already v2)")
            continue
        rows = r["rows"]
        err = r["error_rows"]
        print(f"  {r['input']} -> {r['output']}")
        print(f"      rows={rows}, xmp_read_errors={err} ({100*err/rows:.1f}%)")
        any_fallback = False
        for f, n in r["fallback_counts"].items():
            if n > 0:
                pct = 100 * n / rows
                print(f"      fallback {f}: {n} rows ({pct:.1f}%)")
                any_fallback = True
        if not any_fallback:
            print("      no per-field fallbacks (every row extracted cleanly)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
