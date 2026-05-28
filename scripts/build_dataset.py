#!/usr/bin/env python
"""CLI script to build a Parquet training dataset from RAW+XMP pairs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Sonna Editor training dataset from RAW+XMP pairs."
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing RAW files (and optional .xmp sidecars).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory where dataset.parquet, thumbnails/, and splits/ will be written.",
    )
    parser.add_argument(
        "--profile-name", required=True,
        help="Name to tag all rows with (e.g. 'sonna_v1').",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Also produce train/val/test splits after building.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of data for validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Fraction of data for test split (default: 0.1).",
    )
    args = parser.parse_args()

    # Late import so errors surface cleanly
    from sonna_editor.data.dataset import (
        build_dataset,
        save_split,
        split_dataset,
    )

    output_dir: Path = args.output_dir
    parquet_path = output_dir / "dataset.parquet"
    thumbnail_dir = output_dir / "thumbnails"

    print(f"Input:    {args.input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Profile:  {args.profile_name}")
    print(f"Workers:  {args.workers}")
    print()

    df = build_dataset(
        input_dir=args.input_dir,
        output_path=parquet_path,
        profile_name=args.profile_name,
        thumbnail_dir=thumbnail_dir,
        max_workers=args.workers,
    )

    print(f"\nDataset: {len(df)} photos → {parquet_path}")

    if args.split:
        train, val, test = split_dataset(
            df, val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )
        splits_dir = output_dir / "splits"
        save_split(train, val, test, splits_dir)
        print(f"Splits:  train={len(train)}, val={len(val)}, test={len(test)} → {splits_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
