#!/usr/bin/env python
"""Build a training dataset from a Lightroom Classic catalog.

This path does not require exported XMP sidecars. Slider targets are read
directly from the catalog's develop-settings blobs. The catalog is opened
read-only and Lightroom Classic must be closed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Sonna Editor training dataset from a Lightroom .lrcat."
    )
    parser.add_argument(
        "--catalog-path", required=True, type=Path,
        help="Path to the Lightroom Classic .lrcat file.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory where dataset.parquet, thumbnails/, and optional splits will be written.",
    )
    parser.add_argument(
        "--profile-name", required=True,
        help="Name to tag all rows with, e.g. sonna_v2.",
    )
    parser.add_argument(
        "--limit", type=int, default=30_000,
        help="Maximum number of edited catalog photos to include, most recent first.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel thumbnail/metadata workers.",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Also produce train/val/test splits after building.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.107,
        help="Fraction of photos for validation after shoot-group splitting.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.139,
        help="Fraction of photos for test after shoot-group splitting.",
    )
    parser.add_argument(
        "--splits-dir-name", default="splits_v2_stratified",
        help="Name of the split directory under output-dir.",
    )
    args = parser.parse_args()

    from sonna_editor.data.catalog import CatalogError
    from sonna_editor.data.catalog_dataset import build_dataset_from_catalog
    from sonna_editor.data.dataset import save_split, split_dataset

    output_dir: Path = args.output_dir
    parquet_path = output_dir / "dataset.parquet"
    thumbnail_dir = output_dir / "thumbnails"

    print(f"Catalog:  {args.catalog_path}")
    print(f"Output:   {output_dir}")
    print(f"Profile:  {args.profile_name}")
    print(f"Limit:    {args.limit}")
    print(f"Workers:  {args.workers}")
    print()

    try:
        df, stats = build_dataset_from_catalog(
            catalog_path=args.catalog_path,
            output_path=parquet_path,
            profile_name=args.profile_name,
            thumbnail_dir=thumbnail_dir,
            limit=args.limit,
            max_workers=args.workers,
        )
    except CatalogError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        sys.exit(1)

    stats_path = output_dir / "catalog_build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nDataset: {len(df)} photos -> {parquet_path}")
    print(f"Stats:   {stats_path}")

    if args.split:
        train, val, test = split_dataset(
            df,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        splits_dir = output_dir / args.splits_dir_name
        save_split(train, val, test, splits_dir)
        print(f"Splits:  train={len(train)}, val={len(val)}, test={len(test)} -> {splits_dir}")


if __name__ == "__main__":
    main()
