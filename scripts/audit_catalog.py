#!/usr/bin/env python
"""CLI script to audit a built dataset and produce a quality report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a Sonna Editor training dataset and produce a quality report."
    )
    parser.add_argument(
        "--parquet-path", required=True, type=Path,
        help="Path to the dataset.parquet file produced by build_dataset.py.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write audit_report.md and plot PNGs.",
    )
    args = parser.parse_args()

    from sonna_editor.data.audit import audit_dataset

    summary = audit_dataset(args.parquet_path, args.output_dir)

    status = summary["status"]
    status_symbol = {"GO": "✅", "WARN": "⚠️", "STOP": "🛑"}.get(status, status)

    print(f"\n{status_symbol}  Status: {status}")
    print(f"   Photos:            {summary['n_photos']:,}")
    print(f"   Shoots:            {summary['n_shoots']:,}")
    print(f"   Likely unedited:   {summary['n_unedited']} ({100*summary['unedited_ratio']:.1f}%)")
    print(f"   Outlier sliders:   {summary['n_outlier_sliders']}")
    print(f"   Training estimate: {summary['training_minutes_estimate']:.0f} min (100 epochs, M1 Pro)")
    print(f"\n   Report:  {summary['report_path']}")

    if status == "STOP":
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
