#!/usr/bin/env python
"""Apply a Lightroom preset to a shoot — XMPs written next to source RAW files by default."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Lightroom preset with content-aware adjustments to a folder of RAWs. "
            "XMP sidecars are written next to their source RAW files by default."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Folder containing RAW files.")
    parser.add_argument("--preset", required=True, type=Path,
                        help="Preset file (.xmp, .lrtemplate, or .xmpsettings).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override: write XMPs to this directory instead of next to each RAW.")
    parser.add_argument("--auto-exposure", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-adjust exposure per photo (default: on).")
    parser.add_argument("--auto-white-balance", action=argparse.BooleanOptionalAction, default=False,
                        help="Auto-adjust white balance per photo (default: off).")
    parser.add_argument("--auto-shadow-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto shadow recovery (default: on).")
    parser.add_argument("--auto-highlight-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto highlight recovery (default: on).")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers (default: 4).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse photos but don't write XMPs.")
    args = parser.parse_args()

    from sonna_editor.preset.pipeline import process_shoot

    options = {
        "auto_exposure": args.auto_exposure,
        "auto_white_balance": args.auto_white_balance,
        "auto_shadow_recovery": args.auto_shadow_recovery,
        "auto_highlight_recovery": args.auto_highlight_recovery,
    }

    if args.dry_run:
        print("Dry run — no XMPs will be written.\n")

    summary = process_shoot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset_path=args.preset,
        options=options,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )

    print(f"\nDone.")
    print(f"  Processed: {summary['processed']}")
    print(f"  Failed:    {summary['failed']}")
    if summary["failures"]:
        print("\nFailures:")
        for f in summary["failures"]:
            print(f"  {f['path']}: {f['error']}")
    if not args.dry_run and summary["output_paths"]:
        if args.output_dir:
            print(f"\nXMPs written to: {args.output_dir}/")
        else:
            print(f"\nXMPs written next to source RAWs in: {args.input_dir}/")
        for p in summary["output_paths"][:5]:
            print(f"  {Path(p).name}")
        if len(summary["output_paths"]) > 5:
            print(f"  ...and {len(summary['output_paths']) - 5} more")

    if summary["failed"] > 0 and summary["processed"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
