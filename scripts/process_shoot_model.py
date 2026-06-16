#!/usr/bin/env python
"""Apply trained SonnaEditor model to a shoot — XMPs written next to source RAW files by default."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sonna_editor import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _latest_published_model() -> Path | None:
    candidates = sorted(config.CHECKPOINTS_DIR.glob("model-v*.ckpt"))
    return candidates[-1] if candidates else None


def main() -> None:
    config.ensure_runtime_directories()
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained SonnaEditor model on a folder of RAW files. "
            "XMP sidecars are written next to their source RAW files by default."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Folder containing RAW files.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to trained checkpoint. Defaults to the newest model-v*.ckpt "
            f"under {config.CHECKPOINTS_DIR}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override: write XMPs to this directory instead of next to each RAW.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Images per inference batch (default: 32).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads for preview extraction (default: 4).")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Enable MC dropout uncertainty estimation (slower; flags low-confidence shots).")
    parser.add_argument("--uncertainty-samples", type=int, default=10,
                        help="MC dropout forward passes per image (default: 10). Requires --uncertainty.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: mps, cpu (default: auto-detect).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run inference but don't write XMPs.")
    parser.add_argument("--no-save-predictions", dest="save_predictions",
                        action="store_false", default=True,
                        help="Skip writing sonna_predictions.json (disables continuous learning capture).")
    parser.add_argument("--auto-straighten", action="store_true",
                        help="Estimate and write Lightroom crop-angle metadata when confident.")
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = _latest_published_model()

    if not args.input_dir.is_dir():
        print(f"Error: --input-dir '{args.input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if args.model_path is None:
        print(
            "Error: no published model checkpoint was found. "
            f"Train or publish a profile into '{config.CHECKPOINTS_DIR}' "
            "or pass --model-path explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.model_path.exists():
        print(f"Error: --model-path '{args.model_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    from sonna_editor.inference.pipeline import process_shoot_with_model

    if args.dry_run:
        print("Dry run — no XMPs will be written.\n")

    print(f"Model:     {args.model_path}")
    print(f"Input:     {args.input_dir}")
    if args.output_dir:
        print(f"Output:    {args.output_dir}")
    if args.uncertainty:
        print(f"Uncertainty: MC dropout, {args.uncertainty_samples} samples")
    if args.auto_straighten:
        print("Auto straighten: enabled")
    print()

    summary = process_shoot_with_model(
        input_dir=args.input_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.workers,
        uncertainty=args.uncertainty,
        n_uncertainty_samples=args.uncertainty_samples,
        dry_run=args.dry_run,
        device=args.device,
        save_predictions=args.save_predictions,
        auto_straighten=args.auto_straighten,
    )

    print("Done.")
    print(f"  Processed: {summary['processed']}")
    print(f"  Failed:    {summary['failed']}")

    if summary["failures"]:
        print("\nFailures:")
        for f in summary["failures"]:
            print(f"  {f['path']}: {f['error']}")

    if summary.get("low_confidence"):
        print(f"\nLow-confidence shots ({len(summary['low_confidence'])}):")
        for item in summary["low_confidence"]:
            print(f"  {Path(item['path']).name}  (mean_std={item['mean_std']:.2f})")

    if not args.dry_run and summary["output_paths"]:
        dest = args.output_dir or args.input_dir
        print(f"\nXMPs written to: {dest}/")
        for p in summary["output_paths"][:5]:
            print(f"  {Path(p).name}")
        if len(summary["output_paths"]) > 5:
            print(f"  ...and {len(summary['output_paths']) - 5} more")

    if summary.get("predictions_path"):
        print(f"\nPredictions saved to: {summary['predictions_path']}")
        print("  (Pass this to capture_user_edits after Lightroom tweaks)")

    if summary["failed"] > 0 and summary["processed"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
