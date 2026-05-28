"""CLI wrapper for the Mode B preset-to-checkpoint converter.

Step 2 of the Mode B rebuild track (HANDOVER Part 6 item 17).

Usage:

    uv run python scripts/build_mode_b_checkpoint.py \\
        --preset path/to/preset.xmp \\
        --survey path/to/survey.json \\
        --base-ckpt v1_learning/model-v1.2.3-prod256.ckpt \\
        --output path/to/mode_b.ckpt \\
        --profile-name "Mode B - Wedding Lite" \\
        [--profile-id custom-slug]

The base checkpoint is loaded read-only; the output is written to a new
path. See src/sonna_editor/mode_b/checkpoint_builder.py for the underlying
logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sonna_editor.mode_b.checkpoint_builder import build_mode_b_checkpoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_mode_b_checkpoint",
        description=(
            "Build an initial Mode B SonnaEditor checkpoint from a "
            "Lightroom preset .xmp and a Step 1 style-survey JSON."
        ),
    )
    parser.add_argument("--preset", type=Path, required=True,
                        help="Lightroom preset .xmp")
    parser.add_argument("--survey", type=Path, required=True,
                        help="Style-survey JSON from Step 1")
    parser.add_argument("--base-ckpt", type=Path, required=True,
                        help="Base SonnaEditor checkpoint (v1.2.3 recommended)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the new Mode B checkpoint")
    parser.add_argument("--profile-name", type=str, required=True,
                        help='Display name, e.g. "Mode B - Wedding Lite"')
    parser.add_argument("--profile-id", type=str, default=None,
                        help="Optional profile ID slug; auto-generated if omitted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        sidecar = build_mode_b_checkpoint(
            preset_path=args.preset,
            survey_path=args.survey,
            base_ckpt_path=args.base_ckpt,
            output_ckpt_path=args.output,
            profile_name=args.profile_name,
            profile_id=args.profile_id,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Mode B checkpoint written: {args.output}")
    print(f"Sidecar JSON:              {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
