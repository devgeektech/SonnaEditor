"""CLI wrapper for the Lite preset-to-checkpoint converter.

Step 2 of the Mode B rebuild track (HANDOVER Part 6 item 17).

Usage:

    uv run python scripts/build_mode_b_checkpoint.py \\
        --preset path/to/preset.xmp \\
        --survey path/to/survey.json \\
        --profile-name "Wedding Lite" \\
        [--profile-id custom-slug]

By default the base checkpoint is read from the configured foundation repo. Use
--base-ckpt only for deliberate experiments. The base checkpoint is loaded
read-only; the output is written to a new path. If --output is omitted, a
frontend-visible checkpoint is published as v1_learning/model-v0.N.0.ckpt. See
src/sonna_editor/mode_b/checkpoint_builder.py for the underlying logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sonna_editor import config
from sonna_editor.foundation import resolve_foundation_checkpoint
from sonna_editor.mode_b.checkpoint_builder import build_mode_b_checkpoint


def _next_mode_b_output(publish_dir: Path) -> Path:
    """Return the next frontend-visible Lite checkpoint path."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    seq = 1
    while True:
        candidate = publish_dir / f"model-v0.{seq}.0.ckpt"
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
        seq += 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_mode_b_checkpoint",
        description=(
            "Build an initial Lite SonnaEditor checkpoint from the configured "
            "foundation checkpoint, a Lightroom preset .xmp, and a style-survey JSON."
        ),
    )
    parser.add_argument("--preset", type=Path, required=True,
                        help="Lightroom preset .xmp")
    parser.add_argument("--survey", type=Path, required=True,
                        help="Style-survey JSON from Step 1")
    parser.add_argument("--base-ckpt", type=Path, default=None,
                        help=(
                            "Optional base SonnaEditor checkpoint. Defaults to "
                            "the configured foundation checkpoint."
                        ))
    parser.add_argument("--output", type=Path, default=None,
                        help=(
                            "Path for the new Lite checkpoint. If omitted, "
                            "publishes to v1_learning/model-v0.N.0.ckpt."
                        ))
    parser.add_argument("--publish-dir", type=Path, default=config.CHECKPOINTS_DIR,
                        help=(
                            "Directory used when --output is omitted "
                            f"(default: {config.CHECKPOINTS_DIR})."
                        ))
    parser.add_argument("--profile-name", type=str, required=True,
                        help='Display name, e.g. "Wedding Lite"')
    parser.add_argument("--profile-id", type=str, default=None,
                        help="Optional profile ID slug; auto-generated if omitted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output or _next_mode_b_output(args.publish_dir)
    if output_path.exists() or output_path.with_suffix(".json").exists():
        print(
            f"error: output already exists, refusing to overwrite: {output_path}",
            file=sys.stderr,
        )
        return 1

    try:
        base_ckpt = args.base_ckpt or resolve_foundation_checkpoint()
        sidecar = build_mode_b_checkpoint(
            preset_path=args.preset,
            survey_path=args.survey,
            base_ckpt_path=base_ckpt,
            output_ckpt_path=output_path,
            profile_name=args.profile_name,
            profile_id=args.profile_id,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Lite checkpoint written:   {output_path}")
    print(f"Sidecar JSON:              {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
