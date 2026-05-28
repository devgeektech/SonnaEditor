"""CLI entry point for the Mode B style survey (Step 1 of Mode B rebuild).

Usage:
  uv run python scripts/run_style_survey.py --output <path.json>
  uv run python scripts/run_style_survey.py --output <path.json> \\
      --non-interactive --answers exposure=0,temperature=1,tint=0,\\
                                   contrast=2,saturation=-1,shadows=1

See src/sonna_editor/mode_b/survey.py for the schema and context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sonna_editor.mode_b.survey import (
    build_survey_payload,
    format_summary,
    parse_answers_string,
    run_interactive,
    write_survey,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mode B style survey — captures editing preferences as slider offsets."
        ),
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Where to write the survey JSON.",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip interactive prompts. Requires --answers.",
    )
    parser.add_argument(
        "--answers", default=None,
        help=(
            "Comma-separated key=value pairs. All 6 keys required: "
            "exposure, temperature, tint, contrast, saturation, shadows. "
            "Values: -2..+2."
        ),
    )

    args = parser.parse_args(argv)

    if args.non_interactive and not args.answers:
        parser.error("--non-interactive requires --answers")
    if args.answers and not args.non_interactive:
        parser.error("--answers requires --non-interactive (explicit mode)")

    if not args.output.parent.exists():
        parser.error(
            f"Output parent directory does not exist: {args.output.parent}"
        )

    if args.non_interactive:
        try:
            answers = parse_answers_string(args.answers)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        answers = run_interactive()

    payload = build_survey_payload(answers)

    if not args.non_interactive:
        print()
        print(format_summary(payload))
        print()
        confirm = input(f"Save to {args.output}? [Y/n]: ").strip().lower()
        if confirm and confirm not in ("y", "yes"):
            print("Aborted; no file written.")
            return 1

    if args.output.exists():
        print(f"notice: overwriting existing file {args.output}")

    write_survey(payload, args.output)
    print(f"Wrote survey to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
