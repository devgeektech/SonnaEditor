#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH. Install uv, then run this command again." >&2
  echo "See RUN.md for setup instructions." >&2
  exit 1
fi

uv run python scripts/run_app.py "$@"
