#!/usr/bin/env bash
# Packaging script for Saha (macOS / Linux)
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$DIR/.saha/node/bin:$PATH"

echo "================================================="
echo " Packaging Saha (Clean Production Distribution)"
echo "================================================="

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/package_app.py "$@"
elif [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/package_app.py "$@"
else
    python3 scripts/package_app.py "$@"
fi
