#!/usr/bin/env bash
# Double-clickable macOS launcher for Sonna Editor (Saha)
set -e

# Change to the directory where this script lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================="
echo " Starting Sonna Editor (Saha) for macOS"
echo "================================================="

# 1. Ensure uv is installed (auto-installs if missing)
if ! command -v uv >/dev/null 2>&1; then
    echo "[saha] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Check Node.js
if ! command -v node >/dev/null 2>&1; then
    echo ""
    echo "[!] Node.js is required to run the desktop interface."
    echo "    Please download and install Node.js LTS from: https://nodejs.org/"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

# 3. Pull Git LFS models if they are still pointer files
if command -v git-lfs >/dev/null 2>&1; then
    git lfs pull 2>/dev/null || true
fi

# 4. Launch Saha (syncs python environment and boots Electron)
echo "[saha] Starting desktop application..."
uv run python scripts/run_app.py "$@"
