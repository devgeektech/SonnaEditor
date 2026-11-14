#!/usr/bin/env bash
# Double-clickable macOS launcher for Sonna Editor (Saha)
set -e

# Change to the directory where this script lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Ensure common macOS paths (Homebrew Apple Silicon & Intel, local bin) are on PATH
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$DIR/.saha/node/bin:$PATH"

echo "================================================="
echo " Starting Sonna Editor (Saha) for macOS"
echo "================================================="

# 1. Clean up cross-platform virtual environments (e.g. unzipped from Windows archive)
if [ -d ".venv" ]; then
    if [ ! -f ".venv/bin/python" ]; then
        echo "[saha] Detected non-macOS virtual environment (from Windows zip). Resetting .venv for macOS..."
        rm -rf .venv
    fi
fi

# 2. Clean up cross-platform node_modules (e.g. unzipped from Windows archive with .exe binaries)
if [ -d "saha-app/node_modules" ]; then
    if [ ! -d "saha-app/node_modules/electron/dist/Electron.app" ]; then
        echo "[saha] Detected non-macOS frontend binaries (from Windows zip). Resetting node_modules for macOS..."
        rm -rf saha-app/node_modules
    fi
fi

# 3. Ensure uv is installed (auto-installs if missing)
if ! command -v uv >/dev/null 2>&1; then
    echo "[saha] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Check Node.js (auto-installs portable Node.js LTS if missing)
if ! command -v node >/dev/null 2>&1; then
    echo "[saha] Node.js not detected on system PATH. Bootstrapping portable Node.js LTS..."
    ARCH="$(uname -m)"
    if [ "$ARCH" = "arm64" ]; then
        NODE_DIST="node-v22.11.0-darwin-arm64"
    else
        NODE_DIST="node-v22.11.0-darwin-x64"
    fi
    mkdir -p "$DIR/.saha"
    NODE_TAR="$DIR/.saha/${NODE_DIST}.tar.gz"
    echo "[saha] Downloading ${NODE_DIST}..."
    curl -LsSf "https://nodejs.org/dist/v22.11.0/${NODE_DIST}.tar.gz" -o "$NODE_TAR"
    echo "[saha] Unpacking Node.js..."
    tar -xzf "$NODE_TAR" -C "$DIR/.saha"
    rm -f "$NODE_TAR"
    rm -rf "$DIR/.saha/node"
    mv "$DIR/.saha/${NODE_DIST}" "$DIR/.saha/node"
    export PATH="$DIR/.saha/node/bin:$PATH"
    echo "[saha] Node.js $(node -v) successfully configured."
fi

# 5. Pull Git LFS models if they are still pointer files
if command -v git-lfs >/dev/null 2>&1; then
    git lfs pull 2>/dev/null || true
fi

# 6. Launch Saha (syncs python environment and boots Electron)
echo "[saha] Starting desktop application..."
uv run python scripts/run_app.py "$@"
