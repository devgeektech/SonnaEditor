# macOS Client Testing & Distribution Guide

This document describes how to package, distribute, and run **Sonna Editor (Saha)** on macOS, specifically tailored for Darshil (Founder, Sonna Studios) working on Apple Silicon (M1 Pro / M2 / M3 / M4) and Intel MacBooks.

---

## 1. Executive Summary

Saha is designed as a local-first, cross-platform AI photo editing engine. On macOS:
- **PyTorch Acceleration:** Runs natively on Apple Silicon GPU using Metal Performance Shaders (`mps`) with fp32 (`32-true`) precision for rock-solid stability and zero numerical drift.
- **Unified Frontend/Backend:** Desktop UI is built in Electron + React; backend runs a local FastAPI engine on port 8765.
- **Zero-Setup Launcher:** `start_saha_mac.command` automatically configures the environment on Darshil's Mac, including `uv`, portable Node.js LTS, and model checkpoints.

---

## 2. Packaging the App for Client Testing

To create a clean, self-contained distribution zip ready to send to Darshil:

```bash
uv run python scripts/package_for_client.py
```

### What This Produces:
- Archive: `dist/saha-mac-client-bundle.zip` (~140 MB compressed).
- Includes:
  - Complete backend source (`src/sonna_editor/`) and FastAPI server.
  - Complete desktop frontend (`saha-app/`) with both Lite Profile and Personal AI Wizards.
  - Pre-trained foundation AI checkpoint (`SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-004-visual.ckpt`).
  - Lightroom Classic bridge plugin (`lightroom/SahaBridge.lrplugin`).
  - Double-clickable Mac launcher (`start_saha_mac.command`).
  - Quick-start instructions (`QUICK_START.txt`).
- Excludes: Git history (`.git`), virtual environments (`.venv`), temporary test caches, and node_modules, keeping the zip lightweight and fast to transfer via Google Drive, Dropbox, or AirDrop.

---

## 3. Client Quick Start (Darshil's Workflow on Mac)

Darshil does not need to manually configure Python versions or command lines.

### Step 1: Unzip
Extract `saha-mac-client-bundle.zip` anywhere on the Mac (e.g., in `~/Applications/` or `~/Desktop/`).

### Step 2: Launch
Double-click `start_saha_mac.command`.

> [!NOTE] macOS Gatekeeper First-Launch Prompt
> If macOS displays a warning ("developer cannot be verified"):
> 1. Right-click (or Control-click) `start_saha_mac.command` in Finder.
> 2. Click **Open**, then confirm **Open** in the dialog.
> This only needs to be done once.

### What the Launcher Does Automatically:
1. Detects system PATH (including Homebrew at `/opt/homebrew/bin` on Apple Silicon).
2. Auto-installs `uv` package manager if not present.
3. Auto-bootstraps portable Node.js LTS for `darwin-arm64` if Node is not on the Mac.
4. Verifies model checkpoint weights.
5. Launches the Saha desktop window and FastAPI engine.

---

## 4. Lightroom Classic Integration

To test the end-to-end editing bridge:

1. Open **Adobe Lightroom Classic**.
2. Navigate to **File** -> **Plug-in Manager**.
3. Click **Add** in the bottom-left corner.
4. Choose the `lightroom/SahaBridge.lrplugin` folder inside the extracted Saha directory.
5. Status will show a green dot: `Installed and running`.
6. Select any RAW photos in your catalog, right-click -> **Export** -> select **Saha: Apply Edits**.

---

## 5. Building a Standalone macOS Installer (.dmg)

If you prefer delivering a native macOS disk image installer (`.dmg`) rather than a zip bundle:

Inside `saha-app/` on a Mac:
```bash
npm run build:mac
```

This compiles Vite production assets and packages Electron into:
`saha-app/dist-app/Saha-0.1.0-arm64.dmg`

Darshil can open the DMG and drag **Saha** directly into the Mac **Applications** folder.

---

## 6. Verification Checklist

Before sending any build to the client, verify:
- [x] Pretrained checkpoint present: `SonnaEditorFoundation/checkpoints/foundation-sonna-raw-xmp-004-visual.ckpt` (~119 MB).
- [x] Checkpoint metadata path is relative (no hardcoded `/Users/...` machine paths).
- [x] Slider defaults unified in `src/sonna_editor/config.py`.
- [x] Personal AI profile creation wizard enabled in `saha-app/src/components/profile-view.jsx`.
- [x] Preview memory throttling enabled in `src/sonna_editor/inference/pipeline.py`.
- [x] `start_saha_mac.command` has Unix LF line endings and executable permissions.
