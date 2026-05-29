#!/usr/bin/env node
// Rasterise build/icon-source.svg → multi-size PNGs → iconutil → build/icon.icns.
// Also writes build/icon.png (1024) for Electron's dev-mode dock icon.
//
// macOS only — uses /usr/bin/iconutil. sharp handles the SVG → PNG step.

const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const sharp = require('sharp');

const BUILD_DIR = __dirname;
const SOURCE_SVG = path.join(BUILD_DIR, 'icon-source.svg');
const ICONSET_DIR = path.join(BUILD_DIR, 'icon.iconset');
const ICNS_OUT = path.join(BUILD_DIR, 'icon.icns');
const PNG_OUT = path.join(BUILD_DIR, 'icon.png');

// Apple's required .iconset entries — every nominal size has @2x variant
// at double the pixel count. iconutil rejects iconsets missing any of these.
const ICONSET_ENTRIES = [
  { name: 'icon_16x16.png',     size: 16 },
  { name: 'icon_16x16@2x.png',  size: 32 },
  { name: 'icon_32x32.png',     size: 32 },
  { name: 'icon_32x32@2x.png',  size: 64 },
  { name: 'icon_128x128.png',   size: 128 },
  { name: 'icon_128x128@2x.png', size: 256 },
  { name: 'icon_256x256.png',   size: 256 },
  { name: 'icon_256x256@2x.png', size: 512 },
  { name: 'icon_512x512.png',   size: 512 },
  { name: 'icon_512x512@2x.png', size: 1024 },
];

async function main() {
  if (!fs.existsSync(SOURCE_SVG)) {
    console.error(`Missing source SVG: ${SOURCE_SVG}`);
    process.exit(1);
  }
  if (process.platform !== 'darwin') {
    console.error('iconutil is macOS-only. Run this on macOS.');
    process.exit(1);
  }

  // Clean previous iconset so stale sizes can't sneak into the .icns.
  if (fs.existsSync(ICONSET_DIR)) {
    fs.rmSync(ICONSET_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(ICONSET_DIR);

  const svgBuf = fs.readFileSync(SOURCE_SVG);

  for (const { name, size } of ICONSET_ENTRIES) {
    const out = path.join(ICONSET_DIR, name);
    await sharp(svgBuf, { density: 384 })  // bump density so high-res renders are crisp
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toFile(out);
    console.log(`  ${name.padEnd(24)} ${size}×${size}`);
  }

  // Build .icns via the macOS native tool.
  if (fs.existsSync(ICNS_OUT)) fs.unlinkSync(ICNS_OUT);
  execFileSync('/usr/bin/iconutil', ['--convert', 'icns', ICONSET_DIR, '-o', ICNS_OUT]);
  console.log(`\nWrote ${path.relative(process.cwd(), ICNS_OUT)}`);

  // Standalone 1024 PNG for Electron's BrowserWindow icon (dev mode dock).
  await sharp(svgBuf, { density: 384 })
    .resize(1024, 1024, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toFile(PNG_OUT);
  console.log(`Wrote ${path.relative(process.cwd(), PNG_OUT)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
