// Electron main process — spawns the FastAPI backend, gates window creation
// on /api/health, cleanly terminates the subprocess on quit.

const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const http = require('node:http');

const isDev = !app.isPackaged;
const PORT = 8765;
const HEALTH_URL = `http://127.0.0.1:${PORT}/api/health`;

// Repo-root resolution:
//   dev:        electron/main.js sits at saha-app/electron/, repo root is two up.
//   packaged:   the .app contains no Python; v1 hardcodes Darshil's repo path.
//               7.2c+ will either bundle a Python venv or read SAHA_REPO_ROOT.
const REPO_ROOT = isDev
  ? path.resolve(__dirname, '..', '..')
  : (process.env.SAHA_REPO_ROOT || '/Users/darshil/sonnaeditor');

// Tracks whether we owned the spawn (only then do we kill on quit).
let backend = null;
let externalBackend = false;

function probeHealth(timeoutMs = 500) {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitForHealth(maxAttempts = 50, intervalMs = 200) {
  for (let i = 0; i < maxAttempts; i++) {
    if (await probeHealth(intervalMs)) return true;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}

function spawnBackend() {
  const cmd = process.platform === 'win32'
    ? `cd /d "${REPO_ROOT}" && uv run scripts/serve.py --port ${PORT}`
    : `cd "${REPO_ROOT}" && exec uv run scripts/serve.py --port ${PORT}`;
  const shellExe = process.platform === 'win32' ? 'cmd.exe' : '/bin/bash';
  const shellArgs = process.platform === 'win32' ? ['/c', cmd] : ['-lc', cmd];

  const proc = spawn(shellExe, shellArgs, {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  proc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  proc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  proc.on('exit', (code, signal) => {
    console.log(`[backend] exited code=${code} signal=${signal}`);
    backend = null;
  });
  return proc;
}

async function killBackend() {
  if (!backend || externalBackend) return;
  const proc = backend;
  backend = null;
  proc.kill('SIGTERM');
  // Give it 2s for clean shutdown, then SIGKILL.
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch { /* already gone */ }
      resolve();
    }, 2000);
    proc.once('exit', () => { clearTimeout(timer); resolve(); });
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 780,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#16140F',
    titleBarStyle: 'hiddenInset',
    icon: path.join(__dirname, '..', 'build', 'icon.png'),
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.once('ready-to-show', () => win.show());

  // Forward renderer console messages to the main-process stdout in dev so
  // `npm run dev` shows React/JS logs alongside the backend output.
  if (isDev) {
    win.webContents.on('console-message', (_event, _level, message) => {
      console.log(`[renderer] ${message}`);
    });
  }

  if (isDev) {
    win.loadURL('http://localhost:5173/');
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

function getAppPaths() {
  return {
    repoRoot: REPO_ROOT,
    profilesDir: path.resolve(REPO_ROOT, 'v1_learning'),
    capturesDir: path.resolve(REPO_ROOT, 'data', 'captures'),
  };
}

async function bootstrap() {
  // If a backend is already responding (e.g. dev where Darshil ran serve.py
  // himself, or a previous Electron session left it behind), reuse it and
  // skip the spawn-and-kill lifecycle.
  if (await probeHealth(500)) {
    externalBackend = true;
    console.log('[backend] external server detected on :8765, reusing');
  } else {
    console.log(`[backend] spawning via bash -lc, repo root: ${REPO_ROOT}`);
    backend = spawnBackend();
    const ready = await waitForHealth();
    if (!ready) {
      dialog.showErrorBox(
        'Saha — backend failed to start',
        `The Python API server at ${HEALTH_URL} did not respond within 10s.\n\n` +
        `Check that 'uv' is on your PATH and the repo at ${REPO_ROOT} is intact.`,
      );
      await killBackend();
      app.quit();
      return;
    }
  }
  createWindow();
}

// IPC: native folder picker for the renderer's Browse… button.
ipcMain.handle('saha:pick-folder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
    title: 'Choose a folder of RAW files',
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

// IPC: native file picker (single file). Used by the Lite-profile creation
// flow to pick a .xmp preset. The `filters` argument follows Electron's
// dialog.showOpenDialog filters spec — caller supplies the constraint.
ipcMain.handle('saha:pick-file', async (_event, opts) => {
  const { title, filters } = opts || {};
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    title: title || 'Choose a file',
    filters: filters || [],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

// IPC: open a path in Finder / Explorer (used by the profile view's
// "Profiles directory" link).
ipcMain.handle('saha:reveal-in-finder', async (_event, p) => {
  if (typeof p !== 'string' || !p) return false;
  await shell.openPath(p);
  return true;
});

ipcMain.handle('saha:get-app-paths', async () => getAppPaths());

app.whenReady().then(bootstrap);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// will-quit fires after all windows closed and the app is about to exit;
// we hold quit briefly so SIGTERM has a chance to run cleanly.
let quitting = false;
app.on('will-quit', (event) => {
  if (quitting || !backend || externalBackend) return;
  event.preventDefault();
  quitting = true;
  killBackend().finally(() => app.quit());
});
