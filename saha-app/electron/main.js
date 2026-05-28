// Electron main process — spawns the FastAPI backend, gates window creation
// on /api/health, cleanly terminates the subprocess on quit.

const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');

const isDev = !app.isPackaged;
const PORT = 8765;
const HEALTH_URL = `http://127.0.0.1:${PORT}/api/health`;

// Repo-root resolution:
//   dev:      electron/main.js sits at saha-app/electron/, repo root is two up.
//   package:  prefer SAHA_REPO_ROOT, otherwise use an app-data checkout path.
//             Packaging can later replace this with a bundled backend runtime.
const REPO_ROOT = isDev
  ? path.resolve(__dirname, '..', '..')
  : (process.env.SAHA_REPO_ROOT || path.join(app.getPath('userData'), 'sonna-editor'));

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

function getBackendCommand() {
  const venvPython = process.platform === 'win32'
    ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(REPO_ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) {
    return {
      command: venvPython,
      args: ['scripts/serve.py', '--port', String(PORT)],
      label: 'repo venv python',
    };
  }

  const uvCommand = process.platform === 'win32' ? 'uv.cmd' : 'uv';
  return {
    command: uvCommand,
    args: ['run', 'python', 'scripts/serve.py', '--port', String(PORT)],
    label: 'uv',
  };
}

function spawnBackend() {
  const backendCommand = getBackendCommand();
  console.log(`[backend] spawning via ${backendCommand.label}: ${backendCommand.command}`);

  const proc = spawn(backendCommand.command, backendCommand.args, {
    cwd: REPO_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  proc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  proc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  proc.on('error', (err) => {
    console.error(`[backend] failed to spawn: ${err.message}`);
  });
  proc.on('exit', (code, signal) => {
    console.log(`[backend] exited code=${code} signal=${signal}`);
    backend = null;
  });
  return proc;
}

async function waitForBackendReady(proc) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      proc.off('error', onError);
      resolve(result);
    };
    const onError = (err) => finish({ ready: false, error: err });
    proc.once('error', onError);
    waitForHealth().then((ready) => finish({ ready, error: null }));
  });
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
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
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
  // If a backend is already responding, reuse it and skip the spawn/kill
  // lifecycle. This is common during development on any OS.
  if (await probeHealth(500)) {
    externalBackend = true;
    console.log('[backend] external server detected on :8765, reusing');
  } else {
    console.log(`[backend] repo root: ${REPO_ROOT}`);
    backend = spawnBackend();
    const { ready, error } = await waitForBackendReady(backend);
    if (!ready) {
      const spawnHint = error
        ? `\n\nSpawn error: ${error.message}`
        : '';
      dialog.showErrorBox(
        'Saha — backend failed to start',
        `The Python API server at ${HEALTH_URL} did not respond within 10s.\n\n` +
        `Check that the repo virtual environment exists at ${path.join(REPO_ROOT, '.venv')} ` +
        `or that 'uv' is on your PATH.${spawnHint}`,
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

// IPC: reveal a path in the host file manager.
ipcMain.handle('saha:reveal-path', async (_event, p) => {
  if (typeof p !== 'string' || !p) return false;
  await shell.openPath(p);
  return true;
});

ipcMain.handle('saha:get-app-paths', async () => getAppPaths());

app.whenReady().then(bootstrap).catch((err) => {
  console.error('[electron] bootstrap failed', err);
  dialog.showErrorBox('Saha — startup failed', String(err && err.stack ? err.stack : err));
  app.quit();
});

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
