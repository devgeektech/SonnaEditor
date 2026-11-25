// Electron main process — spawns the FastAPI backend, gates window creation
// on /api/health, cleanly terminates the subprocess on quit.

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const { spawn, spawnSync, execSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');

// Completely disable Electron default application menu across all windows
Menu.setApplicationMenu(null);

// Centralized process-level error logging for the Electron main process
process.on('uncaughtException', (err) => {
  console.error(`\x1b[1;31m[electron ERROR]\x1b[0m Uncaught Exception: ${err?.stack || err?.message || err}`);
});

process.on('unhandledRejection', (reason) => {
  console.error(`\x1b[1;31m[electron ERROR]\x1b[0m Unhandled Rejection: ${reason?.stack || reason?.message || reason}`);
});

const isDev = !app.isPackaged;
const PORT = 8765;
const HEALTH_URL = `http://127.0.0.1:${PORT}/api/health`;
const HEALTH_PROBE_TIMEOUT_MS = 1000;
const HEALTH_RETRY_INTERVAL_MS = 250;

// Ensure standard binary locations (uv, Python, Homebrew / Windows node, local bin)
// are on process.env.PATH even when launched via GUI / double-clicking Saha.app or batch files.
if (process.platform === 'darwin') {
  const home = process.env.HOME || '';
  const extraPaths = [
    path.join(home, '.local', 'bin'),
    path.join(home, 'bin'),
    '/opt/homebrew/bin',
    '/opt/homebrew/sbin',
    '/usr/local/bin',
  ];
  const currentPath = process.env.PATH || '';
  process.env.PATH = `${extraPaths.join(':')}:${currentPath}`;
} else if (process.platform === 'win32') {
  const userProfile = process.env.USERPROFILE || '';
  const extraPaths = [
    'C:\\Program Files\\nodejs',
    path.join(userProfile, '.local', 'bin'),
    path.join(userProfile, 'AppData', 'Roaming', 'npm'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python311'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'Scripts'),
  ].filter(p => fs.existsSync(p));
  const currentPath = process.env.PATH || '';
  process.env.PATH = `${extraPaths.join(';')};${currentPath}`;
}

function backendStartupTimeoutMs() {
  const value = Number(process.env.SAHA_BACKEND_STARTUP_TIMEOUT_MS || 60000);
  return Number.isFinite(value) && value > 0 ? value : 60000;
}

// Repo-root resolution:
// Multi-candidate search to ensure backend is found even when launched
// via packaged build, relative path, or custom cwd.
function resolveRepoRoot() {
  const candidates = [
    process.env.SAHA_REPO_ROOT,
    isDev ? path.resolve(__dirname, '..', '..') : null,
    path.resolve(__dirname, '..', '..'),
    path.resolve(process.resourcesPath || '', '..'),
    path.resolve(process.resourcesPath || '', 'app'),
    process.cwd(),
    path.join(app.getPath('userData'), 'sonna-editor'),
  ].filter(Boolean);

  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, 'scripts', 'serve.py')) || fs.existsSync(path.join(dir, '.venv'))) {
      return dir;
    }
  }
  return candidates[0] || path.resolve(__dirname, '..', '..');
}

const REPO_ROOT = resolveRepoRoot();

// Tracks whether we owned the spawn (only then do we kill on quit).
let backend = null;
let externalBackend = false;
let quitting = false;
let backendRestartingPromise = null;
let watchdogTimer = null;
let consecutiveHealthFailures = 0;

function freeStalePort(port) {
  try {
    if (process.platform === 'win32') {
      const psCommand = `try { Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } } catch {}; try { Get-CimInstance Win32_Process -Filter "CommandLine LIKE '%scripts/serve.py%'" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } catch {}; exit 0`;
      execSync(`powershell -NoProfile -NonInteractive -Command "${psCommand}"`, { stdio: 'ignore', timeout: 5000 });
    } else {
      execSync(`lsof -ti :${port} | xargs kill -9 2>/dev/null || true`, { stdio: 'ignore', timeout: 5000 });
    }
  } catch (e) {
    // Non-fatal if no lingering process
  }
}

function probeHealth(timeoutMs = HEALTH_PROBE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitForHealth(timeoutMs = backendStartupTimeoutMs()) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await probeHealth()) return true;
    await new Promise((r) => setTimeout(r, HEALTH_RETRY_INTERVAL_MS));
  }
  return false;
}

function getBackendCommand() {
  const venvPythonCandidates = [
    process.platform === 'win32'
      ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
      : path.join(REPO_ROOT, '.venv', 'bin', 'python'),
    process.platform === 'win32'
      ? path.join(process.cwd(), '.venv', 'Scripts', 'python.exe')
      : path.join(process.cwd(), '.venv', 'bin', 'python'),
  ];

  for (const venvPython of venvPythonCandidates) {
    if (fs.existsSync(venvPython)) {
      return {
        command: venvPython,
        args: ['scripts/serve.py', '--port', String(PORT)],
        label: 'repo venv python',
      };
    }
  }

  let uvCommand = 'uv';
  if (process.platform === 'win32') {
    const userProfile = process.env.USERPROFILE || '';
    const uvCandidate = path.join(userProfile, '.local', 'bin', 'uv.exe');
    if (fs.existsSync(uvCandidate)) {
      uvCommand = uvCandidate;
    } else {
      uvCommand = 'uv.exe';
    }
  }

  return {
    command: uvCommand,
    args: ['run', 'python', 'scripts/serve.py', '--port', String(PORT)],
    label: 'uv',
  };
}

function logBackendLine(rawLine, isStderr) {
  const line = rawLine.trim();
  if (!line) return;
  if (/^(INFO|DEBUG):/i.test(line)) {
    console.log(`[backend] ${rawLine}`);
    return;
  }
  if (/^(WARNING|WARN):/i.test(line) || (!isStderr && /WARNING|WARN/i.test(line))) {
    console.warn(`\x1b[33m[backend WARN]\x1b[0m ${rawLine}`);
    return;
  }
  if (isStderr || /ERROR|CRITICAL|Traceback|Exception|Failed|Error:/i.test(line)) {
    console.error(`\x1b[1;31m[backend ERROR]\x1b[0m ${rawLine}`);
    return;
  }
  console.log(`[backend] ${rawLine}`);
}

function pipeBackendStream(stream, isStderr = false) {
  let buffer = '';
  stream.on('data', (chunk) => {
    buffer += chunk.toString('utf-8');
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';
    for (const rawLine of lines) {
      logBackendLine(rawLine, isStderr);
    }
  });
  stream.on('end', () => {
    if (buffer.trim()) {
      logBackendLine(buffer, isStderr);
      buffer = '';
    }
  });
}

function spawnBackend() {
  const backendCommand = getBackendCommand();
  console.log(`[backend] spawning via ${backendCommand.label}: ${backendCommand.command}`);

  const proc = spawn(backendCommand.command, backendCommand.args, {
    cwd: REPO_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  pipeBackendStream(proc.stdout, false);
  pipeBackendStream(proc.stderr, true);
  proc.on('error', (err) => {
    console.error(`\x1b[1;31m[backend ERROR]\x1b[0m failed to spawn: ${err.message}`);
  });
  proc.on('exit', (code, signal) => {
    console.log(`[backend] exited code=${code} signal=${signal}`);
    if (backend === proc) {
      backend = null;
    }
    if (!quitting && !externalBackend) {
      console.warn('\x1b[33m[backend WARN]\x1b[0m Backend exited unexpectedly. Triggering automatic restart...');
      ensureBackendRunning().catch((err) => {
        console.error(`\x1b[1;31m[backend ERROR]\x1b[0m Auto-restart failed: ${err?.message || err}`);
      });
    }
  });
  return proc;
}

async function ensureBackendRunning() {
  if (quitting) return false;
  if (backendRestartingPromise) return backendRestartingPromise;

  backendRestartingPromise = (async () => {
    try {
      if (await probeHealth(1000)) {
        consecutiveHealthFailures = 0;
        return true;
      }
      if (quitting) return false;

      console.warn('\x1b[33m[backend WARN]\x1b[0m Backend is unreachable on :8765. Restarting...');
      if (backend && !externalBackend) {
        await killBackend();
      } else {
        freeStalePort(PORT);
      }
      externalBackend = false;
      await new Promise((r) => setTimeout(r, 300));
      if (quitting) return false;

      backend = spawnBackend();
      const { ready, error } = await waitForBackendReady(backend);
      if (ready) {
        consecutiveHealthFailures = 0;
        console.log('[backend] recovered and ready on :8765');
        return true;
      }
      console.error(`\x1b[1;31m[backend ERROR]\x1b[0m Recovery failed${error ? `: ${error.message}` : ''}`);
      return false;
    } finally {
      backendRestartingPromise = null;
    }
  })();

  return backendRestartingPromise;
}

function startBackendWatchdog() {
  if (watchdogTimer) return;
  watchdogTimer = setInterval(async () => {
    if (quitting || backendRestartingPromise) return;
    const ok = await probeHealth(1500);
    if (ok) {
      consecutiveHealthFailures = 0;
      return;
    }
    consecutiveHealthFailures += 1;
    if (consecutiveHealthFailures >= 2) {
      await ensureBackendRunning();
    }
  }, 10000);
  if (typeof watchdogTimer.unref === 'function') {
    watchdogTimer.unref();
  }
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
  if (process.platform === 'win32' && proc.pid) {
    try {
      spawnSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 3000 });
    } catch { }
    freeStalePort(PORT);
  } else {
    try { proc.kill('SIGTERM'); } catch { }
  }
  // Give it 2s for clean shutdown, then SIGKILL.
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      try {
        if (process.platform === 'win32' && proc.pid) {
          spawnSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 2000 });
        } else {
          proc.kill('SIGKILL');
        }
      } catch { /* already gone */ }
      resolve();
    }, 1500);
    proc.once('exit', () => { clearTimeout(timer); resolve(); });
  });
}

function createWindow() {
  Menu.setApplicationMenu(null);

  const win = new BrowserWindow({
    width: 1280,
    height: 780,
    minWidth: 640,
    minHeight: 500,
    backgroundColor: '#16140F',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    autoHideMenuBar: false,
    icon: path.join(__dirname, '..', 'build', 'icon.png'),
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (typeof win.removeMenu === 'function') {
    win.removeMenu();
  }
  win.setMenu(null);
  win.setMenuBarVisibility(false);
  win.setAutoHideMenuBar(false);

  win.once('ready-to-show', () => {
    win.show();
    win.focus();
  });

  // Guarantee window is shown even if ready-to-show is delayed
  setTimeout(() => {
    if (!win.isDestroyed() && !win.isVisible()) {
      win.show();
      win.focus();
    }
  }, 1200);

  // Forward renderer console messages to the main-process stdout/stderr so
  // terminal operators immediately spot errors, warnings, and unhandled rejections.
  win.webContents.on('console-message', (_event, level, message, line, sourceId) => {
    // level: 0 = verbose, 1 = info, 2 = warning, 3 = error
    const isError = level === 3 || message.startsWith('Uncaught') || message.includes('Error:') || message.includes('[API ERROR]') || message.includes('[API NETWORK ERROR]');
    const isWarn = level === 2;
    const loc = sourceId ? ` (${path.basename(sourceId)}:${line})` : '';

    if (isError) {
      console.error(`\x1b[1;31m[renderer ERROR]\x1b[0m ${message}${loc}`);
    } else if (isWarn) {
      console.warn(`\x1b[33m[renderer WARN]\x1b[0m ${message}${loc}`);
    } else if (isDev) {
      console.log(`[renderer] ${message}`);
    }
  });

  win.webContents.on('render-process-gone', (_event, details) => {
    console.error(`\x1b[1;31m[renderer CRASH]\x1b[0m Renderer process terminated unexpectedly: reason=${details.reason}, exitCode=${details.exitCode}`);
  });

  win.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => {
    console.error(`\x1b[1;31m[renderer LOAD ERROR]\x1b[0m Failed to load ${validatedURL} (code ${errorCode}): ${errorDescription}`);
  });

  win.webContents.on('unresponsive', () => {
    console.error(`\x1b[1;31m[renderer ERROR]\x1b[0m Window webContents has become unresponsive.`);
  });

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
    // Proactively clear any hung/dead zombie process occupying port 8765
    freeStalePort(PORT);
    await new Promise((r) => setTimeout(r, 200));

    console.log(`[backend] repo root: ${REPO_ROOT}`);
    backend = spawnBackend();
    const { ready, error } = await waitForBackendReady(backend);
    if (!ready) {
      const spawnHint = error
        ? `\n\nSpawn error: ${error.message}`
        : '';
      dialog.showErrorBox(
        'Saha — backend failed to start',
        `The Python API server at ${HEALTH_URL} did not respond within ${Math.round(backendStartupTimeoutMs() / 1000)}s.\n\n` +
        `Check that the repo virtual environment exists at ${path.join(REPO_ROOT, '.venv')} ` +
        `or that 'uv' is on your PATH.${spawnHint}`,
      );
      await killBackend();
      app.quit();
      return;
    }
  }
  createWindow();
  startBackendWatchdog();
}

// IPC: native folder picker for the renderer's Browse… button.
ipcMain.handle('saha:pick-folder', async () => {
  const parentWindow = BrowserWindow.getFocusedWindow() || BrowserWindow.getAllWindows()[0] || null;
  const options = {
    properties: ['openDirectory'],
    title: 'Choose a folder of RAW files',
  };
  const result = parentWindow
    ? await dialog.showOpenDialog(parentWindow, options)
    : await dialog.showOpenDialog(options);
  if (parentWindow && !parentWindow.isDestroyed()) {
    if (parentWindow.isMinimized()) parentWindow.restore();
    parentWindow.focus();
  }
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

// IPC: get file stats (size, isFile, isDirectory).
ipcMain.handle('saha:get-file-stat', async (_event, filePath) => {
  if (!filePath || typeof filePath !== 'string') return null;
  try {
    const stat = fs.statSync(filePath);
    return {
      size: stat.size,
      isFile: stat.isFile(),
      isDirectory: stat.isDirectory(),
      mtime: stat.mtimeMs,
    };
  } catch (e) {
    return null;
  }
});

// IPC: reveal a path in the host file manager.
ipcMain.handle('saha:reveal-path', async (_event, p) => {
  if (typeof p !== 'string' || !p) return false;
  await shell.openPath(p);
  return true;
});

// IPC: open a catalog (.lrcat) or folder directly in Adobe Lightroom Classic.
ipcMain.handle('saha:open-lightroom', async (_event, targetPath) => {
  const cleanPath = typeof targetPath === 'string' ? targetPath.trim() : '';
  try {
    if (process.platform === 'win32') {
      const candidates = [
        path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Adobe', 'Adobe Lightroom Classic', 'Lightroom.exe'),
        path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Adobe', 'Adobe Lightroom Classic', 'Lightroom.exe'),
      ];
      const exe = candidates.find((c) => fs.existsSync(c));
      if (exe) {
        const args = cleanPath && fs.existsSync(cleanPath) ? [cleanPath] : [];
        const child = spawn(exe, args, { detached: true, stdio: 'ignore' });
        child.unref();
        return { ok: true };
      }
    } else if (process.platform === 'darwin') {
      const appPath = '/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app';
      if (fs.existsSync(appPath)) {
        const args = ['-a', 'Adobe Lightroom Classic'];
        if (cleanPath && fs.existsSync(cleanPath)) args.push(cleanPath);
        const child = spawn('open', args, { detached: true, stdio: 'ignore' });
        child.unref();
        return { ok: true };
      }
    }

    if (cleanPath && cleanPath.toLowerCase().endsWith('.lrcat') && fs.existsSync(cleanPath)) {
      const err = await shell.openPath(cleanPath);
      return { ok: !err, error: err || null };
    }
    return { ok: false, error: 'Adobe Lightroom Classic was not found at the default location.' };
  } catch (err) {
    return { ok: false, error: err?.message || String(err) };
  }
});

ipcMain.handle('saha:get-app-paths', async () => getAppPaths());
ipcMain.handle('saha:ensure-backend', async () => ensureBackendRunning());

app.on('browser-window-created', (_event, window) => {
  if (typeof window.removeMenu === 'function') window.removeMenu();
  window.setMenu(null);
  window.setMenuBarVisibility(false);
  window.setAutoHideMenuBar(false);
});

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  return bootstrap();
}).catch((err) => {
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

app.on('before-quit', () => {
  quitting = true;
  if (watchdogTimer) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
  if (backend && !externalBackend) {
    try {
      if (backend.pid && process.platform === 'win32') {
        spawnSync('taskkill', ['/pid', String(backend.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
      }
    } catch { }
  }
});

// will-quit fires after all windows closed and the app is about to exit;
// we hold quit briefly so SIGTERM has a chance to run cleanly.
app.on('will-quit', (event) => {
  if (watchdogTimer) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
  if (quitting || !backend || externalBackend) {
    quitting = true;
    freeStalePort(PORT);
    return;
  }
  event.preventDefault();
  quitting = true;
  killBackend().finally(() => {
    freeStalePort(PORT);
    app.quit();
  });
});
