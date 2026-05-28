// Preload — minimal contextBridge surface exposed to the renderer as window.saha.
// Network calls happen in the renderer with fetch(); we don't proxy them.

const { contextBridge, ipcRenderer } = require('electron');

const API_BASE_URL = 'http://127.0.0.1:8765';

contextBridge.exposeInMainWorld('saha', {
  // Open native macOS folder picker. Returns absolute path or null.
  pickFolder: () => ipcRenderer.invoke('saha:pick-folder'),

  // Open native macOS file picker (single file). opts: { title?, filters? }
  // where `filters` follows Electron's showOpenDialog filters spec. Returns
  // absolute file path, or null if the user cancels.
  pickFile: (opts) => ipcRenderer.invoke('saha:pick-file', opts),

  // Open a directory in Finder / Explorer. Used by the profile view to
  // expose the checkpoints folder. No-op on non-existent paths.
  revealInFinder: (path) => ipcRenderer.invoke('saha:reveal-in-finder', path),

  // Return the runtime paths the renderer should use for generated assets.
  getAppPaths: () => ipcRenderer.invoke('saha:get-app-paths'),

  // Constant for now; a future preference could redirect this.
  apiBaseUrl: () => API_BASE_URL,
});
