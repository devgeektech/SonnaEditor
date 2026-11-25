// HTTP client for the Sonna Editor FastAPI backend.
// One function per endpoint. Throws ApiError on non-2xx so callers can pattern
// on { source, message } in the editor's error banner.

const BASE = (typeof window !== 'undefined' && window.saha?.apiBaseUrl)
  ? window.saha.apiBaseUrl()
  : 'http://127.0.0.1:8765';

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

async function request(path, init = {}) {
  const url = `${BASE}${path}`;
  const headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
  let resp;
  try {
    resp = await fetch(url, { ...init, headers });
  } catch (e) {
    // If Electron main process can revive a stopped backend, recover and retry once.
    if (typeof window !== 'undefined' && typeof window.saha?.ensureBackend === 'function') {
      try {
        console.warn(`[API] ${init.method || 'GET'} ${path} connection failed; restoring backend...`);
        const recovered = await window.saha.ensureBackend();
        if (recovered) {
          resp = await fetch(url, { ...init, headers });
        }
      } catch (retryErr) {
        console.error(`[API NETWORK ERROR] ${init.method || 'GET'} ${path} - Failed to connect to backend: ${retryErr.message}`);
        throw new ApiError(`Network error: ${retryErr.message}`, 0, null);
      }
    }
    if (!resp) {
      console.error(`[API NETWORK ERROR] ${init.method || 'GET'} ${path} - Failed to connect to backend: ${e.message}`);
      throw new ApiError(`Network error: ${e.message}`, 0, null);
    }
  }

  let body = null;
  const text = await resp.text();
  if (text) {
    try { body = JSON.parse(text); } catch { body = text; }
  }

  if (!resp.ok) {
    const detail = (body && typeof body === 'object' && (body.detail || body.error)) || resp.statusText;
    console.error(`[API ERROR] ${init.method || 'GET'} ${path} (${resp.status}): ${detail}`);
    throw new ApiError(String(detail), resp.status, body);
  }
  return body;
}

// ── Health ─────────────────────────────────────────────────
export const getHealth = () => request('/api/health');

// ── Profiles ───────────────────────────────────────────────
export const listProfiles = () => request('/api/profiles');
export const activateProfile = (id) =>
  request(`/api/profiles/${encodeURIComponent(id)}/activate`, { method: 'POST' });
export const deleteProfile = (id) =>
  request(`/api/profiles/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const createPersonalProfile = (req) =>
  request('/api/profiles/personal', { method: 'POST', body: JSON.stringify(req) });
export const createLiteProfile = (req) =>
  request('/api/profiles/lite', { method: 'POST', body: JSON.stringify(req) });

// ── Folders ────────────────────────────────────────────────
export const scanFolder = (folder_path, source_type = 'folder') =>
  request('/api/folders/scan', { method: 'POST', body: JSON.stringify({ folder_path, source_type }) });
export const listRecentFolders = () => request('/api/folders/recent');
export const listCatalogs = () => request('/api/catalogs');

// ── Process / jobs ─────────────────────────────────────────
export const startProcess = (req) =>
  request('/api/process', { method: 'POST', body: JSON.stringify(req) });
export const getJob = (job_id) =>
  request(`/api/jobs/${encodeURIComponent(job_id)}`);
export const cancelJob = (job_id) =>
  request(`/api/jobs/${encodeURIComponent(job_id)}/cancel`, { method: 'POST' });

// ── Captures + finetune ────────────────────────────────────
export const fetchCaptures = () => request('/api/captures');
export const startFineTune = (req) =>
  request('/api/finetune', { method: 'POST', body: JSON.stringify(req) });

// ── Websocket URL helper ───────────────────────────────────
export function jobStreamUrl(job_id) {
  const wsBase = BASE.replace(/^http/, 'ws');
  return `${wsBase}/api/jobs/${encodeURIComponent(job_id)}/stream`;
}
