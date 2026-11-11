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
    throw new ApiError(`Network error: ${e.message}`, 0, null);
  }

  let body = null;
  const text = await resp.text();
  if (text) {
    try { body = JSON.parse(text); } catch { body = text; }
  }

  if (!resp.ok) {
    const detail = (body && typeof body === 'object' && (body.detail || body.error)) || resp.statusText;
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
