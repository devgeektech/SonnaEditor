// Reconnecting websocket: 3 attempts at 1s/2s/5s, then onFallback() so the
// caller can switch to polling. Status callbacks let the UI surface a
// "Reconnecting…" badge during the retry window.

const RETRY_DELAYS_MS = [1000, 2000, 5000];

export function connectJobStream({ url, onMessage, onStatus, onFallback }) {
  let ws = null;
  let attempt = 0;
  let stopped = false;
  let reconnectTimer = null;
  let terminalReceived = false;

  function emitStatus(status) {
    if (typeof onStatus === 'function') onStatus(status);
  }

  function open() {
    if (stopped) return;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      attempt = 0;
      emitStatus('live');
    };

    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (typeof onMessage === 'function') onMessage(msg);
      if (msg && (msg.type === 'job_complete' || msg.type === 'job_cancelled' || msg.type === 'job_failed')) {
        terminalReceived = true;
        // Server closes the socket after sending a terminal message; that
        // close should not trigger a reconnect.
      }
    };

    ws.onerror = () => { /* swallow; onclose will follow */ };

    ws.onclose = () => {
      ws = null;
      if (stopped || terminalReceived) return;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (stopped) return;
    if (attempt >= RETRY_DELAYS_MS.length) {
      emitStatus('polling');
      if (typeof onFallback === 'function') onFallback();
      return;
    }
    emitStatus('reconnecting');
    const delay = RETRY_DELAYS_MS[attempt];
    attempt += 1;
    reconnectTimer = setTimeout(() => { reconnectTimer = null; open(); }, delay);
  }

  function close() {
    stopped = true;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    if (ws) {
      try { ws.close(); } catch { /* ignore */ }
      ws = null;
    }
  }

  open();
  return { close };
}
