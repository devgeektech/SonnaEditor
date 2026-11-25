import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.jsx';
import { applyTheme } from './tokens.js';

// Global error handlers in the renderer to ensure no errors are silently swallowed
if (typeof window !== 'undefined') {
  window.addEventListener('error', (event) => {
    const err = event.error || event;
    const loc = event.filename ? ` (${event.filename}:${event.lineno}:${event.colno})` : '';
    console.error(`[Uncaught Error] ${err?.stack || err?.message || String(err)}${loc}`);
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    console.error(`[Unhandled Promise Rejection] ${reason?.stack || reason?.message || String(reason)}`);
  });
}

// Default to dark theme on cold start
applyTheme('dark');

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
