import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.jsx';
import { applyTheme } from './tokens.js';

// Default to dark theme on cold start
applyTheme('dark');

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
