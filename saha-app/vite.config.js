import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// Vite roots at src/ so index.html / index.jsx live next to each other.
// Output goes to ../dist (saha-app/dist), which electron-builder bundles.
export default defineConfig({
  root: 'src',
  base: './',
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
