import {defineConfig} from 'vite';

// `npm run dev` serves the UI on :5173 and forwards /api to the Python server (python -m app.server).
export default defineConfig({
  server: {proxy: {'/api': 'http://127.0.0.1:8000'}},
  build: {chunkSizeWarningLimit: 2000},
});
