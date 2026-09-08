import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { sourcemap: false },
  // API access in production is always same-origin through the C++ host.
  // Development also uses a same-origin proxy; upstream Host and Origin are normalized.
  server: { proxy: { '/api': {
    target: 'http://127.0.0.1:8877', changeOrigin: true,
    configure: (proxy) => proxy.on('proxyReq', (req) => {
      req.removeHeader('origin'); req.removeHeader('sec-fetch-site');
    }),
  } } },
});
