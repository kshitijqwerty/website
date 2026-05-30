import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

function nonBlockingCss() {
  return {
    name: 'non-blocking-css',
    enforce: 'post',
    transformIndexHtml: {
      order: 'post',
      handler(html) {
        return html.replace(
            /<link rel="stylesheet"[^>]*href="([^"]+\.css)"[^>]*>/,
            (match, href) =>
              `<link rel="preload" as="style" href="${href}" onload="this.onload=null;this.rel='stylesheet'" />\n    <noscript>${match}</noscript>`
          );
      },
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), nonBlockingCss()],
  base: './',
  build: {
    target: 'esnext',
    cssMinify: 'esbuild',
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-icons': ['lucide-react'],
        },
      },
    },
  },
})
