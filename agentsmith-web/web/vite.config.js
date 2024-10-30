import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import viteBasicSslPlugin from "@vitejs/plugin-basic-ssl";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    viteBasicSslPlugin({
      /** name of certification */
      name: 'app.local',
      /** custom trust domains */
      domains: ['app.local'],
      /** custom certification directory */
      certDir: './certs'
    }),
    react(),
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:3001',
    },
  },
})
