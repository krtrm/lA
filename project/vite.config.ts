import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { splitVendorChunkPlugin } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Fast refresh optimizations
      fastRefresh: true,
      babel: {
        plugins: [
          ['@babel/plugin-transform-react-jsx', { runtime: 'automatic' }]
        ]
      }
    }),
    splitVendorChunkPlugin()
  ],
  build: {
    // Optimize chunk size
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui': ['framer-motion']
        }
      }
    },
    // Enable sourcemap for production for easier debugging
    sourcemap: true,
    // Minimize output
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: true
      }
    }
  },
  optimizeDeps: {
    include: ['react-router-dom', 'framer-motion'],
    exclude: [],
  },
  server: {
    hmr: {
      overlay: true
    },
    // Increase timeouts to prevent disconnects
    timeout: 120000,
    // Optimize for Network
    host: true
  },
  css: {
    devSourcemap: true
  }
})
