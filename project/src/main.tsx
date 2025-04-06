import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Prefetch important assets
const prefetchAssets = () => {
  if (navigator.connection && navigator.connection.saveData) {
    // Skip prefetching if the user has enabled data-saving mode
    return
  }
  
  // Prefetch important routes
  const links = [
    '/login',
    '/dashboard',
    '/spaces'
  ]
  
  links.forEach(href => {
    const link = document.createElement('link')
    link.rel = 'prefetch'
    link.href = href
    document.head.appendChild(link)
  })
}

// Optimize React rendering by using concurrent features
const root = createRoot(document.getElementById('root')!, {
  // Force concurrent features
  identifierPrefix: 'app-'
})

root.render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Defer non-critical initialization
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    setTimeout(() => {
      prefetchAssets()
    }, 1000)
  })
}
