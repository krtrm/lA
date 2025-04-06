import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import { ErrorBoundary } from './components/ErrorBoundary'

// Lazy load all pages for better initial load performance
const LandingPage = lazy(() => import('./pages/LandingPage'))
const LoginPage = lazy(() => import('./pages/LoginPage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const NewsPage = lazy(() => import('./pages/NewsPage'))
const SpacesPage = lazy(() => import('./pages/SpacesPage'))
const PaymentsPage = lazy(() => import('./pages/PaymentsPage'))

// Loading fallback component
const PageLoader = () => (
  <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
    <div className="p-4 max-w-md w-full">
      <div className="skeleton h-8 w-2/3 mb-4"></div>
      <div className="skeleton h-4 w-full mb-2"></div>
      <div className="skeleton h-4 w-5/6 mb-2"></div>
      <div className="skeleton h-4 w-4/6 mb-6"></div>
      <div className="skeleton h-32 w-full mb-4 rounded-lg"></div>
    </div>
  </div>
)

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-background text-foreground">
          <Navbar />
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/news" element={<NewsPage />} />
              <Route path="/spaces" element={<SpacesPage />} />
              <Route path="/payments" element={<PaymentsPage />} />
            </Routes>
          </Suspense>
        </div>
      </Router>
    </ErrorBoundary>
  )
}

export default App