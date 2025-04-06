import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { ClerkProvider, SignedIn, SignedOut, useAuth } from '@clerk/clerk-react';
import { SignIn, SignUp } from '@clerk/clerk-react';
import { ApiProvider } from './context/ApiContext.jsx'; // Use explicit .jsx extension
import Navbar from './components/Navbar';
import { Toaster } from 'react-hot-toast';
import { Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

// Import pages that exist
import DashboardPage from './pages/DashboardPage';
import SpacesPage from './pages/SpacesPage';

// Remove imports of files that don't exist
// import LandingPage from './pages/LandingPage';
// import SignInPage from './pages/SignInPage';
// import SignUpPage from './pages/SignUpPage';
// import NewsPage from './pages/NewsPage';

const CLERK_PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || 'pk_test_dummy-key-for-development';

function AppContent() {
  const { isSignedIn, isLoaded } = useAuth();
  console.log('useAuth:', { isSignedIn, isLoaded }); // Debugging log
  
  // Create a simple landing component with Link navigation
  const LandingComponent = () => (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4 text-white">Welcome to Vaqeel</h1>
        <p className="mb-6 text-white">Your legal research assistant</p>
        <div className="flex gap-4 justify-center">
          {/* Changed from <a> to <Link> */}
          <Link to="/sign-in" className="px-4 py-2 bg-primary text-white rounded">Sign In</Link>
          <Link to="/sign-up" className="px-4 py-2 border border-primary text-primary rounded">Sign Up</Link>
        </div>
      </div>
    </div>
  );
  
  if (!isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          className="flex flex-col items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
          <p className="text-foreground/70">Loading your account...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      <main className="pt-16"> {/* Add padding to account for fixed navbar */}
        <Routes>
          {/* Public home route - show landing page */}
          <Route index element={
            isSignedIn ? <Navigate to="/dashboard" replace /> : <LandingComponent />
          } />
          
          {/* Auth routes - use Clerk's built-in components */}
          <Route path="/sign-in/*" element={
            isSignedIn ? <Navigate to="/dashboard" replace /> : <SignIn />
          } />
          <Route path="/sign-up/*" element={
            isSignedIn ? <Navigate to="/dashboard" replace /> : <SignUp />
          } />
          
          {/* Protected routes */}
          <Route
            path="/dashboard"
            element={
              <>
                <SignedIn>
                  <DashboardPage />
                </SignedIn>
                <SignedOut>
                  <Navigate to="/sign-in" replace />
                </SignedOut>
              </>
            }
          />
          
          <Route
            path="/spaces"
            element={
              <>
                <SignedIn>
                  <SpacesPage />
                </SignedIn>
                <SignedOut>
                  <Navigate to="/sign-in" replace />
                </SignedOut>
              </>
            }
          />
          
          <Route
            path="/spaces/:spaceId"
            element={
              <>
                <SignedIn>
                  <SpacesPage />
                </SignedIn>
                <SignedOut>
                  <Navigate to="/sign-in" replace />
                </SignedOut>
              </>
            }
          />
          
          <Route
            path="/news"
            element={
              <>
                <SignedIn>
                  {/* Replace with a simple placeholder since NewsPage doesn't exist */}
                  <div className="container mx-auto p-4">
                    <h1 className="text-2xl font-bold mb-4">News</h1>
                    <p>News feature coming soon!</p>
                  </div>
                </SignedIn>
                <SignedOut>
                  <Navigate to="/sign-in" replace />
                </SignedOut>
              </>
            }
          />
          
          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      
      {/* Toast notifications */}
      <Toaster 
        position="bottom-right"
        toastOptions={{
          style: {
            background: 'rgba(10, 15, 25, 0.8)',
            color: '#fff',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '0.5rem',
          },
        }}
      />
    </div>
  );
}

function App() {
  return (
    <ClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY}>
      <ApiProvider>
        <Router>
          <AppContent />
        </Router>
      </ApiProvider>
    </ClerkProvider>
  );
}

export default App;