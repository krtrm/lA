import { Routes, Route, Navigate } from "react-router-dom";
import { SignedIn, SignedOut, useAuth } from "@clerk/clerk-react";
import { Toaster } from 'react-hot-toast';
import { Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

import SignInPage from "./components/auth/SignIn";
import SignUpPage from "./components/auth/SignUp";
import { ApiProvider } from "./context/ApiContext";
import DashboardPage from "./pages/DashboardPage";
import SpacesPage from "./pages/SpacesPage";
import NewsPage from "./pages/NewsPage";
import Navbar from "./components/Navbar";
import LandingPage from "./components/LandingPage";

function App() {
  const { isLoaded, isSignedIn } = useAuth();

  // Show loading indicator while Clerk is initializing
  if (!isLoaded) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center"
        >
          <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
          <p className="text-foreground/70">Loading your account...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <ApiProvider>
      <div className="min-h-screen bg-background">
        <Navbar />
        
        <main className="pt-16"> {/* Add padding to account for fixed navbar */}
          <Routes>
            {/* Public home route - show landing page */}
            <Route index element={
              isSignedIn ? <Navigate to="/dashboard" replace /> : <LandingPage />
            } />
            
            {/* Auth routes */}
            <Route path="/sign-in/*" element={
              isSignedIn ? <Navigate to="/dashboard" replace /> : <SignInPage />
            } />
            <Route path="/sign-up/*" element={
              isSignedIn ? <Navigate to="/dashboard" replace /> : <SignUpPage />
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
                    <NewsPage />
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
    </ApiProvider>
  );
}

export default App;