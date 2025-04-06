import { Routes, Route, Navigate } from "react-router-dom";
import { SignedIn, SignedOut, useAuth, UserButton } from "@clerk/clerk-react";
import SignInPage from "./components/auth/SignIn";
import SignUpPage from "./components/auth/SignUp";
import { ApiProvider } from "./context/ApiContext";
import ChatInterface from "./components/ChatInterface";
import LandingPage from "./components/LandingPage";
import DashboardPage from "./components/DashboardPage";
import SpacesPage from "./components/SpacesPage";
import NewsPage from "./components/NewsPage";
import { Link } from "react-router-dom";

// Navigation bar component with user button
const Navbar = () => {
  const { isSignedIn } = useAuth();
  
  return (
    <header className="fixed top-0 left-0 right-0 p-4 z-10 glass-effect">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <Link to="/" className="text-xl font-bold gradient-text">Vaqeel</Link>
        <div className="flex items-center gap-6">
          <SignedIn>
            <div className="flex items-center gap-6">
              <Link to="/dashboard" className="text-foreground/80 hover:text-foreground">Dashboard</Link>
              <Link to="/spaces" className="text-foreground/80 hover:text-foreground">Spaces</Link>
              <Link to="/news" className="text-foreground/80 hover:text-foreground">News</Link>
              <UserButton afterSignOutUrl="/" />
            </div>
          </SignedIn>
          <SignedOut>
            <div className="flex gap-4">
              <Link to="/sign-in" className="button-primary">Sign In</Link>
              <Link to="/sign-up" className="button-secondary">Sign Up</Link>
            </div>
          </SignedOut>
        </div>
      </div>
    </header>
  );
};

function App() {
  const { isLoaded, isSignedIn } = useAuth();

  // Show loading indicator while Clerk is initializing
  if (!isLoaded) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="loading-spinner" />
      </div>
    );
  }

  return (
    <ApiProvider>
      <div className="min-h-screen bg-background">
        <Navbar />
        <Routes>
          {/* Public home route - show landing page */}
          <Route index element={
            isSignedIn ? <Navigate to="/dashboard" replace /> : <LandingPage />
          } />
          <Route path="/" element={
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
                  <Navigate to="/" replace />
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
                  <Navigate to="/" replace />
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
                  <Navigate to="/" replace />
                </SignedOut>
              </>
            }
          />
          
          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </ApiProvider>
  );
}

export default App;