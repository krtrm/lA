import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { motion } from 'framer-motion';
import { LucideBook, MessageSquare, LayoutDashboard, Menu, X } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  
  const isActive = (path: string) => {
    return location.pathname === path;
  };
  
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-effect border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="text-xl font-bold gradient-text flex items-center gap-2">
          <motion.div
            initial={{ rotate: -10 }}
            animate={{ rotate: 0 }}
            transition={{ duration: 0.5 }}
          >
            <LucideBook className="w-6 h-6 text-blue-400" />
          </motion.div>
          Vaqeel
        </Link>
        
        {/* Mobile menu button */}
        <button 
          className="md:hidden p-2 rounded-lg hover:bg-white/5"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? (
            <X className="w-5 h-5 text-white" />
          ) : (
            <Menu className="w-5 h-5 text-white" />
          )}
        </button>
        
        {/* Desktop navigation */}
        <div className="hidden md:flex items-center gap-6">
          <SignedIn>
            <nav className="flex items-center space-x-1">
              <Link 
                to="/dashboard" 
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
              >
                <div className="flex items-center gap-1.5">
                  <LayoutDashboard className="w-4 h-4" />
                  <span>Dashboard</span>
                </div>
              </Link>
              <Link 
                to="/spaces" 
                className={`nav-link ${isActive('/spaces') ? 'active' : ''}`}
              >
                <div className="flex items-center gap-1.5">
                  <MessageSquare className="w-4 h-4" />
                  <span>Spaces</span>
                </div>
              </Link>
              <Link 
                to="/news" 
                className={`nav-link ${isActive('/news') ? 'active' : ''}`}
              >
                <div className="flex items-center gap-1.5">
                  <LucideBook className="w-4 h-4" />
                  <span>News</span>
                </div>
              </Link>
            </nav>
            
            <div className="ml-4 pl-4 border-l border-white/10">
              <UserButton afterSignOutUrl="/" />
            </div>
          </SignedIn>
          
          <SignedOut>
            <div className="flex gap-3">
              <Link to="/sign-in" className="button-primary">
                Sign In
              </Link>
              <Link to="/sign-up" className="button-secondary">
                Sign Up
              </Link>
            </div>
          </SignedOut>
        </div>
        
        {/* Mobile navigation */}
        {isMenuOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-full left-0 right-0 glass-effect border-b border-white/5 md:hidden"
          >
            <div className="px-4 py-6 space-y-4">
              <SignedIn>
                <nav className="flex flex-col space-y-3">
                  <Link 
                    to="/dashboard" 
                    className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <div className="flex items-center gap-1.5">
                      <LayoutDashboard className="w-4 h-4" />
                      <span>Dashboard</span>
                    </div>
                  </Link>
                  <Link 
                    to="/spaces" 
                    className={`nav-link ${isActive('/spaces') ? 'active' : ''}`}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <div className="flex items-center gap-1.5">
                      <MessageSquare className="w-4 h-4" />
                      <span>Spaces</span>
                    </div>
                  </Link>
                  <Link 
                    to="/news" 
                    className={`nav-link ${isActive('/news') ? 'active' : ''}`}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <div className="flex items-center gap-1.5">
                      <LucideBook className="w-4 h-4" />
                      <span>News</span>
                    </div>
                  </Link>
                </nav>
                
                <div className="pt-4 mt-4 border-t border-white/10 flex justify-between items-center">
                  <span className="text-sm text-white/60">Manage account</span>
                  <UserButton afterSignOutUrl="/" />
                </div>
              </SignedIn>
              
              <SignedOut>
                <div className="flex flex-col gap-3">
                  <Link 
                    to="/sign-in" 
                    className="button-primary justify-center"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    Sign In
                  </Link>
                  <Link 
                    to="/sign-up" 
                    className="button-secondary justify-center"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    Sign Up
                  </Link>
                </div>
              </SignedOut>
            </div>
          </motion.div>
        )}
      </div>
    </header>
  );
};

export default Navbar;