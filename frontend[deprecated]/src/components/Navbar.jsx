import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <nav className="fixed top-0 w-full bg-gray-900/90 backdrop-blur-lg border-b border-white/10 z-50">
      <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="text-xl font-bold text-blue-400">
          Vaqeel
        </Link>
        
        {/* Mobile menu button */}
        <button 
          className="md:hidden p-2 text-white"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? 'Close' : 'Menu'}
        </button>
        
        {/* Desktop navigation */}
        <div className="hidden md:flex items-center space-x-6">
          <SignedIn>
            <div className="flex items-center space-x-6">
              <Link to="/dashboard" className="text-white/70 hover:text-white">Dashboard</Link>
              <Link to="/spaces" className="text-white/70 hover:text-white">Spaces</Link>
              <Link to="/news" className="text-white/70 hover:text-white">News</Link>
              <UserButton afterSignOutUrl="/" />
            </div>
          </SignedIn>
          
          <SignedOut>
            <div className="flex space-x-4">
              <Link to="/sign-in" className="px-4 py-2 bg-blue-600 text-white rounded">Sign In</Link>
              <Link to="/sign-up" className="px-4 py-2 border border-blue-600 text-blue-400 rounded">Sign Up</Link>
            </div>
          </SignedOut>
        </div>
        
        {/* Mobile menu */}
        {isOpen && (
          <div className="md:hidden absolute top-full left-0 right-0 bg-gray-900 border-b border-white/10 py-4 px-4">
            <div className="flex flex-col space-y-4">
              <SignedIn>
                <Link to="/dashboard" className="text-white/70 hover:text-white" onClick={() => setIsOpen(false)}>
                  Dashboard
                </Link>
                <Link to="/spaces" className="text-white/70 hover:text-white" onClick={() => setIsOpen(false)}>
                  Spaces
                </Link>
                <Link to="/news" className="text-white/70 hover:text-white" onClick={() => setIsOpen(false)}>
                  News
                </Link>
                <div className="pt-4 border-t border-white/10">
                  <UserButton afterSignOutUrl="/" />
                </div>
              </SignedIn>
              
              <SignedOut>
                <Link to="/sign-in" className="px-4 py-2 bg-blue-600 text-white rounded block text-center" onClick={() => setIsOpen(false)}>
                  Sign In
                </Link>
                <Link to="/sign-up" className="px-4 py-2 border border-blue-600 text-blue-400 rounded block text-center" onClick={() => setIsOpen(false)}>
                  Sign Up
                </Link>
              </SignedOut>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
