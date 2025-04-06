import React from 'react'
import { Link } from 'react-router-dom'
import { Scale, BookOpen, MessageSquare, CreditCard } from 'lucide-react'

const Navbar = () => {
  return (
    <nav className="fixed top-0 w-full z-50 glass-effect">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <Scale className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold gradient-text">LegalAI</span>
            </Link>
          </div>
          
          <div className="hidden md:block">
            <div className="flex items-center space-x-8">
              <Link to="/news" className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                <BookOpen className="h-5 w-5" />
                <span>News</span>
              </Link>
              <Link to="/spaces" className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                <MessageSquare className="h-5 w-5" />
                <span>Spaces</span>
              </Link>
              <Link to="/payments" className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                <CreditCard className="h-5 w-5" />
                <span>Payments</span>
              </Link>
              <Link 
                to="/login" 
                className="px-4 py-2 rounded-md bg-primary text-white hover:bg-primary/90 transition-colors"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar