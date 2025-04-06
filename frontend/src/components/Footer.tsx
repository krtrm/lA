import React from 'react';
import { Link } from 'react-router-dom';
import { LucideBook, Github, Twitter } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="border-t border-white/10 bg-black/30 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <LucideBook className="w-6 h-6 text-primary" />
              <span className="text-xl font-bold gradient-text">Vaqeel</span>
            </div>
            <p className="text-white/60 text-sm">
              AI-powered legal assistance for Indian law professionals
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-white/40 hover:text-white/80">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-white/40 hover:text-white/80">
                <Twitter className="w-5 h-5" />
              </a>
            </div>
          </div>
          
          <div>
            <h3 className="text-white font-medium mb-4">Platform</h3>
            <ul className="space-y-3 text-white/60">
              <li><Link to="/features" className="hover:text-white">Features</Link></li>
              <li><Link to="/pricing" className="hover:text-white">Pricing</Link></li>
              <li><Link to="/about" className="hover:text-white">About</Link></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-white font-medium mb-4">Resources</h3>
            <ul className="space-y-3 text-white/60">
              <li><Link to="/news" className="hover:text-white">Legal News</Link></li>
              <li><Link to="/contact" className="hover:text-white">Contact</Link></li>
              <li><a href="#" className="hover:text-white">API</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-white font-medium mb-4">Legal</h3>
            <ul className="space-y-3 text-white/60">
              <li><a href="#" className="hover:text-white">Terms of Service</a></li>
              <li><a href="#" className="hover:text-white">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-white">Disclaimer</a></li>
            </ul>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-white/10 text-center text-white/40 text-sm">
          © {new Date().getFullYear()} Vaqeel. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
