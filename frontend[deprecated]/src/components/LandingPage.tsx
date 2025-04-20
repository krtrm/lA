import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  LucideBook, 
  MessageSquare, 
  Scale, 
  FileText, 
  Search, 
  ArrowRight, 
  Check
} from 'lucide-react';
import { Button } from './ui/button';

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20">
          <div className="flex flex-col md:flex-row gap-8 items-center">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="flex-1"
            >
              <h1 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
                Your AI-Powered <span className="gradient-text">Legal Assistant</span> for Indian Law
              </h1>
              <p className="text-xl text-white/70 mb-8 leading-relaxed">
                Research, analyze, and draft legal documents with powerful AI tailored for Indian legal professionals.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button asChild className="px-8 py-6 h-auto text-lg font-medium">
                  <Link to="/sign-up">
                    Get Started Free
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Link>
                </Button>
                <Button asChild variant="outline" className="px-8 py-6 h-auto text-lg font-medium">
                  <Link to="/sign-in">
                    Sign In
                  </Link>
                </Button>
              </div>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="flex-1"
            >
              <div className="glass-effect p-6 rounded-xl border border-white/10">
                <div className="bg-white/5 rounded-lg p-4 mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full bg-red-400"></div>
                    <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
                    <div className="w-2 h-2 rounded-full bg-green-400"></div>
                    <div className="text-white/30 text-xs ml-2">Vaqeel Chat</div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex gap-3 items-start">
                      <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center flex-shrink-0">
                        <MessageSquare className="w-4 h-4 text-white/70" />
                      </div>
                      <div className="bg-white/5 rounded-lg p-3 text-white/80 text-sm">
                        What are the requirements for filing a PIL in India?
                      </div>
                    </div>
                    
                    <div className="flex gap-3 items-start">
                      <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                        <LucideBook className="w-4 h-4 text-blue-400" />
                      </div>
                      <div className="bg-white/5 rounded-lg p-3 text-white/80 text-sm">
                        <p className="mb-2">To file a Public Interest Litigation (PIL) in India, you need to meet these requirements:</p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>The matter must concern public interest</li>
                          <li>File in the High Court or Supreme Court</li>
                          <li>No formal requirements for locus standi</li>
                          <li>Simple letter can be treated as PIL</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
        
        {/* Background gradient */}
        <div className="absolute top-0 left-1/3 -translate-x-1/2 w-1/2 h-1/2 bg-blue-500/10 rounded-full blur-[120px] -z-10"></div>
        <div className="absolute bottom-0 right-1/3 translate-x-1/2 w-1/2 h-1/2 bg-purple-500/10 rounded-full blur-[120px] -z-10"></div>
      </section>
      
      {/* Features Section */}
      <section className="py-20 bg-black/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-bold mb-4">Legal AI Tools for Every Need</h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Vaqeel provides specialized AI tools to help with different aspects of legal work
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="glass-effect rounded-xl p-6 border border-white/10"
            >
              <Search className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold mb-3">Legal Research</h3>
              <p className="text-white/70 mb-4">
                Search and analyze case law, statutes, and legal commentary from multiple sources.
              </p>
              <ul className="space-y-2">
                {['Quick insights from vast legal databases', 'Case law recommendations', 'Statute interpretations'].map((item, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-white/80">{item}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="glass-effect rounded-xl p-6 border border-white/10"
            >
              <FileText className="w-12 h-12 text-green-400 mb-4" />
              <h3 className="text-xl font-semibold mb-3">Document Drafting</h3>
              <p className="text-white/70 mb-4">
                Create outlines and drafts for legal documents, contracts, and agreements.
              </p>
              <ul className="space-y-2">
                {['Contract templates and outlines', 'Jurisdiction-specific clauses', 'Customizable to your needs'].map((item, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-white/80">{item}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="glass-effect rounded-xl p-6 border border-white/10"
            >
              <Scale className="w-12 h-12 text-purple-400 mb-4" />
              <h3 className="text-xl font-semibold mb-3">Legal Analysis</h3>
              <p className="text-white/70 mb-4">
                Analyze complex legal scenarios and generate arguments for your case.
              </p>
              <ul className="space-y-2">
                {['Case strength evaluation', 'Argument construction', 'Precedent-based analysis'].map((item, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-white/80">{item}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="glass-effect rounded-xl p-8 md:p-12 border border-white/10 text-center"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Transform Your Legal Practice?</h2>
            <p className="text-xl text-white/70 mb-8 max-w-3xl mx-auto">
              Join thousands of legal professionals using Vaqeel to streamline research, drafting, and analysis.
            </p>
            <Button asChild className="px-8 py-6 h-auto text-lg">
              <Link to="/sign-up">
                Get Started Free <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
            </Button>
          </motion.div>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <LucideBook className="w-6 h-6 text-primary" />
              <span className="text-xl font-bold gradient-text">Vaqeel</span>
            </div>
            <div className="text-white/50 text-sm">
              © {new Date().getFullYear()} Vaqeel. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
