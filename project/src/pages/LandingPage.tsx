import React from 'react'
import { motion } from 'framer-motion'
import { Scale, Sparkles, Shield, Brain } from 'lucide-react'

const LandingPage = () => {
  return (
    <div className="relative min-h-screen">
      {/* Hero Section */}
      <div className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-7xl font-bold mb-8 gradient-text">
              Legal Intelligence,
              <br />
              Redefined
            </h1>
            <p className="text-xl text-gray-400 mb-12 max-w-3xl mx-auto">
              Harness the power of AI to transform your legal practice. Get instant insights, 
              streamline research, and make data-driven decisions with confidence.
            </p>
            <div className="flex justify-center space-x-4">
              <button className="px-8 py-4 rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors">
                Get Started
              </button>
              <button className="px-8 py-4 rounded-lg border border-primary/50 text-primary hover:bg-primary/10 transition-colors">
                Watch Demo
              </button>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 px-4 sm:px-6 lg:px-8 bg-secondary/20">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="p-6 rounded-lg glass-effect"
            >
              <Brain className="h-12 w-12 text-primary mb-4" />
              <h3 className="text-xl font-semibold mb-2">AI-Powered Research</h3>
              <p className="text-gray-400">
                Advanced machine learning algorithms analyze vast legal databases to provide 
                accurate, relevant insights in seconds.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="p-6 rounded-lg glass-effect"
            >
              <Shield className="h-12 w-12 text-primary mb-4" />
              <h3 className="text-xl font-semibold mb-2">Secure & Compliant</h3>
              <p className="text-gray-400">
                Enterprise-grade security ensures your data is protected while maintaining 
                regulatory compliance.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="p-6 rounded-lg glass-effect"
            >
              <Sparkles className="h-12 w-12 text-primary mb-4" />
              <h3 className="text-xl font-semibold mb-2">Smart Automation</h3>
              <p className="text-gray-400">
                Automate routine legal tasks and document analysis, saving time and 
                reducing human error.
              </p>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LandingPage