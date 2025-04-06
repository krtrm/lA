import React, { useState } from 'react'
import { Lock, Github, ToggleLeft as Google } from 'lucide-react'
import { motion } from 'framer-motion'

export default function LoginPage() {
  const [isSignUp, setIsSignUp] = useState(false)

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center bg-gradient-to-br from-gray-900 to-black py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full space-y-8 backdrop-blur-xl bg-white/5 p-8 rounded-2xl border border-white/10"
      >
        <div>
          <motion.div 
            className="mx-auto h-12 w-12 flex items-center justify-center rounded-full bg-blue-500/20"
            whileHover={{ scale: 1.05 }}
          >
            <Lock className="h-6 w-6 text-blue-500" />
          </motion.div>
          <h2 className="mt-6 text-center text-3xl font-bold text-white">
            {isSignUp ? 'Create your account' : 'Welcome back'}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-400">
            {isSignUp ? 'Start your legal AI journey' : 'Continue your legal AI journey'}
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <button className="group relative w-full flex justify-center py-3 px-4 border border-white/10 rounded-xl text-sm font-medium text-white hover:bg-white/5 transition-all duration-200 backdrop-blur-xl">
            <Google className="w-5 h-5 mr-2" /> Continue with Google
          </button>
          <button className="group relative w-full flex justify-center py-3 px-4 border border-white/10 rounded-xl text-sm font-medium text-white hover:bg-white/5 transition-all duration-200 backdrop-blur-xl">
            <Github className="w-5 h-5 mr-2" /> Continue with Github
          </button>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-white/10"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-[#0a0a0a] text-gray-400">Or continue with</span>
          </div>
        </div>

        <form className="mt-8 space-y-6" onSubmit={(e) => e.preventDefault()}>
          <div className="rounded-md space-y-4">
            <div>
              <label htmlFor="email-address" className="sr-only">Email address</label>
              <input
                id="email-address"
                name="email"
                type="email"
                autoComplete="email"
                required
                className="appearance-none rounded-xl relative block w-full px-4 py-3 border border-white/10 placeholder-gray-400 text-white bg-white/5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="Email address"
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                className="appearance-none rounded-xl relative block w-full px-4 py-3 border border-white/10 placeholder-gray-400 text-white bg-white/5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="Password"
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className="h-4 w-4 text-blue-500 focus:ring-blue-500 border-white/10 rounded bg-white/5"
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-400">
                Remember me
              </label>
            </div>

            <div className="text-sm">
              <a href="#" className="font-medium text-blue-500 hover:text-blue-400">
                Forgot password?
              </a>
            </div>
          </div>

          <div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-xl text-white bg-blue-500 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
            >
              {isSignUp ? 'Sign up' : 'Sign in'}
            </motion.button>
          </div>
        </form>

        <div className="text-center">
          <button 
            onClick={() => setIsSignUp(!isSignUp)} 
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
          </button>
        </div>
      </motion.div>
    </div>
  )
}