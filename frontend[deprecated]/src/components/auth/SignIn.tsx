import React from 'react';
import { SignIn } from '@clerk/clerk-react';
import { motion } from 'framer-motion';
import { LucideBook } from 'lucide-react';

const SignInPage: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gradient-to-br from-gray-900 to-black">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-6 flex flex-col items-center"
      >
        <div className="flex items-center gap-2 mb-2">
          <LucideBook className="w-8 h-8 text-primary" />
          <h1 className="text-3xl font-bold gradient-text">Vaqeel</h1>
        </div>
        <p className="text-white/60">Your AI Legal Assistant</p>
      </motion.div>
      
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="glass-effect p-4 rounded-xl border border-white/10 w-full max-w-md mx-auto shadow-xl"
      >
        <SignIn
          appearance={{
            elements: {
              rootBox: 'mx-auto w-full flex items-center justify-center',
              card: 'bg-transparent border-0 w-full max-w-full',
              headerTitle: 'text-2xl text-white text-center',
              headerSubtitle: 'text-white/70 text-center',
              socialButtonsBlockButton: 'glass-effect border border-white/10 text-white hover:bg-white/10',
              formButtonPrimary: 'bg-primary hover:bg-primary/80 text-white',
              formFieldInput: 'bg-white/5 border border-white/10 text-white rounded-lg px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-primary/50',
              formFieldLabel: 'text-white/70 mb-1 block',
              formFieldErrorText: 'text-red-400 mt-1 text-sm',
              footerActionLink: 'text-primary hover:text-primary/80',
              identityPreviewEditButton: 'text-primary hover:text-primary/80',
              formFieldAction: 'text-primary hover:text-primary/80'
            },
            layout: {
              socialButtonsPlacement: 'top',
              showOptionalFields: true,
            }
          }}
          routing="path"
          path="/sign-in"
          redirectUrl="/dashboard"
        />
      </motion.div>
    </div>
  );
};

export default SignInPage;
