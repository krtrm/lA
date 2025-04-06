import React from 'react';
import { SignIn } from '@clerk/clerk-react';
import { motion } from 'framer-motion';
import { LucideBook } from 'lucide-react';

const SignInPage: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
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
        className="glass-effect p-1 rounded-xl border border-white/10 w-full max-w-md"
      >
        <SignIn
          appearance={{
            elements: {
              formButtonPrimary: 'button-primary',
              card: 'bg-transparent border-0',
              headerTitle: 'text-2xl text-white',
              headerSubtitle: 'text-white/70',
              socialButtonsBlockButton: 'glass-effect border border-white/10',
              formFieldInput: 'input-field',
              formFieldLabel: 'text-white/70',
              formFieldErrorText: 'text-red-400',
              footerActionLink: 'text-primary hover:text-primary/80'
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
