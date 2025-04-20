import React, { InputHTMLAttributes, forwardRef } from 'react';
import { cn } from '../../utils/cn';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  wrapperClassName?: string;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, wrapperClassName, error, type = 'text', ...props }, ref) => {
    return (
      <div className={cn('space-y-2', wrapperClassName)}>
        <input
          type={type}
          className={cn(
            'w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg',
            'text-white placeholder:text-white/40',
            'focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-transparent',
            'transition-colors duration-200',
            error && 'border-red-500 focus:ring-red-500/50',
            className
          )}
          ref={ref}
          {...props}
        />
        {error && (
          <p className="text-red-500 text-sm">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export { Input };