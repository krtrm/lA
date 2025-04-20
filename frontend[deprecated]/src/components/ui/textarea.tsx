import React, { TextareaHTMLAttributes, forwardRef } from 'react';
import { cn } from '../../utils/cn';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  wrapperClassName?: string;
  error?: string;
}

const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, wrapperClassName, error, ...props }, ref) => {
    return (
      <div className={cn('space-y-2', wrapperClassName)}>
        <textarea
          className={cn(
            'w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg',
            'text-white placeholder:text-white/40',
            'focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-transparent',
            'transition-colors duration-200 resize-y min-h-[100px]',
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

Textarea.displayName = 'Textarea';

export { Textarea };