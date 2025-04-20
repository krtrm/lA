import React, { useState, useEffect, useRef } from 'react';
import { useApi } from '../context/ApiContext';
import { MessageSquare, Send, User, Bot, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { readStreamChunks } from '../utils/streamingApi';

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => Promise<void>;
  isTyping?: boolean;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isTyping = false,
  placeholder = 'Type your message...',
  disabled = false,
  className = '',
}) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Focus input when component mounts
  useEffect(() => {
    if (inputRef.current && !disabled) {
      inputRef.current.focus();
    }
  }, [disabled]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || disabled) return;
    
    const message = inputValue;
    setInputValue('');
    
    try {
      await onSendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Messages container */}
      <div className="flex-1 overflow-y-auto pb-4">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={`chat-message ${
                message.role === 'user' ? 'chat-message-user' : 'chat-message-assistant'
              }`}
            >
              <div className="max-w-3xl mx-auto">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                    {message.role === 'user' ? (
                      <User className="w-5 h-5 text-white/70" />
                    ) : message.isStreaming ? (
                      <Loader2 className="w-5 h-5 text-white/70 animate-spin" />
                    ) : (
                      <Bot className="w-5 h-5 text-white/70" />
                    )}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white/70">
                        {message.role === 'user' ? 'You' : 'Vaqeel'}
                      </span>
                      <span className="text-xs text-white/30">{message.timestamp}</span>
                    </div>
                    <div className="text-white/90 leading-relaxed text-[15px] whitespace-pre-line">
                      {message.isStreaming ? (
                        <>
                          {message.content || ''}
                          <span className="inline-block w-2 h-4 bg-primary/40 ml-1 animate-pulse"></span>
                        </>
                      ) : (
                        message.content
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="chat-message chat-message-assistant"
            >
              <div className="max-w-3xl mx-auto">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                    <Loader2 className="w-5 h-5 text-white/70 animate-spin" />
                  </div>
                  <div className="flex-1">
                    <div className="text-white/50">
                      Vaqeel is thinking...
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Invisible element for scrolling to bottom */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input form */}
      <div className="bg-black/30 backdrop-blur-md border-t border-white/5 p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={disabled}
              placeholder={disabled ? 'Please wait...' : placeholder}
              className="input-field pr-12"
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={disabled || !inputValue.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-lg bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </motion.button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
