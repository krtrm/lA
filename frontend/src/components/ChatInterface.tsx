import React, { useState, useRef, useEffect } from 'react';
import { useApi } from '../context/ApiContext';
import { StreamStep } from '../services/api';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'thinking';
  content: string;
  sources?: Array<{
    title: string;
    source: string;
    type: string;
  }>;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { streamQuery } = useApi();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add thinking message
    const thinkingId = Date.now().toString() + '-thinking';
    setMessages((prev) => [
      ...prev,
      { id: thinkingId, role: 'thinking', content: 'Thinking...' },
    ]);

    try {
      // Track current full response and sources
      let currentResponse = '';
      let sources: Message['sources'] = [];
      
      // Process streaming response
      await streamQuery(
        { query: userMessage.content, use_web: true },
        (step: StreamStep) => {
          if (step.type === 'complete') {
            // Final answer received
            currentResponse = step.content;
            if (step.details?.sources) {
              sources = step.details.sources;
            }
            
            // Remove thinking message and add assistant message
            setMessages((prev) => 
              prev.filter(msg => msg.id !== thinkingId).concat({
                id: Date.now().toString(),
                role: 'assistant',
                content: currentResponse,
                sources: sources
              })
            );
          } else if (step.type === 'error') {
            // Handle error
            setMessages((prev) => 
              prev.filter(msg => msg.id !== thinkingId).concat({
                id: Date.now().toString(),
                role: 'assistant',
                content: `Error: ${step.content}`
              })
            );
          } else if (step.type === 'thinking' || step.type === 'planning' || 
                     step.type === 'tool_use' || step.type === 'retrieval' || 
                     step.type === 'generation') {
            // Update thinking message
            setMessages((prev) => 
              prev.map(msg => 
                msg.id === thinkingId 
                  ? { ...msg, content: step.content } 
                  : msg
              )
            );
          }
        }
      );
    } catch (error) {
      console.error('Error querying API:', error);
      setMessages((prev) => 
        prev.filter(msg => msg.id !== thinkingId).concat({
          id: Date.now().toString(),
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request.'
        })
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages container */}
      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`chat-message ${
              message.role === 'user'
                ? 'chat-message-user'
                : message.role === 'thinking'
                ? 'chat-message-thinking'
                : 'chat-message-assistant'
            }`}
          >
            <div className="font-medium mb-1">
              {message.role === 'user' ? 'You' : message.role === 'thinking' ? 'Thinking...' : 'Vaqeel AI'}
            </div>
            <div>{message.content}</div>
            
            {/* Display sources if available */}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/10">
                <div className="text-sm text-muted-foreground mb-1">Sources:</div>
                <div className="space-y-1">
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="text-sm">
                      <a
                        href={source.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline"
                      >
                        {source.title || 'Source ' + (idx + 1)}
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="relative">
            <input
              type="text"
              value={input}
              onChange={handleInputChange}
              placeholder="Ask a legal question..."
              disabled={isLoading}
              className="input-field pr-16"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className={`absolute right-2 top-1/2 transform -translate-y-1/2 button-primary py-2 px-4 ${
                isLoading || !input.trim() ? 'button-disabled' : ''
              }`}
            >
              {isLoading ? (
                <span className="loading-spinner"></span>
              ) : (
                <span>Send</span>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
