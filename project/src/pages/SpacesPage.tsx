import React, { useState } from 'react'
import { MessageSquare, Plus, Search, Send, User, Bot, ArrowLeft } from 'lucide-react'
import { motion } from 'framer-motion'

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

interface Space {
  id: number
  title: string
  messages: Message[]
}

export default function SpacesPage() {
  const [selectedSpace, setSelectedSpace] = useState<Space | null>(null)
  const [inputMessage, setInputMessage] = useState('')

  const spaces: Space[] = [
    {
      id: 1,
      title: 'Contract Review',
      messages: [
        {
          id: 1,
          role: 'user',
          content: 'Can you review this NDA agreement?',
          timestamp: '2:30 PM'
        },
        {
          id: 2,
          role: 'assistant',
          content: 'I\'d be happy to review the NDA agreement. Please share the document or paste the content you\'d like me to analyze.',
          timestamp: '2:31 PM'
        }
      ]
    },
    {
      id: 2,
      title: 'Legal Research',
      messages: [
        {
          id: 1,
          role: 'user',
          content: 'What are the recent changes in intellectual property law?',
          timestamp: '3:45 PM'
        },
        {
          id: 2,
          role: 'assistant',
          content: 'There have been several significant developments in intellectual property law recently. Let me outline the key changes...',
          timestamp: '3:46 PM'
        }
      ]
    }
  ]

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputMessage.trim() || !selectedSpace) return

    const newMessage: Message = {
      id: selectedSpace.messages.length + 1,
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    setSelectedSpace({
      ...selectedSpace,
      messages: [...selectedSpace.messages, newMessage]
    })
    setInputMessage('')
  }

  if (selectedSpace) {
    return (
      <div className="min-h-screen bg-black">
        {/* Chat Header */}
        <div className="fixed top-16 left-0 right-0 z-10 glass-effect border-b border-white/5">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedSpace(null)}
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-white/70" />
              </motion.button>
              <div>
                <h2 className="text-lg font-semibold text-white">{selectedSpace.title}</h2>
                <p className="text-sm text-white/50">Active conversation</p>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="pt-32 pb-36">
          {selectedSpace.messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`chat-message ${
                message.role === 'user' ? 'chat-message-user' : 'chat-message-assistant'
              }`}
            >
              <div className="max-w-3xl mx-auto">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                    {message.role === 'user' ? (
                      <User className="w-5 h-5 text-white/70" />
                    ) : (
                      <Bot className="w-5 h-5 text-white/70" />
                    )}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white/70">
                        {message.role === 'user' ? 'You' : 'Assistant'}
                      </span>
                      <span className="text-xs text-white/30">{message.timestamp}</span>
                    </div>
                    <p className="text-white/90 leading-relaxed text-[15px]">{message.content}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Chat Input */}
        <div className="chat-input-container">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Type your message..."
                className="input-field"
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                type="submit"
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-lg bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/90 transition-colors"
              >
                <Send className="w-4 h-4" />
              </motion.button>
            </form>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-black">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-semibold text-white mb-1">Spaces</h1>
            <p className="text-white/50">Your conversations with LegalAI</p>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="button-primary"
          >
            <Plus className="w-5 h-5" /> New Space
          </motion.button>
        </div>

        <div className="relative mb-8">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/30 w-5 h-5" />
          <input
            type="text"
            placeholder="Search spaces..."
            className="input-field pl-10"
          />
        </div>

        <div className="grid gap-4">
          {spaces.map((space) => (
            <motion.div
              key={space.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.01 }}
              onClick={() => setSelectedSpace(space)}
              className="space-card"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors">
                    <MessageSquare className="w-5 h-5 text-white/70" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-white mb-1 group-hover:text-white/90 transition-colors">
                      {space.title}
                    </h3>
                    <p className="text-sm text-white/50 line-clamp-1">
                      {space.messages[space.messages.length - 1].content}
                    </p>
                  </div>
                </div>
                <span className="text-xs font-medium text-white/30 tabular-nums">
                  {space.messages[space.messages.length - 1].timestamp}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}