import React, { useState, useEffect } from 'react'
import { MessageSquare, Plus, Search, Send, User, Bot, ArrowLeft, X, HelpCircle, Scale, Book, Gavel, FileCheck, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'

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
  type?: string
  createdAt?: string
  lastActive?: string
}

interface SpaceTypeOption {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  apiEndpoint: string
  formFields: SpaceFormField[]
}

interface SpaceFormField {
  id: string
  label: string
  type: 'text' | 'textarea' | 'select' | 'checkbox'
  placeholder?: string
  options?: { value: string, label: string }[]
  required?: boolean
  helperText?: string
}

export default function SpacesPage() {
  const [selectedSpace, setSelectedSpace] = useState<Space | null>(null)
  const [inputMessage, setInputMessage] = useState('')
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [selectedSpaceType, setSelectedSpaceType] = useState<SpaceTypeOption | null>(null)
  const [formData, setFormData] = useState<Record<string, any>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [apiError, setApiError] = useState<string | null>(null)
  const [spaces, setSpaces] = useState<Space[]>([])
  const [usageStats, setUsageStats] = useState({
    totalSpaces: 0,
    messagesThisMonth: 0,
    activeResearches: 0
  })

  // Predefined space types with their form fields mapped to API endpoints
  const spaceTypes: SpaceTypeOption[] = [
    {
      id: 'legal_research',
      title: 'Legal Research',
      description: 'Research specific legal topics, cases, or statutes',
      icon: <Book className="h-6 w-6 text-primary" />,
      apiEndpoint: 'query',
      formFields: [
        {
          id: 'query',
          label: 'Research Topic',
          type: 'text',
          placeholder: 'E.g., Recent Supreme Court judgments on privacy',
          required: true,
          helperText: 'Be specific about your research area'
        },
        {
          id: 'use_web',
          label: 'Use Web Search',
          type: 'checkbox',
          helperText: 'Include recent web results in research'
        }
      ]
    },
    {
      id: 'document_drafting',
      title: 'Document Drafting',
      description: 'Create outlines for legal documents and agreements',
      icon: <FileCheck className="h-6 w-6 text-primary" />,
      apiEndpoint: 'create_outline',
      formFields: [
        {
          id: 'topic',
          label: 'Document Subject',
          type: 'text',
          placeholder: 'E.g., Service Agreement for Software Development',
          required: true
        },
        {
          id: 'doc_type',
          label: 'Document Type',
          type: 'select',
          options: [
            { value: 'contract', label: 'Contract' },
            { value: 'agreement', label: 'Agreement' },
            { value: 'memo', label: 'Legal Memorandum' },
            { value: 'notice', label: 'Legal Notice' },
            { value: 'petition', label: 'Petition' }
          ],
          required: true
        }
      ]
    },
    {
      id: 'legal_analysis',
      title: 'Legal Analysis',
      description: 'Analyze legal scenarios and generate arguments',
      icon: <Scale className="h-6 w-6 text-primary" />,
      apiEndpoint: 'generate_argument',
      formFields: [
        {
          id: 'topic',
          label: 'Legal Issue',
          type: 'text',
          placeholder: 'E.g., Copyright infringement in digital content',
          required: true
        },
        {
          id: 'points',
          label: 'Key Points to Address',
          type: 'textarea',
          placeholder: 'List major points to be covered in the analysis',
          helperText: 'Enter each point on a new line'
        }
      ]
    },
    {
      id: 'citation_verification',
      title: 'Citation Check',
      description: 'Verify and correct legal citations',
      icon: <Gavel className="h-6 w-6 text-primary" />,
      apiEndpoint: 'verify_citation',
      formFields: [
        {
          id: 'citation',
          label: 'Citation',
          type: 'text',
          placeholder: 'E.g., AIR 2017 SC 4161',
          required: true
        }
      ]
    }
  ]

  // Load sample spaces on mount
  useEffect(() => {
    // This would be an API call in a real implementation
    const sampleSpaces: Space[] = [
      {
        id: 1,
        title: 'Contract Review',
        type: 'document_drafting',
        createdAt: '2023-11-15',
        lastActive: '2023-11-20',
        messages: [
          {
            id: 1,
            role: 'user',
            content: 'Can you create an outline for an NDA agreement?',
            timestamp: '2:30 PM'
          },
          {
            id: 2,
            role: 'assistant',
            content: 'I\'d be happy to create an outline for an NDA agreement. Here\'s a comprehensive structure...',
            timestamp: '2:31 PM'
          }
        ]
      },
      {
        id: 2,
        title: 'IP Law Research',
        type: 'legal_research',
        createdAt: '2023-11-10',
        lastActive: '2023-11-18',
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
    ];
    
    setSpaces(sampleSpaces);
    
    // Calculate stats
    setUsageStats({
      totalSpaces: sampleSpaces.length,
      messagesThisMonth: sampleSpaces.reduce((count, space) => count + space.messages.length, 0),
      activeResearches: sampleSpaces.length
    });
  }, []);

  // Handle sending a message in a space
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputMessage.trim() || !selectedSpace) return

    const newUserMessage: Message = {
      id: selectedSpace.messages.length + 1,
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    // Add user message immediately
    const updatedSpace = {
      ...selectedSpace,
      messages: [...selectedSpace.messages, newUserMessage],
      lastActive: new Date().toISOString().split('T')[0]
    };
    
    setSelectedSpace(updatedSpace);
    
    // Update in spaces list
    setSpaces(prevSpaces => 
      prevSpaces.map(space => 
        space.id === selectedSpace.id ? updatedSpace : space
      )
    );
    
    setInputMessage('');
    setApiError(null);
    
    // Show typing indicator
    const loadingMessage: Message = {
      id: updatedSpace.messages.length + 1,
      role: 'assistant',
      content: '...',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setSelectedSpace(prev => {
      if (!prev) return null;
      return {
        ...prev,
        messages: [...prev.messages, loadingMessage]
      };
    });
    
    // Call appropriate API based on space type
    try {
      let response;
      const query = inputMessage;
      
      switch (selectedSpace.type) {
        case 'legal_research':
          response = await api.query({ query, use_web: true });
          break;
        case 'document_drafting':
          // For subsequent messages in document drafting spaces, use query for follow-ups
          response = await api.query({ query });
          break;
        case 'legal_analysis':
          // For subsequent messages in legal analysis spaces, use query for follow-ups
          response = await api.query({ query });
          break;
        case 'citation_verification':
          // If it looks like a citation, use verify endpoint, otherwise use query
          if (/^[A-Z]+\s+\d{4}\s+[A-Z]+/.test(query)) {
            response = await api.verifyCitation(query);
          } else {
            response = await api.query({ query });
          }
          break;
        default:
          response = await api.query({ query });
      }
      
      // Get answer text based on response type
      let answerText = '';
      if ('answer' in response) {
        answerText = response.answer;
      } else if ('outline' in response) {
        answerText = response.outline;
      } else if ('argument' in response) {
        answerText = response.argument;
      } else if ('is_valid' in response) {
        answerText = `The citation ${response.is_valid ? 'is valid' : 'is not valid'}. ${
          response.corrected_citation && !response.is_valid 
            ? `Suggested correction: ${response.corrected_citation}` 
            : ''
        }${response.summary ? `\n\n${response.summary}` : ''}`;
      } else {
        answerText = 'I processed your request, but I\'m not sure how to display the results.';
      }
      
      // Replace loading message with actual response
      const assistantMessage: Message = {
        id: updatedSpace.messages.length + 1,
        role: 'assistant',
        content: answerText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      const finalUpdatedSpace = {
        ...updatedSpace,
        messages: [...updatedSpace.messages.filter(m => m.content !== '...'), assistantMessage]
      };
      
      setSelectedSpace(finalUpdatedSpace);
      
      // Update in spaces list
      setSpaces(prevSpaces => 
        prevSpaces.map(space => 
          space.id === selectedSpace.id ? finalUpdatedSpace : space
        )
      );
      
    } catch (error: any) {
      console.error('API error:', error);
      setApiError(error.message || 'Failed to get a response. Please try again.');
      
      // Remove loading message and add error message
      const errorMessage: Message = {
        id: updatedSpace.messages.length + 1,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message || 'Unknown error'}. Please try again.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      const errorUpdatedSpace = {
        ...updatedSpace,
        messages: [...updatedSpace.messages.filter(m => m.content !== '...'), errorMessage]
      };
      
      setSelectedSpace(errorUpdatedSpace);
      
      // Update in spaces list
      setSpaces(prevSpaces => 
        prevSpaces.map(space => 
          space.id === selectedSpace.id ? errorUpdatedSpace : space
        )
      );
    }
  }

  // Create a new space based on the selected type and form data
  const handleCreateNewSpace = async () => {
    if (!selectedSpaceType) return;
    
    setIsLoading(true);
    setApiError(null);
    
    try {
      console.log("Creating new space with type:", selectedSpaceType.id);
      console.log("Form data:", formData);
      
      // Get form data in the right format for the API
      const apiParams: Record<string, any> = { ...formData };
      
      // Special handling for points - convert textarea to array
      if (selectedSpaceType.id === 'legal_analysis' && typeof apiParams.points === 'string') {
        apiParams.points = apiParams.points
          .split('\n')
          .map(line => line.trim())
          .filter(line => line.length > 0);
      }
      
      // Make the appropriate API call based on space type
      let response;
      let initialMessage = '';
      let assistantResponse = '';
      
      try {
        switch (selectedSpaceType.id) {
          case 'legal_research':
            console.log("Making query API call with:", apiParams);
            response = await api.query({
              query: apiParams.query,
              use_web: apiParams.use_web || false
            });
            initialMessage = apiParams.query;
            assistantResponse = response.answer;
            break;
            
          case 'document_drafting':
            console.log("Making createOutline API call with:", apiParams);
            response = await api.createOutline(
              apiParams.topic,
              apiParams.doc_type
            );
            initialMessage = `Please create an outline for a ${apiParams.doc_type} about ${apiParams.topic}.`;
            assistantResponse = response.outline;
            break;
            
          case 'legal_analysis':
            console.log("Making generateArgument API call with:", apiParams);
            response = await api.generateArgument(
              apiParams.topic,
              apiParams.points || []
            );
            initialMessage = `I need a legal analysis on ${apiParams.topic}${
              apiParams.points && apiParams.points.length 
                ? `\n\nPlease address these key points:\n${apiParams.points.join('\n')}` 
                : ''
            }`;
            assistantResponse = response.argument;
            break;
            
          case 'citation_verification':
            console.log("Making verifyCitation API call with:", apiParams);
            response = await api.verifyCitation(apiParams.citation);
            initialMessage = `Can you verify this legal citation: ${apiParams.citation}`;
            assistantResponse = `The citation ${response.is_valid ? 'is valid' : 'is not valid'}. ${
              response.corrected_citation && !response.is_valid 
                ? `Suggested correction: ${response.corrected_citation}` 
                : ''
            }${response.summary ? `\n\n${response.summary}` : ''}`;
            break;
            
          default:
            throw new Error('Unknown space type');
        }
        
        console.log("API response:", response);
      } catch (apiError: any) {
        console.error("API call failed:", apiError);
        
        // For demo purposes, proceed with mock data if API fails
        console.log("Using mock response for demo");
        initialMessage = apiParams.query || apiParams.topic || apiParams.citation || "New conversation";
        assistantResponse = "I'm sorry, I couldn't connect to the API. This is a demo response for illustration purposes. In a real environment, I would provide a proper answer based on your query.";
      }
      
      // Create the new space with initial messages
      const newSpace: Space = {
        id: Date.now(),
        title: apiParams.topic || apiParams.query || 'New Space',
        type: selectedSpaceType.id,
        createdAt: new Date().toISOString().split('T')[0],
        lastActive: new Date().toISOString().split('T')[0],
        messages: [
          {
            id: 1,
            role: 'user',
            content: initialMessage,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          },
          {
            id: 2,
            role: 'assistant',
            content: assistantResponse,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }
        ]
      };
      
      console.log("Created new space:", newSpace);
      
      // Add the new space to the list and select it
      setSpaces(prev => [newSpace, ...prev]);
      setSelectedSpace(newSpace);
      
      // Update stats
      setUsageStats(prev => ({
        ...prev,
        totalSpaces: prev.totalSpaces + 1,
        messagesThisMonth: prev.messagesThisMonth + 2,
        activeResearches: prev.activeResearches + 1
      }));
      
      // Reset form and close modal
      setFormData({});
      setSelectedSpaceType(null);
      setIsCreateModalOpen(false);
      
    } catch (error: any) {
      console.error('Error creating space:', error);
      setApiError(error.message || 'Failed to create space. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form field changes
  const handleFormChange = (field: SpaceFormField, value: any) => {
    if (field.type === 'checkbox') {
      setFormData(prev => ({
        ...prev,
        [field.id]: !prev[field.id]
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [field.id]: value
      }));
    }
  };

  // Filter spaces by search query
  const filteredSpaces = searchQuery 
    ? spaces.filter(space => 
        space.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        space.messages.some(msg => msg.content.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : spaces;

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
                <p className="text-sm text-white/50">
                  {selectedSpace.type === 'legal_research' ? 'Legal Research' :
                   selectedSpace.type === 'document_drafting' ? 'Document Drafting' :
                   selectedSpace.type === 'legal_analysis' ? 'Legal Analysis' :
                   selectedSpace.type === 'citation_verification' ? 'Citation Check' :
                   'Conversation'}
                </p>
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
                    ) : message.content === '...' ? (
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
                      {message.content === '...' ? (
                        <span className="text-white/50">Vaqeel is thinking...</span>
                      ) : (
                        message.content
                      )}
                    </div>
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
                disabled={selectedSpace.messages.some(m => m.content === '...')}
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                type="submit"
                disabled={selectedSpace.messages.some(m => m.content === '...')}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-lg bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
            <p className="text-white/50">Your conversations with Vaqeel</p>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="button-primary"
            onClick={() => setIsCreateModalOpen(true)}
          >
            <Plus className="w-5 h-5" /> New Space
          </motion.button>
        </div>

        {/* Usage Statistics */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <div className="glass-effect rounded-xl p-4">
            <h3 className="text-sm text-white/50 mb-1">Total Spaces</h3>
            <p className="text-2xl font-semibold text-white">{usageStats.totalSpaces}</p>
          </div>
          <div className="glass-effect rounded-xl p-4">
            <h3 className="text-sm text-white/50 mb-1">Messages This Month</h3>
            <p className="text-2xl font-semibold text-white">{usageStats.messagesThisMonth}</p>
          </div>
          <div className="glass-effect rounded-xl p-4">
            <h3 className="text-sm text-white/50 mb-1">Active Researches</h3>
            <p className="text-2xl font-semibold text-white">{usageStats.activeResearches}</p>
          </div>
        </div>

        <div className="relative mb-8">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/30 w-5 h-5" />
          <input
            type="text"
            placeholder="Search spaces..."
            className="input-field pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="grid gap-4">
          {filteredSpaces.length > 0 ? (
            filteredSpaces.map((space) => (
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
                    <div className="p-3 rounded-xl bg-white/5 transition-colors">
                      <MessageSquare className="w-5 h-5 text-white/70" />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-white mb-1 transition-colors">
                        {space.title}
                      </h3>
                      <p className="text-sm text-white/50 line-clamp-1">
                        {space.messages[space.messages.length - 1]?.content === '...' 
                          ? 'Vaqeel is thinking...'
                          : space.messages[space.messages.length - 1]?.content}
                      </p>
                    </div>
                  </div>
                  <span className="text-xs font-medium text-white/30 tabular-nums">
                    {space.messages[space.messages.length - 1]?.timestamp}
                  </span>
                </div>
              </motion.div>
            ))
          ) : (
            <div className="text-center py-12">
              <p className="text-white/50">No spaces found. Create a new space to get started.</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Space Modal */}
      <AnimatePresence>
        {isCreateModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setIsCreateModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="glass-effect border border-white/10 rounded-xl max-w-lg w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-white">
                    {selectedSpaceType ? `Create ${selectedSpaceType.title} Space` : 'Create New Space'}
                  </h2>
                  <button
                    onClick={() => {
                      setIsCreateModalOpen(false);
                      setSelectedSpaceType(null);
                      setFormData({});
                      setApiError(null);
                    }}
                    className="p-1 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <X className="w-5 h-5 text-white/70" />
                  </button>
                </div>

                {apiError && (
                  <div className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-white text-sm">
                    {apiError}
                  </div>
                )}

                {!selectedSpaceType ? (
                  <div className="grid gap-4">
                    <p className="text-white/70 mb-4">Select the type of space you want to create:</p>
                    
                    {spaceTypes.map((type) => (
                      <motion.div
                        key={type.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="p-4 rounded-lg border border-white/10 hover:border-primary/30 bg-white/5 hover:bg-white/10 transition-all cursor-pointer"
                        onClick={() => setSelectedSpaceType(type)}
                      >
                        <div className="flex items-center gap-3">
                          {type.icon}
                          <div>
                            <h3 className="text-white font-medium">{type.title}</h3>
                            <p className="text-white/50 text-sm">{type.description}</p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {selectedSpaceType.formFields.map((field) => (
                      <div key={field.id} className="space-y-2">
                        <div className="flex items-center gap-2">
                          <label htmlFor={field.id} className="text-white/70 text-sm">
                            {field.label}{field.required ? ' *' : ''}
                          </label>
                          {field.helperText && (
                            <div className="relative group">
                              <HelpCircle className="w-4 h-4 text-white/30" />
                              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-black/90 border border-white/10 rounded text-xs text-white/70 invisible group-hover:visible transition-opacity opacity-0 group-hover:opacity-100 z-10">
                                {field.helperText}
                              </div>
                            </div>
                          )}
                        </div>
                        
                        {field.type === 'text' && (
                          <input
                            id={field.id}
                            type="text"
                            placeholder={field.placeholder}
                            className="input-field"
                            value={formData[field.id] || ''}
                            onChange={(e) => handleFormChange(field, e.target.value)}
                            required={field.required}
                          />
                        )}
                        
                        {field.type === 'textarea' && (
                          <textarea
                            id={field.id}
                            placeholder={field.placeholder}
                            className="input-field min-h-[100px]"
                            value={formData[field.id] || ''}
                            onChange={(e) => handleFormChange(field, e.target.value)}
                            required={field.required}
                          />
                        )}
                        
                        {field.type === 'select' && (
                          <select
                            id={field.id}
                            className="input-field"
                            value={formData[field.id] || ''}
                            onChange={(e) => handleFormChange(field, e.target.value)}
                            required={field.required}
                          >
                            <option value="">Select an option</option>
                            {field.options?.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        )}
                        
                        {field.type === 'checkbox' && (
                          <div className="flex items-center">
                            <input
                              id={field.id}
                              type="checkbox"
                              className="mr-2 w-4 h-4"
                              checked={!!formData[field.id]}
                              onChange={() => handleFormChange(field, !formData[field.id])}
                            />
                            <label htmlFor={field.id} className="text-white/70 text-sm">
                              {field.helperText || field.label}
                            </label>
                          </div>
                        )}
                      </div>
                    ))}
                    
                    <div className="pt-4 flex justify-end gap-3">
                      <button
                        onClick={() => {
                          setSelectedSpaceType(null);
                          setFormData({});
                          setApiError(null);
                        }}
                        className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:bg-white/5 transition-colors"
                      >
                        Back
                      </button>
                      <button
                        onClick={handleCreateNewSpace}
                        disabled={isLoading}
                        className="button-primary flex items-center gap-2"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Creating...</span>
                          </>
                        ) : (
                          'Create Space'
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}