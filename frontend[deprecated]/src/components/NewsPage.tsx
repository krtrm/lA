import React, { useState, useEffect } from 'react'
import { ExternalLink, BookOpen, Plus, Search, Calendar, Loader2, User, Newspaper, LawJustice, BarChart, FileText, MessageSquare, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { useUser } from "@clerk/clerk-react";

interface NewsArticle {
  id?: string;
  title: string;
  description?: string;
  content?: string;
  url: string;
  urlToImage?: string;
  publishedAt: string;
  source: {
    id?: string;
    name: string;
  };
  author?: string;
}

export default function NewsPage() {
  const { user } = useUser();
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [opinions, setOpinions] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [opinionLoading, setOpinionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('legal');
  const [searchQuery, setSearchQuery] = useState('');
  const [isComposeOpen, setIsComposeOpen] = useState(false);
  const [newOpinion, setNewOpinion] = useState({
    title: '',
    content: '',
    author: ''
  });

  const tabOptions = [
    { value: 'legal', label: 'Legal News', icon: <LawJustice className="w-4 h-4" /> },
    { value: 'judgments', label: 'Recent Judgments', icon: <FileText className="w-4 h-4" /> },
    { value: 'analysis', label: 'Legal Analysis', icon: <BarChart className="w-4 h-4" /> },
    { value: 'opinion', label: 'Opinion', icon: <MessageSquare className="w-4 h-4" /> }
  ];

  // Format the date to a more readable format
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      day: 'numeric', 
      month: 'short', 
      year: 'numeric' 
    });
  };

  // Get estimated read time based on content length
  const getReadTime = (content: string = '') => {
    const wordsPerMinute = 200;
    const words = content.split(/\s+/).length;
    const minutes = Math.ceil(words / wordsPerMinute);
    return `${minutes} min read`;
  };

  // Load user opinions from localStorage
  const loadOpinions = () => {
    const savedOpinions = api.getSavedOpinions();
    setOpinions(savedOpinions);
  };

  // Fetch law and judicial news
  const fetchNews = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Add legal terms to search query
      let legalQuery = searchQuery;
      if (activeTab === 'legal') {
        legalQuery = searchQuery || "supreme court OR judiciary OR law";
      } else if (activeTab === 'judgments') {
        legalQuery = searchQuery || "judgment OR verdict OR ruling";
      } else if (activeTab === 'analysis') {
        legalQuery = searchQuery || "legal analysis OR implications OR interpretation";
      }
      
      // Don't fetch for opinion tab since we're using localStorage
      if (activeTab !== 'opinion') {
        const data = await api.getNews(legalQuery);
        setArticles(data);
      }
    } catch (err) {
      console.error('Failed to fetch news:', err);
      setError('Failed to load news articles. Please try again later.');
      
      // Use placeholder data for development
      setArticles([
        {
          id: '1',
          title: 'Supreme Court Issues New Guidelines on Bail Applications',
          description: 'The Supreme Court has issued comprehensive guidelines to streamline the bail application process across all courts in India.',
          content: 'The Supreme Court of India has issued a landmark judgment today that establishes comprehensive guidelines for handling bail applications across all courts in the country.',
          url: 'https://example.com/news/1',
          urlToImage: 'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&q=80&w=800',
          publishedAt: '2023-08-15T10:30:00Z',
          source: {
            name: 'Legal Daily'
          }
        },
        {
          id: '2',
          title: 'Parliament Passes Digital Personal Data Protection Act',
          description: 'The Indian Parliament has passed the Digital Personal Data Protection Act, establishing new regulations for handling personal data.',
          content: 'In a significant move toward regulating the digital ecosystem, the Indian Parliament has passed the Digital Personal Data Protection Act after years of deliberation.',
          url: 'https://example.com/news/2',
          urlToImage: 'https://images.unsplash.com/photo-1607799279861-4dd421887fb3?auto=format&fit=crop&q=80&w=800',
          publishedAt: '2023-07-28T14:15:00Z',
          source: {
            name: 'Tech Law Today'
          }
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle search input
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchNews();
  };

  // Handle tab change
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setSearchQuery('');
  };

  // Handle compose form submit for opinions
  const handleComposeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newOpinion.title || !newOpinion.content) {
      alert('Please fill in all required fields');
      return;
    }
    
    setOpinionLoading(true);
    
    try {
      // Prepare the opinion with the user's name from Clerk
      const opinion = {
        title: newOpinion.title,
        content: newOpinion.content,
        author: user?.fullName || user?.username || "Anonymous User"
      };
      
      // Try to post to the API first
      let newArticle: NewsArticle | null = null;
      
      try {
        newArticle = await api.postOpinion(opinion);
      } catch (apiError) {
        console.error('API error when posting opinion:', apiError);
        // Fallback to creating locally if API fails
      }
      
      // If API failed, create the article locally
      if (!newArticle) {
        newArticle = {
          id: `op-${Date.now()}`,
          title: opinion.title,
          content: opinion.content,
          description: opinion.content.substring(0, 120) + '...',
          url: window.location.href + '#opinion',
          publishedAt: new Date().toISOString(),
          source: {
            name: "User Opinion"
          },
          author: opinion.author,
          urlToImage: 'https://images.unsplash.com/photo-1589578527966-fdac0f44566c?auto=format&fit=crop&q=80&w=800'
        };
      }
      
      // Save to localStorage
      api.saveOpinion(newArticle);
      
      // Update the opinions state
      setOpinions([newArticle, ...opinions]);
      
      // Reset form and close compose modal
      setNewOpinion({ title: '', content: '', author: '' });
      setIsComposeOpen(false);
    } catch (err) {
      console.error('Error posting opinion:', err);
      alert('Failed to post your opinion. Please try again.');
    } finally {
      setOpinionLoading(false);
    }
  };

  // Fetch news when component mounts or when tab/search changes
  useEffect(() => {
    if (activeTab === 'opinion') {
      loadOpinions();
    } else {
      fetchNews();
    }
  }, [activeTab]);

  // Decide which articles to display based on the active tab
  const displayArticles = activeTab === 'opinion' ? opinions : articles;

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <h1 className="text-3xl font-bold text-white">Legal News & Updates</h1>
          <div className="flex items-center gap-4">
            <form onSubmit={handleSearch} className="relative">
              <Input
                type="text"
                placeholder="Search news..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full md:w-64 pr-8"
              />
              <button 
                type="submit" 
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
              >
                <Search className="w-4 h-4" />
              </button>
            </form>
            {activeTab === 'opinion' && (
              <Button 
                onClick={() => setIsComposeOpen(true)} 
                className="flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Write Opinion
              </Button>
            )}
          </div>
        </div>
        
        {/* Category tabs */}
        <Tabs value={activeTab} onValueChange={handleTabChange} className="mb-8">
          <TabsList className="mb-4">
            {tabOptions.map((tab) => (
              <TabsTrigger 
                key={tab.value} 
                value={tab.value}
                className="flex items-center gap-2"
              >
                {tab.icon}
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>
          
          {tabOptions.map((tab) => (
            <TabsContent key={tab.value} value={tab.value}>
              <p className="text-gray-400 mb-4">
                {tab.value === 'legal' && 'The latest legal news from courts, law firms, and legal departments'}
                {tab.value === 'judgments' && 'Recent significant judgments from the Supreme Court and High Courts'}
                {tab.value === 'analysis' && 'Expert analysis of important legal developments and their implications'}
                {tab.value === 'opinion' && 'Insightful opinions and perspectives from legal professionals and community members'}
              </p>
            </TabsContent>
          ))}
        </Tabs>
        
        {/* Opinion Composer Modal */}
        <AnimatePresence>
          {isComposeOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
              onClick={() => setIsComposeOpen(false)}
            >
              <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                className="glass-effect border border-white/10 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-6">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-bold text-white">Share Your Legal Opinion</h2>
                    <button
                      onClick={() => setIsComposeOpen(false)}
                      className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                    >
                      <X className="w-5 h-5 text-white/70" />
                    </button>
                  </div>
                  
                  <form onSubmit={handleComposeSubmit} className="space-y-4">
                    <div>
                      <label htmlFor="opinion-title" className="block text-white mb-1 text-sm font-medium">Title</label>
                      <Input
                        id="opinion-title"
                        value={newOpinion.title}
                        onChange={(e) => setNewOpinion({...newOpinion, title: e.target.value})}
                        placeholder="E.g., The Impact of Recent Supreme Court Judgments on Digital Privacy"
                        required
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label htmlFor="opinion-content" className="block text-white mb-1 text-sm font-medium">Content</label>
                      <textarea
                        id="opinion-content"
                        value={newOpinion.content}
                        onChange={(e) => setNewOpinion({...newOpinion, content: e.target.value})}
                        placeholder="Write your legal opinion or analysis here. Support your arguments with references to cases, statutes, or legal principles where applicable."
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-transparent transition-colors duration-200 resize-y min-h-[250px]"
                        required
                      />
                    </div>
                    <div>
                      <p className="text-white/50 text-sm">
                        Your opinion will be published under {user?.fullName || user?.username || "your name"}
                      </p>
                    </div>
                    <div className="flex justify-end gap-2 pt-2">
                      <Button 
                        type="button" 
                        onClick={() => setIsComposeOpen(false)}
                        variant="outline"
                        disabled={opinionLoading}
                      >
                        Cancel
                      </Button>
                      <Button 
                        type="submit"
                        disabled={opinionLoading}
                        className="min-w-[100px]"
                      >
                        {opinionLoading ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Posting...
                          </>
                        ) : 'Publish Opinion'}
                      </Button>
                    </div>
                  </form>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Loading state */}
        {loading && (
          <div className="flex justify-center items-center py-20">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <span className="ml-2 text-white">Loading articles...</span>
          </div>
        )}
        
        {/* Error state */}
        {error && !loading && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg mb-6">
            {error}
          </div>
        )}
        
        {/* Articles grid */}
        {!loading && !error && (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {displayArticles.map((article, index) => (
              <motion.article
                key={article.id || index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ scale: 1.02 }}
                className="rounded-xl overflow-hidden backdrop-blur-xl bg-white/5 border border-white/10 group h-full flex flex-col"
              >
                <div className="relative h-48 overflow-hidden">
                  <img
                    src={article.urlToImage || 'https://images.unsplash.com/photo-1589578527966-fdac0f44566c?auto=format&fit=crop&q=80&w=800'}
                    alt={article.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
                  <div className="absolute bottom-4 left-4 flex flex-wrap gap-2">
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400">
                      {article.source?.name || 'Legal News'}
                    </span>
                    {activeTab === 'opinion' && (
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400">
                        Opinion
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="p-6 flex-grow flex flex-col">
                  <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-blue-400 transition-colors leading-tight">
                    {article.title}
                  </h3>
                  <p className="text-gray-400 mb-4 flex-grow">{article.description || article.content?.substring(0, 120) + '...'}</p>
                  <div className="flex items-center justify-between mt-auto">
                    <div className="flex items-center text-gray-500">
                      {article.author ? (
                        <div className="flex items-center gap-1.5">
                          <User className="w-3.5 h-3.5" />
                          <span className="text-xs truncate max-w-[120px]">{article.author}</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5">
                          <Calendar className="w-3.5 h-3.5" />
                          <span className="text-xs">{formatDate(article.publishedAt)}</span>
                        </div>
                      )}
                    </div>
                    <a 
                      href={article.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-blue-500 hover:text-blue-400 transition-colors text-sm"
                    >
                      Read more <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>
              </motion.article>
            ))}
          </div>
        )}
        
        {/* Empty state */}
        {!loading && !error && displayArticles.length === 0 && (
          <div className="text-center py-20">
            {activeTab === 'opinion' ? (
              <>
                <MessageSquare className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white">No opinions yet</h3>
                <p className="text-gray-400 mt-2 mb-4">
                  Be the first to share your legal perspective with the community
                </p>
                <Button onClick={() => setIsComposeOpen(true)} className="mx-auto">
                  <Plus className="w-4 h-4 mr-2" />
                  Write an Opinion
                </Button>
              </>
            ) : (
              <>
                <Newspaper className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white">No articles found</h3>
                <p className="text-gray-400 mt-2">
                  {searchQuery 
                    ? `No articles found for "${searchQuery}". Try a different search term.`
                    : 'No articles available in this category right now.'
                  }
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
