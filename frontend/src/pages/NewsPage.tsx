import React, { useState, useEffect, useCallback } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import { motion } from 'framer-motion';
import { useApi } from '../context/ApiContext';
import { 
  BookOpen, Calendar, User, Clock, Heart, MessageSquare, 
  Plus, Search, Filter, RefreshCw, 
  Newspaper, CheckVerified, Loader, ChevronDown
} from 'lucide-react';

export default function NewsPage() {
  const { user } = useUser();
  const { getBlogs, getBlogCategories, fetchOfficialNews, isLoading, error, setError } = useApi();
  const [officialNews, setOfficialNews] = useState([]);
  const [communityBlogs, setCommunityBlogs] = useState([]);
  const [categories, setCategories] = useState([]);
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [refreshingNews, setRefreshingNews] = useState(false);
  const [activeTab, setActiveTab] = useState('all'); // 'all', 'official', 'community'
  const [pageLoading, setPageLoading] = useState(true);
  
  const categoryParam = searchParams.get('category');
  
  // Fetch categories when component mounts
  useEffect(() => {
    if (categoryParam) {
      setSelectedCategory(categoryParam);
    }
    
    const fetchCategories = async () => {
      try {
        const response = await getBlogCategories();
        if (response && response.categories) {
          setCategories(response.categories);
        }
      } catch (err) {
        console.error('Error fetching categories:', err);
        setError("Failed to load blog categories. Please try again later.");
      }
    };
    
    fetchCategories();
  }, [categoryParam, getBlogCategories, setError]);
  
  // Fetch blogs based on filters
  const fetchAllNews = useCallback(async () => {
    setPageLoading(true);
    try {
      // Fetch official news
      const officialResponse = await getBlogs(
        selectedCategory || null,
        null,
        true, // is_official = true
        true // published_only = true
      );
      
      if (officialResponse && officialResponse.blogs) {
        setOfficialNews(officialResponse.blogs);
      }
      
      // Fetch community blogs
      const communityResponse = await getBlogs(
        selectedCategory || null,
        null, 
        false, // is_official = false
        true // published_only = true
      );
      
      if (communityResponse && communityResponse.blogs) {
        setCommunityBlogs(communityResponse.blogs);
      }
    } catch (err) {
      console.error('Error fetching news and blogs:', err);
      setError("Failed to load blogs. Please try again later.");
    } finally {
      setPageLoading(false);
    }
  }, [getBlogs, selectedCategory, setError]);
  
  useEffect(() => {
    fetchAllNews();
  }, [fetchAllNews]);
  
  const handleRefreshOfficialNews = async () => {
    try {
      setRefreshingNews(true);
      const response = await fetchOfficialNews(true); // force_refresh = true
      
      if (response && response.news) {
        setOfficialNews(response.news);
      }
      // Re-fetch all news after refresh to ensure everything is up to date
      await fetchAllNews();
    } catch (err) {
      console.error('Error refreshing official news:', err);
      setError("Failed to refresh official news. Please try again later.");
    } finally {
      setRefreshingNews(false);
    }
  };
  
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    // Update URL params for search
    const params = new URLSearchParams(searchParams);
    if (searchTerm) {
      params.set('search', searchTerm);
    } else {
      params.delete('search');
    }
    setSearchParams(params);
  };
  
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    // Update URL params for category
    const params = new URLSearchParams(searchParams);
    if (category) {
      params.set('category', category);
    } else {
      params.delete('category');
    }
    setSearchParams(params);
  };
  
  // Filter blogs based on search term
  const filteredOfficialNews = officialNews.filter(blog => {
    if (!blog || !blog.title) return false; // Safety check for null blogs
    
    if (searchTerm && !blog.title.toLowerCase().includes(searchTerm.toLowerCase()) && 
        !blog.summary?.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    return true;
  });
  
  const filteredCommunityBlogs = communityBlogs.filter(blog => {
    if (!blog || !blog.title) return false; // Safety check for null blogs
    
    if (searchTerm && !blog.title.toLowerCase().includes(searchTerm.toLowerCase()) && 
        !blog.summary?.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    return true;
  });
  
  // Combined blogs for "All" tab, prioritizing official news
  const allPosts = [...filteredOfficialNews, ...filteredCommunityBlogs];
  
  // Determine which posts to display based on active tab
  const displayPosts = activeTab === 'all' ? allPosts : 
                       activeTab === 'official' ? filteredOfficialNews : 
                       filteredCommunityBlogs;
  
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
        {/* Header Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Legal Insights Hub</h1>
          <p className="text-white/60">
            Stay informed with verified legal updates and community perspectives on Indian law
          </p>
        </div>
        
        {/* Tab bar */}
        <div className="flex border-b border-white/10 mb-6">
          <button 
            className={`py-3 px-5 font-medium text-sm ${activeTab === 'all' ? 'text-white border-b-2 border-primary' : 'text-white/60 hover:text-white/80'}`}
            onClick={() => setActiveTab('all')}
          >
            All Articles
          </button>
          <button 
            className={`py-3 px-5 font-medium text-sm flex items-center ${activeTab === 'official' ? 'text-white border-b-2 border-primary' : 'text-white/60 hover:text-white/80'}`}
            onClick={() => setActiveTab('official')}
          >
            <CheckVerified className="h-4 w-4 mr-2" />
            Verified Updates
          </button>
          <button 
            className={`py-3 px-5 font-medium text-sm flex items-center ${activeTab === 'community' ? 'text-white border-b-2 border-primary' : 'text-white/60 hover:text-white/80'}`}
            onClick={() => setActiveTab('community')}
          >
            <Newspaper className="h-4 w-4 mr-2" />
            Community Perspectives
          </button>
        </div>
        
        {/* Action bar */}
        <div className="flex flex-col md:flex-row gap-4 mb-8 justify-between">
          {/* Search */}
          <form onSubmit={handleSearchSubmit} className="flex-1">
            <div className="relative">
              <input
                type="text"
                placeholder="Search articles..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-field pl-10"
              />
              <Search className="absolute top-3 left-3 h-5 w-5 text-white/40" />
            </div>
          </form>
          
          <div className="flex gap-4">
            {/* Category filter */}
            <div className="relative group">
              <button className="button-secondary w-full md:w-auto flex items-center justify-center">
                <Filter className="h-4 w-4 mr-2" />
                {selectedCategory || 'All Categories'}
                <ChevronDown className="h-4 w-4 ml-2" />
              </button>
              
              <div className="absolute right-0 top-full mt-2 bg-secondary rounded-xl p-2 border border-white/10 shadow-xl z-10 w-48 hidden group-hover:block">
                <div className="py-1 px-2 text-sm text-white/70 hover:bg-white/5 rounded-lg cursor-pointer" onClick={() => handleCategoryChange('')}>
                  All Categories
                </div>
                {categories.map((category) => (
                  <div 
                    key={category}
                    className="py-1 px-2 text-sm text-white/70 hover:bg-white/5 rounded-lg cursor-pointer"
                    onClick={() => handleCategoryChange(category)}
                  >
                    {category}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Refresh news button */}
            {activeTab !== 'community' && (
              <button 
                className="button-secondary w-full md:w-auto flex items-center justify-center"
                onClick={handleRefreshOfficialNews}
                disabled={refreshingNews}
              >
                {refreshingNews ? (
                  <Loader className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4 mr-2" />
                )}
                Refresh Updates
              </button>
            )}
            
            {/* Create New Post button (only for logged in users) */}
            {user && (
              <Link to="/blogs/new" className="button-primary w-full md:w-auto flex items-center justify-center">
                <Plus className="h-4 w-4 mr-2" />
                Write Article
              </Link>
            )}
          </div>
        </div>
        
        {/* Error alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-white">
            {error}
          </div>
        )}
        
        {/* Loading state */}
        {isLoading || pageLoading ? (
          <div className="glass-effect rounded-xl p-8 text-center">
            <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-white/60">Loading articles...</p>
          </div>
        ) : displayPosts && displayPosts.length > 0 ? (
          <div className="space-y-8">
            {/* Featured section - show only on "All" or "Official" tabs */}
            {(activeTab === 'all' || activeTab === 'official') && filteredOfficialNews && filteredOfficialNews.length > 0 && (
              <div className="mb-12">
                <FeaturedArticle article={filteredOfficialNews[0]} />
              </div>
            )}
            
            {/* Main grid of posts */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {displayPosts
                .slice(activeTab === 'all' || activeTab === 'official' ? (filteredOfficialNews.length > 0 ? 1 : 0) : 0)
                .map((post) => (
                  <ArticleCard key={post.blog_id} article={post} />
                ))}
            </div>
          </div>
        ) : (
          <div className="glass-effect rounded-xl p-8 text-center">
            <BookOpen className="h-12 w-12 text-white/20 mx-auto mb-4" />
            <h3 className="text-xl font-medium text-white mb-2">No articles found</h3>
            <p className="text-white/60 mb-6">
              {activeTab === 'official' ? 
                "No official news articles found. Try refreshing or checking back later." :
                activeTab === 'community' ?
                "No community articles found. Be the first to contribute!" :
                "No articles match your search criteria."}
            </p>
            {user && activeTab === 'community' && (
              <Link to="/blogs/new" className="button-primary inline-flex items-center">
                <Plus className="h-4 w-4 mr-2" />
                Write an Article
              </Link>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function FeaturedArticle({ article }) {
  if (!article) return null;
  
  return (
    <Link to={`/blogs/${article.blog_id}`}>
      <motion.div
        whileHover={{ scale: 1.01 }}
        className="glass-effect rounded-xl overflow-hidden cursor-pointer hover:border-blue-500/30 border border-white/5 transition-colors"
      >
        <div className="p-8">
          {/* Category and Official badge */}
          <div className="flex items-center mb-4 gap-3">
            {article.category && (
              <span className="px-3 py-1 text-xs font-medium rounded-full bg-blue-500/20 text-blue-400">
                {article.category}
              </span>
            )}
            {article.is_official && (
              <span className="flex items-center px-3 py-1 text-xs font-medium rounded-full bg-green-500/20 text-green-400">
                <CheckVerified className="h-3 w-3 mr-1" />
                Official
              </span>
            )}
            {article.source_name && (
              <span className="text-white/50 text-xs flex items-center">
                Source: {article.source_name}
              </span>
            )}
          </div>
          
          {/* Title */}
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">{article.title}</h2>
          
          {/* Summary */}
          {article.summary && (
            <p className="text-white/80 text-lg mb-6">{article.summary}</p>
          )}
          
          {/* Meta info */}
          <div className="flex flex-wrap items-center text-white/50 text-sm gap-x-4 gap-y-2">
            <div className="flex items-center">
              <User className="h-4 w-4 mr-1" />
              <span>{article.author}</span>
            </div>
            <div className="flex items-center">
              <Calendar className="h-4 w-4 mr-1" />
              <span>{article.published_at}</span>
            </div>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              <span>{article.read_time_minutes} min read</span>
            </div>
            <div className="flex items-center">
              <Heart className="h-4 w-4 mr-1" />
              <span>{article.likes || 0}</span>
            </div>
            <div className="flex items-center">
              <MessageSquare className="h-4 w-4 mr-1" />
              <span>{article.comment_count || 0}</span>
            </div>
          </div>
        </div>
      </motion.div>
    </Link>
  );
}

function ArticleCard({ article }) {
  if (!article) return null;
  
  return (
    <Link to={`/blogs/${article.blog_id}`}>
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className="glass-effect rounded-xl p-6 cursor-pointer hover:border-blue-500/30 border border-white/5 transition-colors h-full flex flex-col"
      >
        {/* Category badge and official badge if exists */}
        <div className="flex flex-wrap gap-2 mb-3">
          {article.category && (
            <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-500/20 text-blue-400">
              {article.category}
            </span>
          )}
          {article.is_official && (
            <span className="flex items-center px-2 py-1 text-xs font-medium rounded-full bg-green-500/20 text-green-400">
              <CheckVerified className="h-3 w-3 mr-1" />
              Official
            </span>
          )}
        </div>
        
        {/* Title */}
        <h3 className="text-xl font-medium text-white mb-2 line-clamp-2">{article.title}</h3>
        
        {/* Summary */}
        {article.summary && (
          <p className="text-white/70 text-sm mb-4 line-clamp-3 flex-grow">{article.summary}</p>
        )}
        
        {/* Meta info */}
        <div className="flex flex-wrap items-center text-white/50 text-xs gap-x-3 gap-y-2 mt-auto pt-4 border-t border-white/10">
          <div className="flex items-center">
            <User className="h-3 w-3 mr-1" />
            <span className="truncate max-w-[80px]">{article.author}</span>
          </div>
          <div className="flex items-center">
            <Calendar className="h-3 w-3 mr-1" />
            <span>{article.published_at}</span>
          </div>
          <div className="flex items-center">
            <Clock className="h-3 w-3 mr-1" />
            <span>{article.read_time_minutes} min</span>
          </div>
          <div className="flex items-center">
            <Heart className="h-3 w-3 mr-1" />
            <span>{article.likes || 0}</span>
          </div>
          <div className="flex items-center">
            <MessageSquare className="h-3 w-3 mr-1" />
            <span>{article.comment_count || 0}</span>
          </div>
        </div>
      </motion.div>
    </Link>
  );
}