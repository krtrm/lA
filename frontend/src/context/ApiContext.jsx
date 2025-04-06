import React, { createContext, useContext, useState } from 'react';
import { api } from '../services/api';

// Create a context for the API
const ApiContext = createContext(null);

// Export a hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within ApiProvider');
  }
  return context;
};

export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to query the API
  const query = async (request) => {
    setIsLoading(true);
    setError(null);
    try {
      console.log("Making query:", request);
      const result = await api.query(request);
      return result;
    } catch (err) {
      console.error("API query error in context:", err);
      setError(err.message || "An error occurred during the request");
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Simple placeholder functions that don't rely on other missing pieces
  const getUserSpaces = async (limit = 3) => {
    setIsLoading(true);
    try {
      console.log("Getting user spaces, limit:", limit);
      // Mocked response for now
      return {
        spaces: [
          { space_id: 1, title: "Legal Research", type: "legal_research", message_count: 5, created_at: "2023-01-01" },
          { space_id: 2, title: "Document Analysis", type: "document_drafting", message_count: 3, created_at: "2023-01-02" }
        ]
      };
    } catch (error) {
      console.error("Error fetching spaces:", error);
      setError("Failed to fetch spaces");
      return { spaces: [] };
    } finally {
      setIsLoading(false);
    }
  };

  const getUserStats = async () => {
    setIsLoading(true);
    try {
      console.log("Getting user stats");
      // Mocked response for now
      return {
        stats: {
          total_spaces: 2,
          messages_this_month: 25,
          active_researches: 1
        }
      };
    } catch (error) {
      console.error("Error fetching stats:", error);
      setError("Failed to fetch user statistics");
      return { 
        stats: { 
          total_spaces: 0, 
          messages_this_month: 0, 
          active_researches: 0 
        }
      };
    } finally {
      setIsLoading(false);
    }
  };

  // Mock placeholder for blogs to make the app work without backend
  const getBlogs = async () => {
    return { blogs: [] };
  };

  const getBlogCategories = async () => {
    return { categories: ["Constitutional Law", "Criminal Law", "Civil Law"] };
  };

  const getBlog = async () => {
    return { blog: null };
  };

  const value = {
    isLoading,
    error,
    setError,
    query,
    getBlogs,
    getBlogCategories,
    getBlog,
    getUserSpaces,
    getUserStats,
    // Add other stub methods
    createBlog: async () => ({}),
    updateBlog: async () => ({}),
    deleteBlog: async () => ({}),
    likeBlog: async () => ({}),
    addBlogComment: async () => ({}),
    generateBlogContent: async () => ({}),
    fetchOfficialNews: async () => ({ news: [] })
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};