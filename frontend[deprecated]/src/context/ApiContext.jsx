import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from '@clerk/clerk-react';
import toast from 'react-hot-toast';
import { api } from '../services/api';

// Create the API context
const ApiContext = createContext(null);

export const ApiProvider = ({ children }) => {
  const { getToken, isSignedIn, userId } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Initialize localStorage token on auth state change
  useEffect(() => {
    const updateAuthToken = async () => {
      if (isSignedIn) {
        try {
          const token = await getToken();
          localStorage.setItem('clerk-auth-token', token);
        } catch (err) {
          console.error('Failed to get auth token:', err);
        }
      } else {
        localStorage.removeItem('clerk-auth-token');
      }
    };
    
    updateAuthToken();
  }, [isSignedIn, getToken]);

  // General error handler for API calls
  const handleApiError = useCallback((error, customMessage) => {
    console.error(customMessage || 'API error:', error);
    setError(error.message || 'An unexpected error occurred');
    toast.error(error.message || 'Something went wrong');
    return null;
  }, []);

  // Wrapper for API calls with loading state
  const withLoading = useCallback(async (apiCall, customErrorMessage) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiCall();
      return result;
    } catch (error) {
      return handleApiError(error, customErrorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [handleApiError]);

  // Query endpoint wrapper
  const query = useCallback((params) => {
    return withLoading(
      () => api.query(params),
      'Failed to process query'
    );
  }, [withLoading]);

  // Stream query with proper error handling
  const streamQuery = useCallback(async (params) => {
    setIsLoading(true);
    setError(null);
    
    try {
      return await api.streamQuery(params);
    } catch (error) {
      handleApiError(error, 'Failed to start streaming');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [handleApiError]);

  // Verify citation wrapper
  const verifyCitation = useCallback((citation) => {
    return withLoading(
      () => api.verifyCitation(citation),
      'Failed to verify citation'
    );
  }, [withLoading]);

  // Create outline wrapper
  const createOutline = useCallback((topic, docType) => {
    return withLoading(
      () => api.createOutline(topic, docType),
      'Failed to create document outline'
    );
  }, [withLoading]);

  // Generate argument wrapper
  const generateArgument = useCallback((topic, points) => {
    return withLoading(
      () => api.generateArgument(topic, points),
      'Failed to generate legal argument'
    );
  }, [withLoading]);

  // Get user spaces wrapper
  const getUserSpaces = useCallback((limit) => {
    return withLoading(
      () => api.getUserSpaces(limit),
      'Failed to fetch spaces'
    );
  }, [withLoading]);

  // Get space details wrapper
  const getSpace = useCallback((spaceId) => {
    return withLoading(
      () => api.getSpace(spaceId),
      'Failed to fetch space details'
    );
  }, [withLoading]);

  // Create space wrapper
  const createSpace = useCallback((title, type) => {
    return withLoading(
      () => api.createSpace(title, type),
      'Failed to create space'
    );
  }, [withLoading]);

  // Send message in space wrapper
  const sendMessage = useCallback((spaceId, message) => {
    return withLoading(
      () => api.sendMessage(spaceId, message),
      'Failed to send message'
    );
  }, [withLoading]);

  // Get user statistics wrapper
  const getUserStats = useCallback(() => {
    return withLoading(
      () => api.getUserStats(),
      'Failed to fetch user statistics'
    );
  }, [withLoading]);

  // Get blogs wrapper
  const getBlogs = useCallback((category, limit) => {
    return withLoading(
      () => api.getBlogs(category, limit),
      'Failed to fetch blogs'
    );
  }, [withLoading]);

  // Get blog details wrapper
  const getBlog = useCallback((blogId) => {
    return withLoading(
      () => api.getBlog(blogId),
      'Failed to fetch blog'
    );
  }, [withLoading]);

  // Create blog wrapper
  const createBlog = useCallback((blogData) => {
    return withLoading(
      () => api.createBlog(blogData),
      'Failed to create blog'
    );
  }, [withLoading]);

  // Like blog wrapper
  const likeBlog = useCallback((blogId) => {
    return withLoading(
      () => api.likeBlog(blogId),
      'Failed to like blog'
    );
  }, [withLoading]);

  // Comment on blog wrapper
  const commentOnBlog = useCallback((blogId, comment) => {
    return withLoading(
      () => api.commentOnBlog(blogId, comment),
      'Failed to add comment'
    );
  }, [withLoading]);

  // Clear any stored error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // The context value that will be passed to consumers
  const contextValue = {
    isLoading,
    error,
    clearError,
    // API methods
    query,
    streamQuery,
    verifyCitation,
    createOutline,
    generateArgument,
    getUserSpaces,
    getSpace,
    createSpace,
    sendMessage,
    getUserStats,
    getBlogs,
    getBlog,
    createBlog,
    likeBlog,
    commentOnBlog,
  };

  return (
    <ApiContext.Provider value={contextValue}>
      {children}
    </ApiContext.Provider>
  );
};

// Custom hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};