import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '@clerk/clerk-react';
import toast from 'react-hot-toast';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API context type
interface ApiContextType {
  isLoading: boolean;
  error: string | null;
  getUserSpaces: (limit?: number, offset?: number) => Promise<any>;
  createSpace: (title: string, type: string) => Promise<any>;
  getSpace: (spaceId: number) => Promise<any>;
  sendMessage: (spaceId: number, content: string, useWeb?: boolean) => Promise<any>;
  getUserStats: () => Promise<any>;
  clearError: () => void;
}

// Create context
const ApiContext = createContext<ApiContextType | undefined>(undefined);

// API provider component
export const ApiProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { userId, getToken, isSignedIn } = useAuth();

  // Configure axios interceptors to add auth token
  useEffect(() => {
    const requestInterceptor = api.interceptors.request.use(
      async (config) => {
        if (isSignedIn) {
          try {
            const token = await getToken();
            if (token) {
              config.headers['Authorization'] = `Bearer ${token}`;
            }
          } catch (err) {
            console.error('Error getting auth token:', err);
          }
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );
    
    // Add response interceptor for global error handling
    const responseInterceptor = api.interceptors.response.use(
      (response) => response,
      (error) => {
        // Handle 401 Unauthorized errors globally
        if (error.response?.status === 401) {
          toast.error('Your session has expired. Please sign in again.');
        }
        
        // Handle server errors
        if (error.response?.status >= 500) {
          toast.error('Server error. Please try again later.');
        }
        
        return Promise.reject(error);
      }
    );

    return () => {
      api.interceptors.request.eject(requestInterceptor);
      api.interceptors.response.eject(responseInterceptor);
    };
  }, [getToken, isSignedIn]);

  // Get user spaces
  const getUserSpaces = async (limit = 10, offset = 0) => {
    if (!userId) return { spaces: [] };
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.get(`/user/spaces?user_id=${userId}&limit=${limit}&offset=${offset}`);
      return response.data;
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to fetch spaces';
      setError(errorMsg);
      return { spaces: [] };
    } finally {
      setIsLoading(false);
    }
  };

  // Create a new space
  const createSpace = async (title: string, type: string) => {
    if (!userId) throw new Error('User not authenticated');
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Get user data from Clerk
      const userAuth = {
        user_id: userId,
        email: '', // Will be populated from auth token
        first_name: '',
        last_name: ''
      };
      
      const response = await api.post('/user/spaces', {
        title,
        type,
        user_auth: userAuth
      });
      
      toast.success('Space created successfully!');
      return response.data;
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to create space';
      setError(errorMsg);
      toast.error(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  // Get a space by ID
  const getSpace = async (spaceId: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.get(`/spaces/${spaceId}`);
      return response.data;
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to fetch space';
      setError(errorMsg);
      toast.error(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  // Send a message in a space
  const sendMessage = async (spaceId: number, content: string, useWeb = true) => {
    if (!userId) throw new Error('User not authenticated');
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.post(`/spaces/${spaceId}/chat?user_id=${userId}`, {
        query: content,
        use_web: useWeb
      });
      return response.data;
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to send message';
      setError(errorMsg);
      toast.error(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  // Get user statistics
  const getUserStats = async () => {
    if (!userId) return { stats: { total_spaces: 0, messages_this_month: 0, active_researches: 0 } };
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.get(`/user/${userId}/stats`);
      return response.data;
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to fetch user stats';
      setError(errorMsg);
      console.error(errorMsg);
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

  // Clear error
  const clearError = () => setError(null);

  // Context value
  const value = {
    isLoading,
    error,
    getUserSpaces,
    createSpace,
    getSpace,
    sendMessage,
    getUserStats,
    clearError
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

// Hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

// Export the axios instance for direct use
export { api };
