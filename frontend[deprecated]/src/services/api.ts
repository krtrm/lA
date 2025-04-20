import axios from 'axios';

// Define API base URL with fallback to localhost
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to add auth token
apiClient.interceptors.request.use(function (config) {
  const token = localStorage.getItem('clerk-auth-token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, function (error) {
  return Promise.reject(error);
});

// Define interface for API responses
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Define specific response types
interface QueryResponse {
  answer: string;
  sources?: Array<{
    title: string;
    content: string;
    url?: string;
  }>;
}

interface OutlineResponse {
  outline: string;
}

interface ArgumentResponse {
  argument: string;
}

interface CitationResponse {
  is_valid: boolean;
  corrected_citation?: string;
  summary?: string;
}

interface UserStats {
  total_spaces: number;
  messages_this_month: number;
  active_researches: number;
}

interface Space {
  space_id: string;
  title: string;
  type: string;
  created_at: string;
  last_active: string;
  message_count: number;
  last_message?: string;
}

interface Message {
  message_id: string;
  content: string;
  role: string;
  created_at: string;
}

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

interface Opinion {
  title: string;
  content: string;
  author: string;
}

// API service with methods for each endpoint
export const api = {
  // Query the LLM with a question
  async query(params: { query: string; use_web?: boolean }): Promise<QueryResponse> {
    try {
      const response = await apiClient.post<QueryResponse>('/query', params);
      return response.data;
    } catch (error) {
      console.error('API query error:', error);
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.error || 'Failed to get answer');
      }
      throw new Error('Network error when querying API');
    }
  },

  // Create a document outline
  async createOutline(topic: string, docType: string): Promise<OutlineResponse> {
    try {
      const response = await apiClient.post<OutlineResponse>('/create_outline', {
        topic,
        doc_type: docType,
      });
      return response.data;
    } catch (error) {
      console.error('API create outline error:', error);
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.error || 'Failed to create outline');
      }
      throw new Error('Network error when creating outline');
    }
  },

  // Generate a legal argument
  async generateArgument(topic: string, points: string[]): Promise<ArgumentResponse> {
    try {
      const response = await apiClient.post<ArgumentResponse>('/generate_argument', {
        topic,
        points,
      });
      return response.data;
    } catch (error) {
      console.error('API generate argument error:', error);
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.error || 'Failed to generate argument');
      }
      throw new Error('Network error when generating argument');
    }
  },

  // Verify a legal citation
  async verifyCitation(citation: string): Promise<CitationResponse> {
    try {
      const response = await apiClient.post<CitationResponse>('/verify_citation', {
        citation,
      });
      return response.data;
    } catch (error) {
      console.error('API verify citation error:', error);
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.error || 'Failed to verify citation');
      }
      throw new Error('Network error when verifying citation');
    }
  },

  // Get user statistics
  async getUserStats(): Promise<UserStats> {
    try {
      const response = await apiClient.get<ApiResponse<UserStats>>('/user/stats');
      return response.data.data || { 
        total_spaces: 0, 
        messages_this_month: 0, 
        active_researches: 0 
      };
    } catch (error) {
      console.error('API get user stats error:', error);
      // Return default stats if API fails
      return { 
        total_spaces: 0, 
        messages_this_month: 0, 
        active_researches: 0 
      };
    }
  },

  // Get user spaces
  async getUserSpaces(limit?: number): Promise<Space[]> {
    try {
      const params = limit ? { limit } : {};
      const response = await apiClient.get<ApiResponse<Space[]>>('/spaces', { params });
      return response.data.data || [];
    } catch (error) {
      console.error('API get user spaces error:', error);
      return [];
    }
  },

  // Get space by ID
  async getSpace(spaceId: string): Promise<Space | null> {
    try {
      const response = await apiClient.get<ApiResponse<Space>>(`/spaces/${spaceId}`);
      return response.data.data || null;
    } catch (error) {
      console.error(`API get space ${spaceId} error:`, error);
      return null;
    }
  },

  // Create new space
  async createSpace(title: string, type: string): Promise<Space | null> {
    try {
      const response = await apiClient.post<ApiResponse<Space>>('/spaces', { title, type });
      return response.data.data || null;
    } catch (error) {
      console.error('API create space error:', error);
      return null;
    }
  },

  // Get messages for a space
  async getMessages(spaceId: string): Promise<Message[]> {
    try {
      const response = await apiClient.get<ApiResponse<Message[]>>(`/spaces/${spaceId}/messages`);
      return response.data.data || [];
    } catch (error) {
      console.error(`API get messages for space ${spaceId} error:`, error);
      return [];
    }
  },

  // Send message to a space
  async sendMessage(spaceId: string, message: string): Promise<Message | null> {
    try {
      const response = await apiClient.post<ApiResponse<Message>>(`/spaces/${spaceId}/messages`, { 
        content: message 
      });
      return response.data.data || null;
    } catch (error) {
      console.error(`API send message to space ${spaceId} error:`, error);
      return null;
    }
  },

  // Get news articles - only legal news now
  async getNews(query?: string, category?: string, country: string = 'in'): Promise<NewsArticle[]> {
    try {
      const params: any = { country };
      if (query) params.q = query;
      if (category) params.category = category;
      
      const response = await apiClient.get('/news', { params });
      return response.data.articles || [];
    } catch (error) {
      console.error('API get news error:', error);
      return [];
    }
  },
  
  // Post an opinion piece
  async postOpinion(opinion: Opinion): Promise<NewsArticle | null> {
    try {
      const response = await apiClient.post('/news/opinion', opinion);
      return response.data as NewsArticle;
    } catch (error) {
      console.error('API post opinion error:', error);
      return null;
    }
  },

  // Get saved opinions from localStorage
  getSavedOpinions(): NewsArticle[] {
    try {
      const savedOpinions = localStorage.getItem('vaqeel_opinions');
      return savedOpinions ? JSON.parse(savedOpinions) : [];
    } catch (error) {
      console.error('Error getting saved opinions:', error);
      return [];
    }
  },
  
  // Save opinion to localStorage
  saveOpinion(opinion: NewsArticle): void {
    try {
      const savedOpinions = this.getSavedOpinions();
      savedOpinions.unshift(opinion);
      localStorage.setItem('vaqeel_opinions', JSON.stringify(savedOpinions));
    } catch (error) {
      console.error('Error saving opinion:', error);
    }
  },

  // Get streaming responses
  getStreamingUrl(endpoint: string): string {
    return `${API_BASE_URL}${endpoint}`;
  },

  // Utility method to handle fetch streaming responses
  async fetchWithStreaming(url: string, data: any, options = {}): Promise<Response> {
    const token = localStorage.getItem('clerk-auth-token');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    return fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
      ...options,
    });
  },

  // Stream query results
  async streamQuery(params: { query: string; use_web?: boolean }): Promise<Response> {
    return this.fetchWithStreaming(this.getStreamingUrl('/query/stream'), params);
  },

  // Stream outline creation
  async streamOutline(topic: string, docType: string): Promise<Response> {
    return this.fetchWithStreaming(this.getStreamingUrl('/create_outline/stream'), {
      topic,
      doc_type: docType,
    });
  },

  // Stream argument generation
  async streamArgument(topic: string, points: string[]): Promise<Response> {
    return this.fetchWithStreaming(this.getStreamingUrl('/generate_argument/stream'), {
      topic,
      points,
    });
  },

  // Stream citation verification
  async streamCitation(citation: string): Promise<Response> {
    return this.fetchWithStreaming(this.getStreamingUrl('/verify_citation/stream'), {
      citation,
    });
  },
};

export default api;
