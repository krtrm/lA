declare module './context/ApiContext' {
  import { ReactNode } from 'react';
  
  export interface Article {
    blog_id: string;
    title: string;
    summary: string;
    content: string;
    author: string;
    published_at: string;
    read_time_minutes: number;
    likes: number;
    comment_count: number;
    tags: string[];
    category: string;
    is_official: boolean;
    source_name: string;
    user_id: string;
    comments: Comment[];
  }
  
  export interface Comment {
    user_id: string;
    comment: string;
    timestamp: string;
    author_name?: string;
  }
  
  export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
  }
  
  export interface ChatSession {
    id: string;
    title: string;
    messages: ChatMessage[];
    createdAt: string;
    updatedAt: string;
  }
  
  export interface ApiContextType {
    articles: Article[];
    searchResults: Article[];
    isLoading: boolean;
    chatSessions: ChatSession[];
    currentChatSession: ChatSession | null;
    setSearchResults: (results: Article[]) => void;
    setIsLoading: (loading: boolean) => void;
    fetchBlogs: () => Promise<Article[]>;
    fetchBlogById: (id: string) => Promise<Article>;
    likeBlog: (blogId: string) => Promise<void>;
    commentOnBlog: (blogId: string, comment: string) => Promise<void>;
    searchBlogs: (query: string) => Promise<Article[]>;
    fetchBlogsByCategory: (category: string) => Promise<Article[]>;
    createChatSession: () => Promise<ChatSession>;
    loadChatSessions: () => Promise<ChatSession[]>;
    loadChatSession: (sessionId: string) => Promise<ChatSession>;
    sendChatMessage: (sessionId: string, message: string) => Promise<ChatMessage>;
    createBlog: (blog: Partial<Article>) => Promise<Article>;
  }
  
  export const ApiProvider: React.FC<{ children: ReactNode }>;
  export const useApi: () => ApiContextType;
}

declare module '../context/ApiContext' {
  export * from './context/ApiContext';
}
