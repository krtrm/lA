declare module 'react-markdown';
declare module 'remark-gfm';

/**
 * Global TypeScript definitions for Vaqeel.app
 */

// User-related types
export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  profileImageUrl?: string;
}

// Space-related types
export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface Space {
  id: number;
  title: string;
  messages: Message[];
  type?: 'legal_research' | 'document_drafting' | 'legal_analysis' | 'citation_verification' | 'statute_interpretation';
  createdAt?: string;
  lastActive?: string;
}

export interface SpaceFormField {
  id: string;
  label: string;
  type: 'text' | 'textarea' | 'select' | 'checkbox';
  placeholder?: string;
  options?: { value: string, label: string }[];
  required?: boolean;
  helperText?: string;
}

export interface SpaceTypeOption {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  apiEndpoint: string;
  formFields: SpaceFormField[];
}

// Blog-related types
export interface BlogPost {
  id: string;
  title: string;
  content: string;
  summary: string;
  author: string;
  publishedAt: string;
  readTimeMinutes: number;
  likes: number;
  commentCount: number;
  tags: string[];
  category: string;
  isOfficial: boolean;
  userId?: string;
}

export interface BlogComment {
  id: string;
  blogId: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  content: string;
  createdAt: string;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// Form types
export interface SearchParams {
  query: string;
  filters?: Record<string, string | string[]>;
  sort?: string;
  page?: number;
  pageSize?: number;
}

// Common component props
export interface WithChildren {
  children: React.ReactNode;
}

export interface WithClassName {
  className?: string;
}

// Theme types
export type ThemeMode = 'light' | 'dark' | 'system';

// Define blog/article interfaces
interface BlogArticle {
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

interface Comment {
  user_id: string;
  comment: string;
  timestamp: string;
}
