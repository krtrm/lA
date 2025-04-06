// API service for communicating with the backend

// Base API URL - change this to match your backend deployment
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types for API requests and responses
interface BaseResponse {
  status?: string;
  error?: string;
}

export interface QueryRequest {
  query: string;
  use_web?: boolean;
}

export interface QueryResponse extends BaseResponse {
  answer: string;
  sources: Array<{
    title: string;
    source: string;
    type: string;
  }>;
  steps: string[];
}

export interface StreamStep {
  type: 'thinking' | 'planning' | 'tool_use' | 'retrieval' | 'generation' | 'complete' | 'error';
  content: string;
  timestamp: number;
  details?: any;
}

interface KeywordExtractionRequest {
  text: string;
}

interface KeywordExtractionResponse extends BaseResponse {
  terms: Record<string, string>;
  count: number;
}

interface ArgumentGenerationRequest {
  topic: string;
  points: string[];
}

interface ArgumentGenerationResponse extends BaseResponse {
  argument: string;
  word_count: number;
  character_count: number;
}

interface OutlineGenerationRequest {
  topic: string;
  doc_type: string;
}

interface OutlineGenerationResponse extends BaseResponse {
  outline: string;
  section_count: number;
  subsection_count: number;
}

interface CitationVerificationRequest {
  citation: string;
}

interface CitationVerificationResponse extends BaseResponse {
  original_citation: string;
  is_valid: boolean;
  corrected_citation?: string;
  summary?: string;
  error_details?: string;
}

// Function to handle API errors
function handleApiError(error: any): never {
  console.error("API Error:", error);
  if (error.response) {
    throw {
      status: error.response.status,
      message: error.response.data?.detail || error.response.data?.error || "An error occurred"
    };
  }
  throw { status: 500, message: error.message || "Network error" };
}

// API methods
export const api = {
  // Basic query endpoint
  async query(request: QueryRequest): Promise<QueryResponse> {
    const controller = new AbortController();
    try {
      console.log("Sending query request:", request);
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw { response: { data: errorData, status: response.status } };
      }
      
      const responseData = await response.json();
      
      // Handle case where response is in {content: xyz} format
      if (responseData && typeof responseData === 'object' && 'content' in responseData) {
        return {
          answer: responseData.content,
          sources: responseData.sources || [],
          steps: responseData.steps || []
        };
      }
      
      return responseData;
    } catch (error) {
      throw handleApiError(error);
    } finally {
      controller.abort();
    }
  },

  // Streaming query endpoint
  async streamQuery(request: QueryRequest, onStep: (step: StreamStep) => void): Promise<() => void> {
    const controller = new AbortController();
    
    try {
      const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...request,
          stream_thinking: true
        }),
        signal: controller.signal
      });
      
      // Add streaming timeout
      const timeout = setTimeout(() => controller.abort(), 30000);
      
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body is not readable');
      
      const decoder = new TextDecoder();
      let buffer = '';
      
      const processStream = async () => {
        // Process the stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          // Process chunks in batches
          buffer += decoder.decode(value, { stream: true });
          const batches = buffer.split('\n\n');  // Batch multiple steps
          buffer = batches.pop() || '';
          
          for (const batch of batches) {
            await new Promise(r => setTimeout(r, 0)); // Yield to UI thread
            const lines = batch.split('\n');
            for (const line of lines) {
              if (line.trim()) {
                try {
                  const step = JSON.parse(line) as StreamStep;
                  onStep(step);
                } catch (e) {
                  console.error('Error parsing stream data:', e);
                }
              }
            }
          }
        }
        clearTimeout(timeout);
      };
      
      processStream();
      return () => controller.abort();
      
    } catch (error) {
      onStep({
        type: 'error',
        content: handleApiError(error).message,
        timestamp: Date.now()
      });
    }
  },

  // Additional API endpoints
  async extractKeywords(text: string): Promise<KeywordExtractionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/extract_keywords`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw { response: { data: errorData, status: response.status } };
      }
      
      return await response.json();
    } catch (error) {
      throw handleApiError(error);
    }
  },

  async generateArgument(topic: string, points: string[]): Promise<ArgumentGenerationResponse> {
    try {
      console.log("Generating argument with params:", { topic, points });
      const response = await fetch(`${API_BASE_URL}/generate_argument`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic, points }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw { response: { data: errorData, status: response.status } };
      }
      
      const responseData = await response.json();
      
      // Handle case where response is in {content: xyz} format
      if (responseData && typeof responseData === 'object' && 'content' in responseData) {
        return {
          status: "success",
          argument: responseData.content,
          word_count: responseData.content.split(/\s+/).length,
          character_count: responseData.content.length
        };
      }
      
      return responseData;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  async createOutline(topic: string, doc_type: string): Promise<OutlineGenerationResponse> {
    try {
      console.log("Creating outline with params:", { topic, doc_type });
      const response = await fetch(`${API_BASE_URL}/create_outline`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic, doc_type }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw { response: { data: errorData, status: response.status } };
      }
      
      const responseData = await response.json();
      
      // Handle case where response is in {content: xyz} format
      if (responseData && typeof responseData === 'object' && 'content' in responseData) {
        return {
          status: "success",
          outline: responseData.content,
          section_count: 0, // Cannot determine from content field alone
          subsection_count: 0
        };
      }
      
      return responseData;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  async verifyCitation(citation: string): Promise<CitationVerificationResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/verify_citation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ citation }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw { response: { data: errorData, status: response.status } };
      }
      
      return await response.json();
    } catch (error) {
      throw handleApiError(error);
    }
  }
};
