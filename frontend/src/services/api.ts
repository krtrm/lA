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
        const errorData = await response.json().catch(() => ({}));
        throw { 
          status: response.status, 
          message: errorData.message || `Error: ${response.status} ${response.statusText}` 
        };
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("API query error:", error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw { status: 499, message: "Request was cancelled" };
      }
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
        content: error instanceof Error ? error.message : String(error),
        timestamp: Date.now()
      });
      return () => controller.abort();
    }
  }
};
