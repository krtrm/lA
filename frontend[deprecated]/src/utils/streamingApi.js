/**
 * Utility functions for consuming the streaming API endpoints
 */

// Update to use Vite environment variable syntax
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Process a streaming response from the API
 * 
 * @param {string} endpoint - API endpoint to call
 * @param {Object} data - Request data to send
 * @param {Function} onStep - Callback for each step (takes step object as parameter)
 * @param {Function} onComplete - Callback when all steps are received (takes array of all steps)
 * @param {Function} onError - Callback for errors
 */
export const fetchStreaming = async (
  endpoint,
  data,
  onStep,
  onComplete,
  onError
) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const allSteps = [];
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        // Process any remaining data in the buffer
        if (buffer.trim()) {
          try {
            const step = JSON.parse(buffer.trim());
            allSteps.push(step);
            onStep(step);
          } catch (e) {
            console.error('Error parsing final JSON chunk:', e);
          }
        }
        break;
      }
      
      // Decode the chunk and add to buffer
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      
      // Process complete lines in the buffer
      const lines = buffer.split('\n');
      // Keep the last potentially incomplete line in the buffer
      buffer = lines.pop();
      
      // Process each complete line
      for (const line of lines) {
        if (line.trim()) {
          try {
            const step = JSON.parse(line);
            allSteps.push(step);
            onStep(step);
            
            // If this is a completion or error step, we're done
            if (step.type === 'complete' || step.type === 'error') {
              onComplete(allSteps, step);
            }
          } catch (e) {
            console.error('Error parsing JSON:', e, 'Line:', line);
          }
        }
      }
    }
    
    onComplete(allSteps);
  } catch (error) {
    console.error('Streaming API error:', error);
    onError(error);
  }
};

/**
 * Examples showing how to use the streaming API
 */
export const streamingApiExamples = {
  // Stream a query response
  query: (query, onStep, onComplete, onError) => {
    return fetchStreaming(
      '/query/stream',
      { query, stream_thinking: true },
      onStep,
      onComplete,
      onError
    );
  },
  
  // Stream keyword extraction
  extractKeywords: (text, onStep, onComplete, onError) => {
    return fetchStreaming(
      '/extract_keywords/stream',
      { text },
      onStep,
      onComplete,
      onError
    );
  },
  
  // Stream argument generation
  generateArgument: (topic, points, onStep, onComplete, onError) => {
    return fetchStreaming(
      '/generate_argument/stream',
      { topic, points },
      onStep,
      onComplete,
      onError
    );
  },
  
  // Stream outline creation
  createOutline: (topic, doc_type, onStep, onComplete, onError) => {
    return fetchStreaming(
      '/create_outline/stream',
      { topic, doc_type },
      onStep,
      onComplete,
      onError
    );
  },
  
  // Stream citation verification
  verifyCitation: (citation, onStep, onComplete, onError) => {
    return fetchStreaming(
      '/verify_citation/stream',
      { citation },
      onStep,
      onComplete,
      onError
    );
  }
};

export default streamingApiExamples;

/**
 * Utility functions for handling server-sent events and streaming responses in chat
 */

/**
 * Reads streaming chunks from a streaming response and calls the callback for each chunk
 * @param {Response} response - The fetch Response object from a streaming endpoint
 * @param {Function} onChunk - Callback that receives each text chunk as it arrives
 * @param {Function} onComplete - Optional callback that is called when the stream completes
 * @param {Function} onError - Optional callback for handling errors
 * @returns {Promise<void>} - Promise that resolves when the stream is complete
 */
export async function readStreamChunks(
  response, 
  onChunk, 
  onComplete = () => {}, 
  onError = (err) => console.error('Stream error:', err)
) {
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Stream request failed: ${response.status} ${errorText}`);
  }

  if (!response.body) {
    throw new Error('Response has no body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        onComplete();
        break;
      }
      
      // Convert Uint8Array to string
      const chunk = decoder.decode(value, { stream: true });
      
      // Process the chunk
      onChunk(chunk);
    }
  } catch (error) {
    onError(error);
    throw error;
  }
}

/**
 * Parses a stream of server-sent events (SSE)
 * @param {string} chunk - A chunk of text from an SSE stream
 * @returns {Array} - Array of parsed events with { event, data, id } properties
 */
export function parseSSEChunk(chunk) {
  const events = [];
  const lines = chunk.split('\n');
  
  let event = {};
  
  for (const line of lines) {
    if (line.trim() === '') {
      // Empty line denotes the end of an event
      if (Object.keys(event).length > 0) {
        events.push({ ...event });
        event = {};
      }
      continue;
    }
    
    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) continue;
    
    const field = line.slice(0, colonIndex).trim();
    const value = line.slice(colonIndex + 1).trim();
    
    if (field === 'event') {
      event.event = value;
    } else if (field === 'data') {
      event.data = value;
    } else if (field === 'id') {
      event.id = value;
    }
  }
  
  // Push the last event if there's no trailing newline
  if (Object.keys(event).length > 0) {
    events.push(event);
  }
  
  return events;
}

/**
 * Creates a formatted chat message object
 * @param {string} content - Message content
 * @param {'user'|'assistant'} role - Message role (user or assistant)
 * @param {number} id - Message ID
 * @returns {Object} - Formatted message object
 */
export function createChatMessage(content, role = 'assistant', id = Date.now()) {
  return {
    id,
    role,
    content,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };
}

/**
 * Handle a streaming AI response and update the UI progressively
 * @param {Response} response - The streaming response
 * @param {Function} onUpdate - Callback with the current content string as parameter
 * @param {Function} onComplete - Callback when streaming is complete
 */
export async function handleStreamingResponse(response, onUpdate, onComplete) {
  let contentSoFar = '';
  
  await readStreamChunks(
    response,
    (chunk) => {
      // Simple case: just append text without parsing SSE
      contentSoFar += chunk;
      onUpdate(contentSoFar);
    },
    () => {
      // When stream completes
      onComplete(contentSoFar);
    }
  );
  
  return contentSoFar;
}
