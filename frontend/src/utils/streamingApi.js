/**
 * Utility functions for consuming the streaming API endpoints
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
