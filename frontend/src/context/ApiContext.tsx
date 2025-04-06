import React, { createContext, useContext, ReactNode } from 'react';
import { api, QueryRequest, QueryResponse, StreamStep } from '../services/api';

// Define the context shape
type ApiContextType = {
  query: (request: QueryRequest) => Promise<QueryResponse>;
  streamQuery: (request: QueryRequest, onStep: (step: StreamStep) => void) => Promise<void>;
  extractKeywords: (text: string) => Promise<any>;
  generateArgument: (topic: string, points: string[]) => Promise<any>;
  createOutline: (topic: string, doc_type: string) => Promise<any>;
  verifyCitation: (citation: string) => Promise<any>;
};

// Create the context
const ApiContext = createContext<ApiContextType | undefined>(undefined);

// Hook for using the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

// Provider component
export const ApiProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // The value for the context is just the API methods
  const value = {
    query: api.query,
    streamQuery: api.streamQuery,
    extractKeywords: api.extractKeywords,
    generateArgument: api.generateArgument,
    createOutline: api.createOutline,
    verifyCitation: api.verifyCitation,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};
