import os
import logging
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from init_rag import init_rag_system

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Vaqeel.app API", description="Legal AI assistant for Indian law")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system with existing Pinecone index
index_name = "legal-documents"
rag_system = init_rag_system(index_name=index_name, namespace="indian-law")

if not rag_system:
    logger.error(f"Failed to initialize RAG system with index '{index_name}'")

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    use_web: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def root():
    return {"message": "Welcome to the Vaqeel.app Legal AI API"}

@app.post("/query", response_model=QueryResponse)
async def query_legal_ai(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    
    try:
        # Query the RAG system
        result = await rag_system.query(request.query, use_web=request.use_web)
        
        # Extract answer
        answer = result.content if hasattr(result, 'content') else str(result)
        
        # Extract sources from the documents if available
        sources = []
        try:
            if hasattr(result, '_source_documents'):
                for doc in result._source_documents[:5]:  # Limit to top 5 sources
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.append({
                            'title': doc.metadata.get('title', 'Unknown'),
                            'source': doc.metadata['source'],
                            'type': doc.metadata.get('type', 'document')
                        })
        except Exception as e:
            logger.warning(f"Error extracting sources: {e}")
        
        return QueryResponse(answer=answer, sources=sources)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vaqeel.app API server on http://0.0.0.0:8000")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
