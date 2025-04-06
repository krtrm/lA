import os
import logging
import asyncio
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Added security middlewares
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from init_rag import init_rag_system
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Literal
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from functools import lru_cache

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

# Add TrustedHostMiddleware but with more permissive settings for development
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Allow all hosts during development, restrict in production
)

# Only use HTTPS redirect in production environments
if os.getenv("ENVIRONMENT", "development").lower() == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
    logger.info("HTTPS redirect middleware enabled (production mode)")
else:
    logger.info("HTTPS redirect middleware disabled (development mode)")

# Initialize RAG system with correct Pinecone index name
# Use the actual available index from logs: llama-text-embed-v2-index
index_name = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")
namespace = os.getenv("PINECONE_NAMESPACE", "indian-law")

logger.info(f"Initializing RAG system with index: {index_name}, namespace: {namespace}")
rag_system = init_rag_system(index_name=index_name, namespace=namespace)

if not rag_system:
    logger.error(f"Failed to initialize RAG system with index '{index_name}'")

# Add connection pool for RAG system
@asynccontextmanager
async def get_rag_context():
    try:
        yield rag_system
    finally:
        # Safe cleanup that doesn't rely on client.close()
        if hasattr(rag_system, 'vector_store'):
            # For new Pinecone SDK, there's no explicit client.close() method needed
            # Just make sure we don't maintain any unnecessary references
            pass

# Add caching for frequent operations
@lru_cache(maxsize=1024)
async def cached_rag_query(query: str):
    return await rag_system.query(query)

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    use_web: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    steps: list = []  # Added steps field for process feedback

# New request and response models for specialized features
class KeywordExtractionRequest(BaseModel):
    text: str

class KeywordExtractionResponse(BaseModel):
    status: str
    terms: Dict[str, str]
    count: int = 0
    error: Optional[str] = None

class ArgumentGenerationRequest(BaseModel):
    topic: str
    points: List[str]

class ArgumentGenerationResponse(BaseModel):
    status: str
    argument: str
    word_count: int = 0
    character_count: int = 0
    error: Optional[str] = None

class OutlineGenerationRequest(BaseModel):
    topic: str
    doc_type: str

class OutlineGenerationResponse(BaseModel):
    status: str
    outline: str
    section_count: int = 0
    subsection_count: int = 0
    error: Optional[str] = None

class CitationVerificationRequest(BaseModel):
    citation: str

class CitationVerificationResponse(BaseModel):
    status: str
    original_citation: str
    is_valid: bool
    corrected_citation: Optional[str] = None
    summary: Optional[str] = None
    error_details: Optional[str] = None
    error: Optional[str] = None

# New streaming response models
class StreamStep(BaseModel):
    type: Literal["thinking", "planning", "tool_use", "retrieval", "generation", "complete", "error"]
    content: str
    timestamp: float = 0.0
    details: Optional[Dict[str, Any]] = None

class StreamingQueryRequest(QueryRequest):
    stream_thinking: bool = True  # Whether to include thinking steps in stream

# Define streaming response format
def stream_response_generator(steps_generator):
    """Convert an async generator of steps into a proper streaming response"""
    async def generate():
        try:
            async for step in steps_generator:
                yield json.dumps(step.dict()) + "\n"
        except Exception as e:
            error_step = StreamStep(
                type="error",
                content=f"Error during streaming: {str(e)}",
                timestamp=time.time()
            )
            yield json.dumps(error_step.dict()) + "\n"
    
    return generate()

@app.get("/")
async def root():
    return {"message": "Welcome to the Vaqeel.app Legal AI API"}

# Update query endpoint with connection pooling
@app.post("/query", response_model=QueryResponse)
async def query_legal_ai(request: QueryRequest):
    async with get_rag_context() as rag:
        steps = []
        steps.append("Starting query processing")
        if not rag_system:
            steps.append("RAG system not initialized")
            raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
        try:
            steps.append("Querying the RAG system")
            # Query the RAG system
            result = await rag.query(request.query, use_web=request.use_web)
            steps.append("RAG returned a result")
            
            # Extract answer
            answer = result.content if hasattr(result, 'content') else str(result)
            steps.append("Extracted the answer")
            
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
                    steps.append("Extracted sources from documents")
            except Exception as e:
                steps.append("Warning: error extracting sources")
                logger.warning(f"Error extracting sources: {e}")
            
            steps.append("Completed query processing")
            return QueryResponse(answer=answer, sources=sources, steps=steps)
        
        except Exception as e:
            steps.append("Error processing query")
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Streaming version of query endpoint
@app.post("/query/stream")
async def stream_query_legal_ai(request: StreamingQueryRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Please ensure the vector database is set up."}
        )
    
    import time
    
    # Modify streaming generator with backpressure control
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Add yield point for event loop
            await asyncio.sleep(0)
            # Initial step - thinking
            yield StreamStep(
                type="thinking",
                content="Analyzing your query...",
                timestamp=time.time()
            )
            
            # Check if tools are needed
            needs_tools = False
            try:
                needs_tools = await rag_system._needs_tools(request.query)
                yield StreamStep(
                    type="planning",
                    content=f"Determined that {'external tools are' if needs_tools else 'no external tools are'} needed for this query.",
                    timestamp=time.time(),
                    details={"needs_tools": needs_tools}
                )
            except Exception as e:
                yield StreamStep(
                    type="error",
                    content=f"Error determining tool necessity: {str(e)}",
                    timestamp=time.time()
                )
                needs_tools = len(request.query.split()) > 5  # Fallback
            
            # If simple query, respond directly
            if not needs_tools:
                yield StreamStep(
                    type="generation",
                    content="Generating response directly without tools...",
                    timestamp=time.time()
                )
                response = await rag_system.generate_simple_response(request.query)
                yield StreamStep(
                    type="complete",
                    content=response.get("content", "I'm not sure how to respond to that."),
                    timestamp=time.time()
                )
                return
            
            # Get the plan
            plan = await rag_system.tool_manager.create_plan(request.query)
            yield StreamStep(
                type="planning",
                content=f"Created plan: {plan['plan']}",
                timestamp=time.time(),
                details={"plan": plan}
            )
            
            # Tool execution phase
            all_results = []
            tool_tasks = []
            for i, tool_step in enumerate(plan.get("tools", [])):
                tool_name = tool_step.get("tool")
                parameters = tool_step.get("parameters", {})
                reason = tool_step.get("reason", "No reason provided")
                
                yield StreamStep(
                    type="tool_use",
                    content=f"Using tool: {tool_name} - {reason}",
                    timestamp=time.time(),
                    details={"tool": tool_name, "parameters": parameters, "step": i+1, "total_steps": len(plan.get("tools", []))}
                )
                
                tool_tasks.append(rag_system.tool_manager.execute_tool(tool_name, **parameters))
            
            # Process tools in parallel with timeout
            for result in await asyncio.gather(*tool_tasks, return_exceptions=True):
                await asyncio.sleep(0)  # Yield control
                if result.get("status") == "success":
                    results = result.get("results", [])
                    if results:
                        all_results.extend(results)
                        result_count = len(results)
                        yield StreamStep(
                            type="retrieval",
                            content=f"Found {result_count} results from {tool_name}",
                            timestamp=time.time(),
                            details={"source": tool_name, "count": result_count}
                        )
            
            # Convert to documents
            yield StreamStep(
                type="generation",
                content="Generating response based on retrieved information...",
                timestamp=time.time(),
                details={"source_count": len(all_results)}
            )
            
            # Generate final answer
            documents = []
            for result in all_results:
                if not result.get("content"):
                    continue
                
                from langchain_core.documents import Document
                documents.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "domain": result.get("domain", ""),
                            "type": result.get("type", "unknown")
                        }
                    )
                )
            
            # Manage token count
            token_budget = rag_system.max_input_tokens
            question_tokens = rag_system._count_tokens(request.query)
            token_budget -= question_tokens
            
            # Reserve tokens for the model's response and prompt
            token_budget -= 4000  # Rough estimate for prompt + response
            
            # Select documents to include within token budget
            docs_for_context = rag_system._select_docs_within_budget(documents, token_budget)
            
            # Generate response
            result = await rag_system.qa_chain.ainvoke({
                "input": request.query,
                "context": docs_for_context
            })
            
            # Return final answer
            yield StreamStep(
                type="complete",
                content=result.get("content", "I couldn't find a good answer based on the information available."),
                timestamp=time.time(),
                details={"sources": [
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "source": doc.metadata.get("source", ""),
                        "type": doc.metadata.get("type", "document")
                    } for doc in docs_for_context[:5]  # Limit to top 5
                ]}
            )
        
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield StreamStep(
                type="error",
                content=f"Error processing your query: {str(e)}",
                timestamp=time.time()
            )
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

# New endpoints for specialized features
@app.post("/extract_keywords", response_model=KeywordExtractionResponse)
async def extract_legal_keywords(request: KeywordExtractionRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    try:
        logger.info(f"Extracting keywords from text of length {len(request.text)}")
        result = await rag_system.extract_legal_keywords(request.text)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
        return KeywordExtractionResponse(
            status=result.get("status", "success"),
            terms=result.get("terms", {}),
            count=result.get("count", 0)
        )
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return KeywordExtractionResponse(
            status="error",
            terms={},
            count=0,
            error=str(e)
        )

# Streaming version of keyword extraction
@app.post("/extract_keywords/stream")
async def stream_extract_legal_keywords(request: KeywordExtractionRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Please ensure the vector database is set up."}
        )
    
    import time
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Initial thinking step
            yield StreamStep(
                type="thinking",
                content=f"Analyzing text of length {len(request.text)} to extract legal terms...",
                timestamp=time.time()
            )
            
            # Retrieval step - getting relevant context
            yield StreamStep(
                type="retrieval",
                content="Retrieving relevant legal context to help define terms...",
                timestamp=time.time()
            )
            
            # Process step
            yield StreamStep(
                type="generation",
                content="Extracting and defining legal terms...",
                timestamp=time.time()
            )
            
            # Actual processing
            result = await rag_system.extract_legal_keywords(request.text)
            
            if result.get("status") == "error":
                yield StreamStep(
                    type="error",
                    content=f"Error: {result.get('error', 'Unknown error')}",
                    timestamp=time.time()
                )
            else:
                terms = result.get("terms", {})
                yield StreamStep(
                    type="complete",
                    content=f"Extracted {len(terms)} legal terms",
                    timestamp=time.time(),
                    details={"terms": terms, "count": len(terms)}
                )
        
        except Exception as e:
            logger.error(f"Error in streaming keyword extraction: {e}")
            yield StreamStep(
                type="error",
                content=f"Error extracting keywords: {str(e)}",
                timestamp=time.time()
            )
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

@app.post("/generate_argument", response_model=ArgumentGenerationResponse)
async def generate_legal_argument(request: ArgumentGenerationRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    try:
        logger.info(f"Generating legal argument for topic: {request.topic}")
        result = await rag_system.generate_legal_argument(request.topic, request.points)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
        return ArgumentGenerationResponse(
            status=result.get("status", "success"),
            argument=result.get("argument", ""),
            word_count=result.get("word_count", 0),
            character_count=result.get("character_count", 0)
        )
    except Exception as e:
        logger.error(f"Error generating argument: {e}")
        return ArgumentGenerationResponse(
            status="error",
            argument="",
            error=str(e)
        )

# Streaming version of argument generation
@app.post("/generate_argument/stream")
async def stream_generate_legal_argument(request: ArgumentGenerationRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Please ensure the vector database is set up."}
        )
    
    import time
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Initial thinking step
            yield StreamStep(
                type="thinking",
                content=f"Planning legal argument on topic: {request.topic}...",
                timestamp=time.time(),
                details={"topic": request.topic, "points_count": len(request.points)}
            )
            
            # Retrieval step
            yield StreamStep(
                type="retrieval",
                content="Retrieving relevant legal information and precedents...",
                timestamp=time.time()
            )
            
            # Generation step
            yield StreamStep(
                type="generation",
                content="Crafting structured legal argument...",
                timestamp=time.time()
            )
            
            # Actual processing
            result = await rag_system.generate_legal_argument(request.topic, request.points)
            
            if result.get("status") == "error":
                yield StreamStep(
                    type="error",
                    content=f"Error: {result.get('error', 'Unknown error')}",
                    timestamp=time.time()
                )
            else:
                argument = result.get("argument", "")
                yield StreamStep(
                    type="complete",
                    content=argument,
                    timestamp=time.time(),
                    details={
                        "word_count": result.get("word_count", 0),
                        "character_count": result.get("character_count", 0)
                    }
                )
        
        except Exception as e:
            logger.error(f"Error in streaming argument generation: {e}")
            yield StreamStep(
                type="error",
                content=f"Error generating argument: {str(e)}",
                timestamp=time.time()
            )
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

@app.post("/create_outline", response_model=OutlineGenerationResponse)
async def create_document_outline(request: OutlineGenerationRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    try:
        logger.info(f"Creating outline for {request.doc_type} about: {request.topic}")
        result = await rag_system.create_document_outline(request.topic, request.doc_type)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
        return OutlineGenerationResponse(
            status=result.get("status", "success"),
            outline=result.get("outline", ""),
            section_count=result.get("section_count", 0),
            subsection_count=result.get("subsection_count", 0)
        )
    except Exception as e:
        logger.error(f"Error creating document outline: {e}")
        return OutlineGenerationResponse(
            status="error",
            outline="",
            error=str(e)
        )

# Streaming version of document outline creation
@app.post("/create_outline/stream")
async def stream_create_document_outline(request: OutlineGenerationRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Please ensure the vector database is set up."}
        )
    
    import time
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Initial thinking step
            yield StreamStep(
                type="thinking",
                content=f"Planning outline for {request.doc_type} on topic: {request.topic}...",
                timestamp=time.time(),
                details={"doc_type": request.doc_type, "topic": request.topic}
            )
            
            # Retrieval step
            yield StreamStep(
                type="retrieval",
                content=f"Researching standard structure for {request.doc_type} documents...",
                timestamp=time.time()
            )
            
            # Generation step
            yield StreamStep(
                type="generation",
                content="Creating document outline...",
                timestamp=time.time()
            )
            
            # Actual processing
            result = await rag_system.create_document_outline(request.topic, request.doc_type)
            
            if result.get("status") == "error":
                yield StreamStep(
                    type="error",
                    content=f"Error: {result.get('error', 'Unknown error')}",
                    timestamp=time.time()
                )
            else:
                outline = result.get("outline", "")
                yield StreamStep(
                    type="complete",
                    content=outline,
                    timestamp=time.time(),
                    details={
                        "section_count": result.get("section_count", 0),
                        "subsection_count": result.get("subsection_count", 0)
                    }
                )
        
        except Exception as e:
            logger.error(f"Error in streaming outline creation: {e}")
            yield StreamStep(
                type="error",
                content=f"Error creating outline: {str(e)}",
                timestamp=time.time()
            )
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

@app.post("/verify_citation", response_model=CitationVerificationResponse)
async def verify_legal_citation(request: CitationVerificationRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    try:
        logger.info(f"Verifying citation: {request.citation}")
        result = await rag_system.verify_citation(request.citation)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
        return CitationVerificationResponse(
            status=result.get("status", "success"),
            original_citation=result.get("original_citation", request.citation),
            is_valid=result.get("is_valid", False),
            corrected_citation=result.get("corrected_citation"),
            summary=result.get("summary"),
            error_details=result.get("error_details")
        )
    except Exception as e:
        logger.error(f"Error verifying citation: {e}")
        return CitationVerificationResponse(
            status="error",
            original_citation=request.citation,
            is_valid=False,
            error=str(e)
        )

# Streaming version of citation verification
@app.post("/verify_citation/stream")
async def stream_verify_legal_citation(request: CitationVerificationRequest):
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Please ensure the vector database is set up."}
        )
    
    import time
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Initial thinking step
            yield StreamStep(
                type="thinking",
                content=f"Analyzing citation: {request.citation}...",
                timestamp=time.time()
            )
            
            # Tool use step - checking Indian Kanoon
            yield StreamStep(
                type="tool_use",
                content="Searching Indian Kanoon for citation details...",
                timestamp=time.time(),
                details={"tool": "indian_kanoon_search", "citation": request.citation}
            )
            
            # Retrieval step
            yield StreamStep(
                type="retrieval",
                content="Retrieving relevant information about the citation...",
                timestamp=time.time()
            )
            
            # Analysis step
            yield StreamStep(
                type="generation",
                content="Verifying citation format and details...",
                timestamp=time.time()
            )
            
            # Actual processing
            result = await rag_system.verify_citation(request.citation)
            
            if result.get("status") == "error":
                yield StreamStep(
                    type="error",
                    content=f"Error: {result.get('error', 'Unknown error')}",
                    timestamp=time.time()
                )
            else:
                is_valid = result.get("is_valid", False)
                corrected = result.get("corrected_citation", "")
                
                response_content = (
                    f"The citation {'is valid' if is_valid else 'is not valid'}. "
                    f"{f'Corrected citation: {corrected}' if corrected and not is_valid else ''}"
                )
                
                yield StreamStep(
                    type="complete",
                    content=response_content,
                    timestamp=time.time(),
                    details={
                        "original_citation": result.get("original_citation", request.citation),
                        "is_valid": is_valid,
                        "corrected_citation": corrected,
                        "summary": result.get("summary", ""),
                        "error_details": result.get("error_details", "")
                    }
                )
        
        except Exception as e:
            logger.error(f"Error in streaming citation verification: {e}")
            yield StreamStep(
                type="error",
                content=f"Error verifying citation: {str(e)}",
                timestamp=time.time()
            )
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

# Modify server config at bottom
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vaqeel.app API server on http://0.0.0.0:8000")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            timeout_keep_alive=120,  # Keep connections alive longer
            limit_concurrency=100     # Prevent overloading
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

from datetime import datetime, timedelta
from database import db

# User authentication models
class UserAuthRequest(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str

# User data models
class SpaceRequest(BaseModel):
    title: str
    type: str

class MessageRequest(BaseModel):
    space_id: int
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# User authentication endpoint
@app.post("/user/auth")
async def authenticate_user(request: UserAuthRequest):
    """Authenticate or create a user"""
    try:
        user = db.get_or_create_user(
            request.user_id,
            request.email,
            request.first_name,
            request.last_name
        )
        
        # Log login action
        db.log_user_action(request.user_id, "login")
        
        if not user:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to authenticate user"}
            )
        
        return {
            "status": "success",
            "user": user
        }
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Authentication error: {str(e)}"}
        )

# User spaces endpoints
@app.post("/user/spaces")
async def create_user_space(request: SpaceRequest, user_auth: UserAuthRequest):
    """Create a new space for the user"""
    try:
        space = db.create_space(user_auth.user_id, request.title, request.type)
        
        if not space:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create space"}
            )
        
        # Log space creation
        db.log_user_action(user_auth.user_id, "create_space", {"space_id": space["space_id"]})
        
        return {
            "status": "success",
            "space": space
        }
    except Exception as e:
        logger.error(f"Error creating space: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error creating space: {str(e)}"}
        )

@app.get("/user/spaces")
async def get_user_spaces(user_id: str, limit: int = 10, offset: int = 0):
    """Get all spaces for a user"""
    try:
        spaces = db.get_user_spaces(user_id, limit, offset)
        
        return {
            "status": "success",
            "spaces": spaces,
            "count": len(spaces)
        }
    except Exception as e:
        logger.error(f"Error getting spaces: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting spaces: {str(e)}"}
        )

@app.get("/spaces/{space_id}")
async def get_space(space_id: int):
    """Get a space by ID"""
    try:
        space = db.get_space(space_id)
        
        if not space:
            return JSONResponse(
                status_code=404,
                content={"error": "Space not found"}
            )
        
        # Get messages for this space
        messages = db.get_space_messages(space_id)
        space["messages"] = messages
        
        return {
            "status": "success",
            "space": space
        }
    except Exception as e:
        logger.error(f"Error getting space: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting space: {str(e)}"}
        )

# Message endpoints
@app.post("/spaces/{space_id}/messages")
async def add_message(space_id: int, request: MessageRequest):
    """Add a message to a space"""
    try:
        # Verify space exists
        space = db.get_space(space_id)
        if not space:
            return JSONResponse(
                status_code=404,
                content={"error": "Space not found"}
            )
        
        message = db.add_message(space_id, request.role, request.content, request.metadata)
        
        if not message:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to add message"}
            )
        
        # Log message action
        db.log_user_action(space["user_id"], "send_message", {
            "space_id": space_id,
            "message_id": message["message_id"],
            "role": request.role
        })
        
        return {
            "status": "success",
            "message": message
        }
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error adding message: {str(e)}"}
        )

@app.get("/spaces/{space_id}/messages")
async def get_space_messages(space_id: int):
    """Get all messages for a space"""
    try:
        # Verify space exists
        space = db.get_space(space_id)
        if not space:
            return JSONResponse(
                status_code=404,
                content={"error": "Space not found"}
            )
        
        messages = db.get_space_messages(space_id)
        
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting messages: {str(e)}"}
        )

# User stats endpoint
@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get usage statistics for a user"""
    try:
        stats = db.get_user_stats(user_id)
        
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting user stats: {str(e)}"}
        )

# AI chat processing - Extended to work with database
@app.post("/spaces/{space_id}/chat")
async def process_chat_message(space_id: int, request: QueryRequest, user_id: str):
    """Process a chat message and store in database"""
    try:
        # Verify space exists
        space = db.get_space(space_id)
        if not space:
            return JSONResponse(
                status_code=404,
                content={"error": "Space not found"}
            )
        
        # Verify user owns this space
        if space["user_id"] != user_id:
            return JSONResponse(
                status_code=403,
                content={"error": "You don't have permission to access this space"}
            )
        
        # Store user message
        user_message = db.add_message(space_id, "user", request.query)
        if not user_message:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save user message"}
            )
        
        # Process with RAG system
        if not rag_system:
            return JSONResponse(
                status_code=503,
                content={"error": "RAG system not initialized"}
            )
        
        # Handle based on space type
        space_type = space["type"]
        response = None
        
        try:
            if space_type == "legal_research":
                response = await rag_system.query(request.query, use_web=request.use_web)
            elif space_type == "document_drafting":
                response = await rag_system.query(request.query)
            elif space_type == "legal_analysis":
                response = await rag_system.query(request.query)
            elif space_type == "citation_verification":
                # If it looks like a citation, use verify endpoint, otherwise use query
                if re.match(r'^[A-Z]+\s+\d{4}\s+[A-Z]+', request.query):
                    response = await rag_system.verify_citation(request.query)
                else:
                    response = await rag_system.query(request.query)
            elif space_type == "statute_interpretation":
                response = await rag_system.query(request.query, use_web=True)
            else:
                response = await rag_system.query(request.query)
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            # Store error message
            error_message = db.add_message(
                space_id, 
                "assistant", 
                f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
                {"error": str(e)}
            )
            return {
                "status": "error",
                "message": error_message,
                "error": str(e)
            }
        
        # Extract content from response based on type
        answer_text = ""
        if isinstance(response, str):
            answer_text = response
        elif response and isinstance(response, dict):
            if "content" in response:
                answer_text = response["content"]
            elif "answer" in response:
                answer_text = response["answer"]
            elif "outline" in response:
                answer_text = response["outline"]
            elif "argument" in response:
                answer_text = response["argument"]
            elif "is_valid" in response:
                answer_text = f"The citation {response['is_valid'] and 'is valid' or 'is not valid'}. "
                if response.get("corrected_citation") and not response["is_valid"]:
                    answer_text += f"Suggested correction: {response['corrected_citation']} "
                if response.get("summary"):
                    answer_text += f"\n\n{response['summary']}"
            else:
                answer_text = json.dumps(response, indent=2)
        else:
            answer_text = "I processed your request, but received an unexpected response format."
        
        # Store assistant message
        assistant_message = db.add_message(space_id, "assistant", answer_text)
        
        if not assistant_message:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save assistant message"}
            )
        
        # Log chat completion
        db.log_user_action(user_id, "chat_completion", {
            "space_id": space_id,
            "query_length": len(request.query),
            "response_length": len(answer_text)
        })
        
        return {
            "status": "success",
            "user_message": user_message,
            "assistant_message": assistant_message
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing chat: {str(e)}"}
        )

import requests
import re
from datetime import datetime, timedelta

# Blog related models and endpoints
class BlogPostRequest(BaseModel):
    title: str
    content: str
    summary: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    published: Optional[bool] = True

class BlogCommentRequest(BaseModel):
    content: str

@app.post("/blogs")
async def create_blog_post(request: BlogPostRequest, user_id: str):
    """Create a new blog post"""
    try:
        blog = db.create_blog(
            user_id=user_id,
            title=request.title,
            content=request.content,
            summary=request.summary,
            category=request.category,
            tags=request.tags
        )
        
        if not blog:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create blog post"}
            )
        
        # Log blog creation
        db.log_user_action(user_id, "create_blog", {"blog_id": blog["blog_id"]})
        
        return {
            "status": "success",
            "blog": blog
        }
    except Exception as e:
        logger.error(f"Error creating blog post: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error creating blog post: {str(e)}"}
        )

@app.get("/blogs")
async def get_blog_posts(
    limit: int = 10, 
    offset: int = 0, 
    category: Optional[str] = None,
    user_id: Optional[str] = None,
    is_official: Optional[bool] = None,
    published_only: bool = True
):
    """Get blog posts with optional filtering"""
    try:
        blogs = db.get_blogs(
            limit=limit,
            offset=offset,
            category=category,
            user_id=user_id,
            is_official=is_official,
            published_only=published_only
        )
        
        return {
            "status": "success",
            "blogs": blogs,
            "count": len(blogs)
        }
    except Exception as e:
        logger.error(f"Error getting blog posts: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting blog posts: {str(e)}"}
        )

@app.get("/blogs/{blog_id}")
async def get_blog_post(blog_id: int):
    """Get a blog post by ID"""
    try:
        blog = db.get_blog(blog_id)
        
        if not blog:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found"}
            )
        
        # Get comments for this blog
        comments = db.get_blog_comments(blog_id)
        blog["comments"] = comments
        
        return {
            "status": "success",
            "blog": blog
        }
    except Exception as e:
        logger.error(f"Error getting blog post: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting blog post: {str(e)}"}
        )

@app.put("/blogs/{blog_id}")
async def update_blog_post(blog_id: int, request: BlogPostRequest, user_id: str):
    """Update a blog post"""
    try:
        # Create dict of updates
        updates = request.dict(exclude_unset=True)
        
        blog = db.update_blog(blog_id, user_id, updates)
        
        if not blog:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found or you don't have permission to edit it"}
            )
        
        # Log blog update
        db.log_user_action(user_id, "update_blog", {"blog_id": blog_id})
        
        return {
            "status": "success",
            "blog": blog
        }
    except Exception as e:
        logger.error(f"Error updating blog post: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error updating blog post: {str(e)}"}
        )

@app.delete("/blogs/{blog_id}")
async def delete_blog_post(blog_id: int, user_id: str):
    """Delete a blog post"""
    try:
        success = db.delete_blog(blog_id, user_id)
        
        if not success:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found or you don't have permission to delete it"}
            )
        
        # Log blog deletion
        db.log_user_action(user_id, "delete_blog", {"blog_id": blog_id})
        
        return {
            "status": "success",
            "message": "Blog post deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting blog post: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting blog post: {str(e)}"}
        )

@app.post("/blogs/{blog_id}/comments")
async def add_blog_comment(blog_id: int, request: BlogCommentRequest, user_id: str):
    """Add a comment to a blog post"""
    try:
        # Verify blog exists
        blog = db.get_blog(blog_id)
        if not blog:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found"}
            )
        
        comment = db.add_blog_comment(blog_id, user_id, request.content)
        
        if not comment:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to add comment"}
            )
        
        # Log comment action
        db.log_user_action(user_id, "comment_blog", {
            "blog_id": blog_id,
            "comment_id": comment["comment_id"]
        })
        
        return {
            "status": "success",
            "comment": comment
        }
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error adding comment: {str(e)}"}
        )

@app.get("/blogs/{blog_id}/comments")
async def get_blog_comments(blog_id: int):
    """Get all comments for a blog post"""
    try:
        # Verify blog exists
        blog = db.get_blog(blog_id)
        if not blog:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found"}
            )
        
        comments = db.get_blog_comments(blog_id)
        
        return {
            "status": "success",
            "comments": comments,
            "count": len(comments)
        }
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting comments: {str(e)}"}
        )

@app.post("/blogs/{blog_id}/like")
async def like_blog_post(blog_id: int, user_id: str):
    """Like a blog post"""
    try:
        # Verify blog exists
        blog = db.get_blog(blog_id)
        if not blog:
            return JSONResponse(
                status_code=404,
                content={"error": "Blog post not found"}
            )
        
        success = db.like_blog(blog_id)
        
        if not success:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to like blog post"}
            )
        
        # Log like action
        db.log_user_action(user_id, "like_blog", {"blog_id": blog_id})
        
        return {
            "status": "success",
            "message": "Blog post liked successfully"
        }
    except Exception as e:
        logger.error(f"Error liking blog post: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error liking blog post: {str(e)}"}
        )

@app.get("/blogs/categories")
async def get_blog_categories():
    """Get all unique blog categories"""
    try:
        categories = db.get_blog_categories()
        
        return {
            "status": "success",
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error getting blog categories: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting blog categories: {str(e)}"}
        )

# News sources endpoints
@app.post("/news/sources")
async def add_news_source(
    name: str,
    url: Optional[str] = None,
    logo_url: Optional[str] = None,
    description: Optional[str] = None,
    is_verified: bool = False,
    user_id: str = None
):
    """Add a news source"""
    try:
        # Admin check here if needed
        
        source = db.add_news_source(
            name=name,
            url=url,
            logo_url=logo_url,
            description=description,
            is_verified=is_verified
        )
        
        if not source:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to add news source"}
            )
        
        # Log source creation
        if user_id:
            db.log_user_action(user_id, "add_news_source", {"source_id": source["source_id"]})
        
        return {
            "status": "success",
            "source": source
        }
    except Exception as e:
        logger.error(f"Error adding news source: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error adding news source: {str(e)}"}
        )

@app.get("/news/sources")
async def get_news_sources(verified_only: bool = False):
    """Get all news sources"""
    try:
        sources = db.get_news_sources(verified_only)
        
        return {
            "status": "success",
            "sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        logger.error(f"Error getting news sources: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting news sources: {str(e)}"}
        )

# Fetch official news from APIs
@app.get("/news/official")
async def fetch_official_news(force_refresh: bool = False):
    """Fetch official legal news from external APIs"""
    try:
        # Check if we need to fetch new data or if cached data is available
        # This is a simplified example - in a real app, you'd check timestamps of latest news
        if not force_refresh:
            # Check for existing official news in last 24 hours
            recent_news = db.get_blogs(limit=5, is_official=True)
            if recent_news and len(recent_news) > 0:
                return {
                    "status": "success",
                    "news": recent_news,
                    "count": len(recent_news),
                    "source": "database"
                }
        
        # We'll simulate fetching from an API here
        # In a real app, you'd use requests to call actual legal news APIs
        # Example: Indian Kanoon API, Legal News API, etc.
        
        # For this example, I'll create some simulated legal news
        # In a real implementation, replace with actual API call
        
        legal_news = await fetch_legal_news_from_api()
        
        # Process and store news
        saved_news = []
        for news in legal_news:
            # Get or create source
            sources = db.get_news_sources(verified_only=True)
            source_id = sources[0]["source_id"] if sources else None
            
            if not source_id:
                # Create a default verified source
                source = db.add_news_source(
                    name="Indian Legal News",
                    url="https://indianlegalnews.org",
                    logo_url="https://example.com/logo.png",
                    description="Official source for Indian legal news",
                    is_verified=True
                )
                source_id = source["source_id"] if source else None
            
            # Store the news
            saved = db.add_official_news(
                title=news["title"],
                content=news["content"],
                summary=news["summary"],
                source_id=source_id,
                source_url=news["url"],
                user_id="system",  # Using a system user ID
                category=news["category"],
                tags=news["tags"]
            )
            
            if saved:
                saved_news.append(saved)
        
        return {
            "status": "success",
            "news": saved_news,
            "count": len(saved_news),
            "source": "api"
        }
    except Exception as e:
        logger.error(f"Error fetching official news: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching official news: {str(e)}"}
        )

async def fetch_legal_news_from_api():
    """
    Simulate fetching from a legal news API
    In a real implementation, this would make actual API calls
    """
    # Sample legal news data
    return [
        {
            "title": "Supreme Court Issues New Guidelines on Bail Applications",
            "summary": "The Supreme Court has issued comprehensive guidelines to streamline the bail application process across all courts in India.",
            "content": """# Supreme Court Issues New Guidelines on Bail Applications

The Supreme Court of India has issued a landmark judgment today that establishes comprehensive guidelines for handling bail applications across all courts in the country. The bench, headed by the Chief Justice, emphasized that bail should be the rule and jail the exception.

## Key Guidelines:

1. **Timely Processing**: All bail applications must be heard within 3 working days of filing
2. **Reasoned Orders**: Courts must provide detailed reasoning for granting or denying bail
3. **Financial Constraints**: Inability to furnish bail bonds due to financial constraints should not keep undertrials in custody
4. **Special Provisions**: Special considerations for women, elderly, and first-time offenders

The Court noted that overcrowding in prisons, with a majority being undertrials, necessitated these reforms. "The right to personal liberty is a precious fundamental right and should be curtailed only when necessary," stated the Chief Justice.

Legal experts have welcomed this development, noting it could significantly reduce the burden on the prison system while protecting individual rights.

The guidelines will come into effect immediately and will be binding on all courts across India.
            """,
            "url": "https://supremecourt.gov.in/news/guidelines-bail-applications",
            "category": "Supreme Court",
            "tags": ["Bail", "Criminal Procedure", "Supreme Court", "Guidelines"]
        },
        {
            "title": "Parliament Passes Digital Personal Data Protection Act",
            "summary": "The Indian Parliament has passed the Digital Personal Data Protection Act, establishing new regulations for handling personal data.",
            "content": """# Parliament Passes Digital Personal Data Protection Act

In a significant move toward regulating the digital ecosystem, the Indian Parliament has passed the Digital Personal Data Protection Act after years of deliberation. The legislation aims to protect the personal data of Indian citizens while establishing clear regulations for data processing by both government and private entities.

## Key Provisions:

1. **Data Fiduciary Responsibilities**: Organizations collecting personal data will have specific obligations regarding data security, transparency, and purpose limitation
2. **Individual Rights**: Citizens gain the right to access, correct, and erase their personal data
3. **Data Protection Authority**: Creation of an independent regulatory body to enforce compliance
4. **Penalties**: Significant financial penalties for violations, up to 4% of global turnover

"This law strikes a balance between data protection and allowing innovation in the digital economy," said the Minister for Electronics and IT. The legislation brings India in line with global standards such as the GDPR in Europe while addressing India-specific concerns.

Industry representatives have generally welcomed the move while expressing concerns about implementation challenges. Privacy advocates have praised the rights-based approach but noted that certain exemptions for government agencies could be potentially problematic.

The Act will come into force in phases over the next 12 months, giving organizations time to adjust their data practices to the new requirements.
            """,
            "url": "https://meity.gov.in/digital-personal-data-protection-act",
            "category": "Legislation",
            "tags": ["Data Protection", "Privacy", "Legislation", "Technology Law"]
        },
        {
            "title": "Delhi High Court Rules on Trademark Infringement in E-commerce",
            "summary": "A landmark judgment from the Delhi High Court addresses trademark infringement liability for e-commerce platforms.",
            "content": """# Delhi High Court Rules on Trademark Infringement in E-commerce

The Delhi High Court has delivered a landmark judgment on trademark infringement in the e-commerce space, potentially reshaping how online marketplaces operate in India. The ruling clarifies the extent of liability e-commerce platforms have when third-party sellers list counterfeit products.

## Highlights of the Judgment:

1. **Safe Harbor Provisions**: The court narrowed the interpretation of safe harbor provisions that protect intermediaries
2. **Due Diligence Requirements**: E-commerce platforms must implement specific verification measures for branded products
3. **Notice and Takedown**: Clarified timelines and procedures for addressing infringement complaints
4. **Financial Penalties**: Established guidelines for calculating damages in trademark infringement cases

Justice Sharma noted, "While e-commerce has created tremendous opportunities for businesses and consumers alike, it has also facilitated the proliferation of counterfeit goods. Platforms cannot claim immunity while benefiting financially from such transactions."

The case, filed by a leading luxury brand against a major e-commerce platform, alleged that counterfeit products bearing their trademark were being sold despite multiple notifications.

Legal experts suggest this ruling sets an important precedent that balances the interests of brand owners with the practical realities of operating online marketplaces. The judgment gives platforms 90 days to implement the required verification systems.
            """,
            "url": "https://delhihighcourt.nic.in/judgments/trademark-ecommerce-2023",
            "category": "Intellectual Property",
            "tags": ["Trademark", "E-commerce", "Intellectual Property", "Delhi High Court"]
        }
    ]

# Blog AI assistance endpoint
@app.post("/blogs/ai/generate")
async def generate_blog_content(query: str, user_id: str):
    """Generate blog content using AI"""
    try:
        if not rag_system:
            return JSONResponse(
                status_code=503,
                content={"error": "RAG system not initialized"}
            )
        
        # Generate blog content
        prompt = f"""Generate a well-structured blog post about the following legal topic: {query}
        
        The blog should include:
        1. A descriptive title
        2. A concise summary (2-3 sentences)
        3. A comprehensive main content with proper headings and subheadings
        4. Focus on Indian legal context where applicable
        5. Format the content using Markdown with proper headings
        
        Make the content informative and accessible to law students and practitioners.
        """
        
        result = await rag_system.query(prompt, use_web=True)
        
        # Extract content
        content = ""
        if isinstance(result, str):
            content = result
        elif result and isinstance(result, dict) and "content" in result:
            content = result["content"]
        else:
            content = "I couldn't generate a blog post. Please try with a different topic."
        
        # Log AI generation
        db.log_user_action(user_id, "generate_blog_ai", {"query": query})
        
        # Try to extract title and summary from the generated content
        title = query
        summary = ""
        
        # Look for a title (Markdown heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Look for a summary (typically first paragraph after title)
        summary_match = re.search(r'^#.+\n\n(.+?)\n\n', content, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        elif len(content.split('\n\n')) > 1:
            # Just use the first paragraph as summary
            summary = content.split('\n\n')[1].strip()
        
        return {
            "status": "success",
            "content": content,
            "title": title,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating blog content: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating blog content: {str(e)}"}
        )
