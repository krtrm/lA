import os
import logging
import asyncio
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
# Added security middlewares
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from init_rag import init_rag_system
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Literal
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from functools import lru_cache
from database import Database
import concurrent.futures

# Import the news API router
from news_api import router as news_router

# Initialize database
db = Database()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Vaqeel.app API", description="Legal AI assistant for Indian law")

# Auth setup
security = HTTPBearer()

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

# Initialize RAG system with correct Pinecone index name with timeout
# Use the actual available index from logs: llama-text-embed-v2-index
index_name = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")
namespace = os.getenv("PINECONE_NAMESPACE", "indian-law")

# Set proper data directory for IKAPITool in the current user workspace
os.environ["INDIANKANOON_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ik_data")
logger.info(f"Setting IK data dir to: {os.environ['INDIANKANOON_DATA_DIR']}")

# Global RAG system variable
rag_system = None

@app.on_event("startup")
async def initialize_rag_system_on_startup():
    """Initialize RAG system at application startup if not already initialized"""
    global rag_system
    if rag_system is None:
        try:
            # Offload to threadpool to avoid blocking event loop
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
            if rag_system:
                logger.info("RAG system initialized at startup")
            else:
                logger.error("init_rag_system returned None at startup")
        except Exception as e:
            logger.error(f"Error initializing RAG system at startup: {e}")

logger.info(f"Attempting to initialize RAG system with index: {index_name}, namespace: {namespace}")

# Try to initialize with timeout
try:
    # Use a ThreadPoolExecutor with timeout to prevent hanging
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(init_rag_system, index_name, namespace)
        try:
            # Set a 30-second timeout
            rag_system = future.result(timeout=30)
            if rag_system:
                logger.info("RAG system initialized successfully")
            else:
                logger.error(f"Failed to initialize RAG system with index '{index_name}' - returned None")
        except concurrent.futures.TimeoutError:
            logger.error("RAG system initialization timed out after 30 seconds")
            # Make sure we're not leaving the thread hanging
            future.cancel()
except Exception as e:
    logger.error(f"Exception during RAG system initialization: {str(e)}")

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

# Improved auth handling with proper JWT validation when in production
async def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validate auth token and extract user ID
    In development mode, returns a test user ID
    In production, would validate JWT with Clerk
    """
    token = credentials.credentials
    
    # For development environment, allow test tokens
    if os.getenv("ENVIRONMENT", "development").lower() != "production":
        if token == "test-token":
            return "test-user-id"
    
    try:
        # Split by dots, assuming JWT structure
        parts = token.split('.')
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        # In production, here we would properly validate the JWT with Clerk 
        # and extract the actual user ID from the token claims
        # For now, extract a user ID from the token for testing
        import base64
        import json
        
        # Extract and decode the payload part of the JWT
        try:
            # Add padding if needed
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload = json.loads(base64.b64decode(padded))
            if "sub" in payload:
                return payload["sub"]
            raise HTTPException(status_code=401, detail="User ID not found in token")
        except Exception as e:
            logger.error(f"Error decoding token payload: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token payload")
            
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

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
                yield json.dumps(step.model_dump()) + "\n"
        except Exception as e:
            error_step = StreamStep(
                type="error",
                content=f"Error during streaming: {str(e)}",
                timestamp=time.time()
            )
            yield json.dumps(error_step.model_dump()) + "\n"
    
    return generate()

@app.get("/")
async def root():
    return {"message": "Welcome to the Vaqeel.app Legal AI API"}

@app.post("/query", response_model=QueryResponse)
async def query_legal_ai(request: QueryRequest):
    """
    Non-streaming endpoint for legal queries, returns JSON with answer and sources.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please ensure the vector database is set up.")
    # Call non-streaming query method
    result = await rag_system.query_non_streaming(request.query, request.use_web)
    # Extract answer content and sources
    answer = result.get("content") if isinstance(result, dict) else None
    sources = []
    if isinstance(result, dict):
        sources = result.get("details", {}).get("sources", [])
    return {"answer": answer or "", "sources": sources, "steps": []}

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
                # Handle both string and dict responses
                if isinstance(response, str):
                    content = response
                elif isinstance(response, dict):
                    content = response.get("content", "I'm not sure how to respond to that.")
                else:
                    content = "I'm not sure how to respond to that."
                    
                yield StreamStep(
                    type="complete",
                    content=content,
                    timestamp=time.time()
                )
                return
            
            # Get the plan
            plan = await rag_system.tool_manager.create_plan(request.query)
            # Ensure plan is a dict with tools list
            if not isinstance(plan, dict):
                raw_plan = plan
                plan = {"plan": str(raw_plan), "tools": []}
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
            gathered = await asyncio.gather(*tool_tasks, return_exceptions=True)
            for idx, result in enumerate(gathered):
                await asyncio.sleep(0)  # Yield control
                # Skip exceptions
                if isinstance(result, Exception):
                    yield StreamStep(
                        type="error",
                        content=f"Tool execution error: {str(result)}",
                        timestamp=time.time(),
                        details={"tool": plan.get("tools", [])[idx].get("tool")}
                    )
                    continue
                # Ensure result is a dict
                if not isinstance(result, dict):
                    continue
                if result.get("status") == "success":
                    res_list = result.get("results", []) or []
                    if res_list:
                        all_results.extend(res_list)
                        yield StreamStep(
                            type="retrieval",
                            content=f"Found {len(res_list)} results from {plan.get('tools', [])[idx].get('tool')}",
                            timestamp=time.time(),
                            details={"source": plan.get('tools', [])[idx].get('tool'), "count": len(res_list)}
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
            raw_resp = await rag_system.qa_chain.ainvoke({
                "input": request.query,
                "context": docs_for_context
            })
            # Extract content safely
            if isinstance(raw_resp, dict):
                content = raw_resp.get("content", "I couldn't find a good answer based on the information available.")
            elif hasattr(raw_resp, 'get'):
                content = raw_resp.get("content", "I couldn't find a good answer based on the information available.")
            elif hasattr(raw_resp, 'content'):
                content = raw_resp.content
            else:
                content = str(raw_resp)
            # Return final answer
            yield StreamStep(
                type="complete",
                content=content,
                timestamp=time.time(),
                details={"sources": [
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "source": doc.metadata.get("source", ""),
                        "type": doc.metadata.get("type", "document")
                    } for doc in docs_for_context[:5]
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
@app.post("/extract_keywords")
async def extract_legal_keywords(request: KeywordExtractionRequest):
    # Stream extraction for all keyword requests
    return await stream_extract_legal_keywords(request)

@app.post("/extract_keywords/stream")
async def stream_extract_legal_keywords(request: KeywordExtractionRequest):
    global rag_system
    if rag_system is None:
        try:
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
        except Exception:
            pass
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

@app.post("/generate_argument")
async def generate_legal_argument(request: ArgumentGenerationRequest):
    # Stream argument generation consistently
    return await stream_generate_legal_argument(request)

@app.post("/generate_argument/stream")
async def stream_generate_legal_argument(request: ArgumentGenerationRequest):
    global rag_system
    if rag_system is None:
        try:
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
        except Exception:
            pass
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

@app.post("/create_outline")
async def create_document_outline(request: OutlineGenerationRequest):
    # Stream outline creation
    return await stream_create_document_outline(request)

@app.post("/create_outline/stream")
async def stream_create_document_outline(request: OutlineGenerationRequest):
    global rag_system
    if rag_system is None:
        # lazy initialize if startup failed
        try:
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
        except Exception:
            pass
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

@app.post("/verify_citation")
async def verify_legal_citation(request: CitationVerificationRequest):
    # Stream citation verification
    return await stream_verify_legal_citation(request)

@app.post("/verify_citation/stream")
async def stream_verify_legal_citation(request: CitationVerificationRequest):
    global rag_system
    if rag_system is None:
        try:
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
        except Exception:
            pass
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
