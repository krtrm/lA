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
        await rag_system.vector_store.client.close()

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

# Add database connection timeout
@app.on_event("startup")
async def startup_db():
    await rag_system.vector_store.client.create_index(
        timeout=30  # Add 30s timeout for DB operations
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
