import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader, YoutubeLoader
from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup
import httpx
import tldextract
from typing import List, Dict, Any, Optional, Union, Callable
import json
import asyncio
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
import tempfile
import fitz  # PyMuPDF
from pytube import YouTube
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from ikapi import IKApi, FileStorage
import logging
import requests  # Added for Serper API
import urllib.parse  # Added for URL encoding
from langchain.chains.combine_documents import create_stuff_documents_chain
import tiktoken  # Add this import for token counting
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables - try absolute path first, then fallback to auto-find
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Pinecone client
def get_pinecone_client():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc

# Replace the embeddings initialization with a subclass that slices vectors to 1024 dimensions
class SlicedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text: str) -> List[float]:  # Specify list of float not just list
        vec = super().embed_query(text)
        return vec[:1024]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # Proper type annotation
        vectors = super().embed_documents(texts)
        return [v[:1024] for v in vectors]

embeddings = SlicedOpenAIEmbeddings(model="text-embedding-3-small")

# Document processing pipeline
def create_vector_store(doc_paths: List[str], index_name: str = "llama-text-embed-v2-index", namespace: str = "indian-law", books_folder: str = None):
    """
    Create or update a production vector store with documents, prioritizing books folder PDFs
    """
    # First prioritize books if specified
    all_docs = []
    books_docs = []
    other_docs = []
    
    # Process books folder first if specified
    if books_folder and os.path.isdir(books_folder):
        logger.info(f"Processing books folder: {books_folder}")
        books_loader = DirectoryLoader(books_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        books_docs = books_loader.load()
        logger.info(f"Loaded {len(books_docs)} documents from books folder")
    
    # Process other document paths
    for doc_path in doc_paths:
        if os.path.isdir(doc_path):
            loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        else:
            # Automatically detect file type
            if doc_path.endswith('.pdf'):
                loader = PyPDFLoader(doc_path)
            else:
                loader = UnstructuredFileLoader(doc_path)
        
        docs = loader.load()
        other_docs.extend(docs)
    
    # Combine with books first
    all_docs = books_docs + other_docs
    logger.info(f"Total documents loaded: {len(all_docs)} (Books: {len(books_docs)}, Other: {len(other_docs)})")
    
    # Split documents with proper parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    
    # Process metadata
    for i, doc in enumerate(splits):
        if not doc.metadata.get('doc_id'):
            doc.metadata['doc_id'] = f"doc_{i}"
        doc.metadata['chunk_id'] = i
        doc.metadata['source_type'] = 'document'
        # Mark books for better retrieval
        if i < len(books_docs):
            doc.metadata['priority'] = 'high'
            doc.metadata['source_type'] = 'book'
    
    # Initialize Pinecone
    pc = get_pinecone_client()
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pinecone_region = os.getenv("PINECONE_REGION", "gcp-starter")
        logger.info(f"Creating Pinecone index '{index_name}' in region '{pinecone_region}'")
        
        try:
            if '-' in pinecone_region:
                cloud = pinecone_region.split('-')[0]
                region_val = '-'.join(pinecone_region.split('-')[1:])
                # If region is "starter", then omit the region key
                if region_val.lower() == "starter":
                    spec_config = {"serverless": {"cloud": cloud}}
                else:
                    spec_config = {"serverless": {"cloud": cloud, "region": region_val}}
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=spec_config
                )
            else:
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": pinecone_region}}
                )
        except Exception as e:
            logger.warning(f"Error creating index with region-specific config: {e}")
            # Try alternative configurations
            try:
                # Try with simplified config (just cloud, no region)
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": "gcp"}}
                )
                logger.info("Successfully created index with simplified GCP config")
            except Exception as e2:
                # Final fallback - try AWS
                logger.warning(f"Error with GCP config: {e2}, trying AWS...")
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=1536,
                        metric="cosine",
                        spec={"serverless": {"cloud": "aws"}}
                    )
                    logger.info("Successfully created index with AWS config")
                except Exception as e3:
                    logger.error(f"All index creation attempts failed: {e3}")
                    raise Exception(f"Unable to create Pinecone index: {e3}")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Set up vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=namespace
    )
    
    # Add documents to vector store in batches to avoid payload exceeding size limits
    batch_size = 50  # adjust as needed
    for i in range(0, len(splits), batch_size):
        batch = splits[i: i+batch_size]
        vectorstore.add_documents(batch)
    
    return vectorstore

# Tool system architecture
class Tool:
    """Base class for all tools in the RAG system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the tool and return results"""
        raise NotImplementedError("Tool must implement run method")
    
    def to_dict(self) -> Dict[str, str]:
        """Return tool metadata as dict"""
        return {
            "name": self.name,
            "description": self.description
        }

class IKAPITool(Tool):
    """Tool for searching Indian Kanoon legal documents"""
    
    def __init__(self):
        super().__init__(
            name="indian_kanoon_search",
            description="Search for Indian legal documents, cases, and judgments. Provides authoritative legal information from Indian Kanoon."
        )
        token = os.getenv("INDIANKANOON_API_TOKEN")
        data_dir = os.getenv("INDIANKANOON_DATA_DIR", "/tmp/ik_data")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.storage = FileStorage(data_dir)
        
        # Configure minimal args with proper type annotations
        class Args:
            token: str = None
            maxcites: int = 5
            maxcitedby: int = 5
            orig: bool = False
            maxpages: int = 1
            pathbysrc: bool = True
            addedtoday: bool = False
            numworkers: int = 1
            fromdate: Optional[str] = None
            todate: Optional[str] = None
            sortby: Optional[str] = None
        
        args = Args()
        args.token = token
        args.maxcites = 5
        args.maxcitedby = 5
        args.orig = False
        args.maxpages = 1
        args.pathbysrc = True
        args.addedtoday = False
        args.numworkers = 1
        args.fromdate = None
        args.todate = None
        args.sortby = None
        
        self.ikapi = IKApi(args, self.storage)
    
    async def run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search case law in Indian Kanoon"""
        start_time = time.time()
        try:
            results = self.ikapi.search(query, 0, 1)  # First page only
            result_obj = json.loads(results)
            
            if 'docs' not in result_obj:
                return {"status": "success", "results": [], "source": "indian_kanoon", "time_taken": time.time() - start_time}
                
            docs = result_obj['docs'][:max_results]
            processed_docs = []
            
            # Process docs concurrently for speed
            with ThreadPoolExecutor(max_workers=min(len(docs), 3)) as executor:
                future_to_doc = {executor.submit(self._fetch_document, doc): doc for doc in docs}
                for future in as_completed(future_to_doc):
                    try:
                        processed_doc = future.result()
                        if processed_doc:
                            processed_docs.append(processed_doc)
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"IK API search completed in {elapsed_time:.2f}s with {len(processed_docs)} results")
            
            return {
                "status": "success", 
                "results": processed_docs, 
                "source": "indian_kanoon",
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error searching Indian Kanoon: {e}")
            return {"status": "error", "error": str(e), "results": [], "source": "indian_kanoon", "time_taken": time.time() - start_time}
    
    def _fetch_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper to fetch and process a single document"""
        try:
            doc_id = doc.get('tid')  # Use .get() instead of ['tid'] for safety
            if doc_id is None:
                return None
                
            doc_json = self.ikapi.fetch_doc(doc_id)  # type: ignore  # Ignore type error for ikapi
            doc_data = json.loads(doc_json)
            
            return {
                "title": doc_data.get('title', ''),
                "content": doc_data.get('doc', ''),
                "source": f"indiankanoon.org/doc/{doc_id}/",
                "domain": "indiankanoon.org",
                "docid": doc_id,
                "citation": doc_data.get('citation', ''),
                "type": "legal_document"
            }
        except Exception as e:
            logger.error(f"Error fetching document {doc.get('tid', 'unknown')}: {e}")
            return None

class WebSearchTool(Tool):
    """Tool for searching and retrieving web content"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information about legal topics, laws, and procedures. Returns content from various websites."
        )
        # Increase timeout for better reliability
        self.http_client = httpx.AsyncClient(timeout=15, verify=False)
        # Track already processed URLs to avoid duplicates
        self.processed_urls = set()
        # Domain priorities for ranking results
        self.domain_priorities = {
            # Government and legal sites (highest priority)
            "indiankanoon.org": "very_high",
            "supremecourt.gov.in": "very_high",
            "sci.gov.in": "very_high",
            "main.sci.gov.in": "very_high",
            "doj.gov.in": "very_high",
            "legalservices.gov.in": "very_high",
            "indiacode.nic.in": "very_high",
            "lawmin.gov.in": "very_high",
            "legislative.gov.in": "very_high",
            "highcourtofkerala.nic.in": "very_high",
            "bombayhighcourt.nic.in": "very_high", 
            "karnatakajudiciary.kar.nic.in": "very_high",
            "delhihighcourt.nic.in": "very_high",
            # Any .gov.in domain
            "gov.in": "high",
            "nic.in": "high",
            # Educational and organizational domains
            "nalsar.ac.in": "high",
            "nls.ac.in": "high",
            "nludelhi.ac.in": "high",
            "jgu.edu.in": "high",
            # Legal resource sites
            "scconline.com": "high",
            "livelaw.in": "high",
            "barandbench.com": "high",
            "manupatra.com": "high",
            "niti.gov.in": "high",
            # Educational sites
            "edu": "medium",
            "ac.in": "medium",
            # Other organizational sites
            "org": "medium",
            # Commercial sites
            "com": "low",
            # Video sites
            "youtube.com": "medium",
            "youtu.be": "medium"
        }
        # Set thresholds
        self.min_results_threshold = 3
        self.max_search_queries = 2
        self.max_urls_per_query = 3
        self.concurrent_requests = 3  # How many urls to process simultaneously
    
    async def run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web and process results with improved efficiency"""
        start_time = time.time()
        search_results = await self._search_web(query, max_results * 2)  # Get more results than needed for filtering
        
        if not search_results:
            return {"status": "error", "error": "No search results found", "results": [], "time_taken": time.time() - start_time}
        
        # Filter out already processed URLs
        filtered_results = [r for r in search_results if r["url"] not in self.processed_urls]
        
        # Sort by domain priority
        prioritized_results = self._prioritize_results(filtered_results)
        
        # Take top N results
        results_to_process = prioritized_results[:min(max_results, self.max_urls_per_query)]
        
        # Update processed URLs set
        for result in results_to_process:
            self.processed_urls.add(result["url"])
        
        # Process each URL concurrently
        processed_results = []
        if results_to_process:
            tasks = []
            for result in results_to_process:
                tasks.append(self._process_content(result["url"]))
            
            # Process in batches for better resource management
            for i in range(0, len(tasks), self.concurrent_requests):
                batch = tasks[i:i+self.concurrent_requests]
                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Error processing content: {result}")
                        elif result:
                            processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Web search completed in {elapsed_time:.2f}s with {len(processed_results)} results")
        
        return {
            "status": "success", 
            "results": processed_results, 
            "source": "web_search",
            "time_taken": elapsed_time
        }
    
    def _prioritize_results(self, results: List[Dict]) -> List[Dict]:
        """Sort results by domain priority"""
        # Calculate priority score for each result
        for result in results:
            url = result.get("url", "")
            domain = tldextract.extract(url).registered_domain
            
            # Default priority is lowest
            priority_score = 0
            
            # Check for exact domain matches first
            if domain in self.domain_priorities:
                priority = self.domain_priorities[domain]
                if priority == "very_high":
                    priority_score = 100
                elif priority == "high":
                    priority_score = 75 
                elif priority == "medium":
                    priority_score = 50
                elif priority == "low":
                    priority_score = 25
            else:
                # Check for partial domain matches (.gov.in etc)
                for key, value in self.domain_priorities.items():
                    if key in url:
                        if value == "very_high":
                            priority_score = 100
                            break
                        elif value == "high":
                            priority_score = 75
                            break
                        elif value == "medium":
                            priority_score = 50
                            break
                        elif value == "low":
                            priority_score = 25
                            break
            
            result["priority_score"] = priority_score
        
        # Sort by priority score (descending)
        return sorted(results, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    async def _search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform a web search using Serper API with Indian geolocation
        """
        # Add warnings filter to suppress SSL verification warnings
        import warnings
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        try:
            # Use Serper API as primary search method if available
            serper_api_key = os.getenv("SERPER_API_KEY")
            if (serper_api_key):
                logger.info(f"Searching with Serper API: {query}")
                url = "https://google.serper.dev/search"
                
                payload = json.dumps({
                    "q": query,
                    "gl": "in",  # Set geolocation to India
                    "num": max_results * 2  # Request more results to have backup options
                })
                
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                data = response.json()
                
                results = []
                
                # Process organic search results
                if "organic" in data:
                    for item in data["organic"][:max_results]:
                        # Skip sites that commonly cause SSL errors
                        if any(domain in item.get("link", "") for domain in ["rtionline.up.gov.in"]):
                            continue
                            
                        results.append({
                            "url": item.get("link", ""),
                            "title": item.get("title", ""),
                            "description": item.get("snippet", "")
                        })
                
                # Process knowledge graph if available
                if "knowledgeGraph" in data and len(results) < max_results:
                    kg = data["knowledgeGraph"]
                    if "descriptionLink" in kg:
                        results.append({
                            "url": kg.get("descriptionLink", ""),
                            "title": kg.get("title", ""),
                            "description": kg.get("description", "")
                        })
                
                return results[:max_results]
                
            # Fall back to DuckDuckGo if Serper API key not available
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = await self.http_client.get(url)
            data = response.json()
            
            results = []
            # Extract organic results
            for result in data.get("Results", [])[:max_results]:
                results.append({
                    "url": result.get("FirstURL", ""),
                    "title": result.get("Text", ""),
                    "description": ""
                })
            
            # Also include related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "FirstURL" in topic:
                    results.append({
                        "url": topic.get("FirstURL", ""),
                        "title": topic.get("Text", ""),
                        "description": ""
                    })
            
            return results
                
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            # Fallback to a simple Google search page scraping as last resort
            try:
                search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = await self.http_client.get(search_url, headers=headers)
                
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                
                for g in soup.find_all('div', class_='g'):
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text if g.find('h3') else "No title"
                        results.append({
                            "url": link,
                            "title": title,
                            "description": ""
                        })
                        
                        if len(results) >= max_results:
                            break
                
                return results
            except Exception as e2:
                logger.error(f"Error in fallback search: {e2}")
                return []
    
    async def _process_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Process different content types from the web"""
        try:
            # Determine content type
            if "youtube.com" in url or "youtu.be" in url:
                return await self._extract_youtube(url)
            elif url.endswith(".pdf"):
                return await self._extract_pdf(url)
            else:
                return await self._extract_webpage(url)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    async def _extract_youtube(self, url: str) -> Dict[str, Any]:
        """Extract content from YouTube videos"""
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True, language=["en", "hi"]
            )
            docs = loader.load()
            
            if not docs:
                # Fallback to pytube
                yt = YouTube(url)
                transcript = yt.captions.get_by_language_code('en')
                
                if transcript:
                    content = transcript.generate_srt_captions()
                else:
                    content = yt.description
                
                return {
                    "content": content,
                    "source": url,
                    "domain": "youtube.com",
                    "title": yt.title,
                    "type": "video"
                }
            
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            metadata = docs[0].metadata if docs else {}
            
            return {
                "content": combined_text,
                "source": url,
                "domain": "youtube.com",
                "title": metadata.get("title", "YouTube Video"),
                "type": "video"
            }
        except Exception as e:
            logger.error(f"Error extracting YouTube content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": "youtube.com",
                "title": "YouTube Video",
                "type": "video"
            }
    
    async def _extract_pdf(self, url: str) -> Dict[str, Any]:
        """Extract content from PDF documents with OCR"""
        try:
            response = await self.http_client.get(url)
            
            # Download the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            text_content = ""
            try:
                # First try to extract text directly
                doc = fitz.open(tmp_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text_content += page.get_text()
                doc.close()
                
                # If no text extracted, use OCR
                if not text_content.strip():
                    doc = fitz.open(tmp_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img = Image.open(BytesIO(pix.tobytes()))
                        text_content += pytesseract.image_to_string(img)
                    doc.close()
            finally:
                os.unlink(tmp_path)
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": text_content,
                "source": url,
                "domain": domain,
                "title": f"PDF Document from {domain}",
                "type": "pdf"
            }
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": tldextract.extract(url).domain,
                "title": "PDF Document",
                "type": "pdf"
            }
    
    async def _extract_webpage(self, url: str) -> Dict[str, Any]:
        """Extract content from webpages"""
        try:
            # Skip specific problematic domains
            if any(domain in url for domain in ["rtionline.up.gov.in"]):
                return None
                
            # Use a custom loader with SSL verification disabled for government sites
            loader = AsyncHtmlLoader([url], verify_ssl=False)
            docs = await loader.aload()
            
            if not docs:
                return None
                
            transformer = Html2TextTransformer()
            docs_transformed = transformer.transform_documents(docs)
            
            if not docs_transformed:
                return None
                
            content = docs_transformed[0].page_content
            metadata = docs_transformed[0].metadata
            
            # Extract title and other metadata using BeautifulSoup
            soup = BeautifulSoup(docs[0].page_content, "html.parser")
            title = soup.title.text if soup.title else url
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": content,
                "source": url,
                "domain": domain,
                "title": title,
                "type": "webpage"
            }
        except Exception as e:
            logger.error(f"Error extracting webpage content: {e}")
            return None

class VectorDBLookupTool(Tool):
    """Tool for retrieving information from the vector database"""
    
    def __init__(self, vectorstore):
        super().__init__(
            name="vector_db_lookup",
            description="Search the legal knowledge base for relevant information. Contains high-quality legal documents, acts, and precedents."
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
    
    async def run(self, query: str, k: int = 8) -> Dict[str, Any]:
        """Retrieve documents from vector database"""
        start_time = time.time()
        try:
            # Set a reasonable default
            k = min(max(1, k), 10)
            
            # Update retrieval parameters
            self.retriever.search_kwargs["k"] = k
            
            # Get documents
            docs = self.retriever.invoke(query)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "vector database"),
                    "title": doc.metadata.get("title", "Document"),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "type": doc.metadata.get("source_type", "document")
                })
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vector DB lookup completed in {elapsed_time:.2f}s with {len(results)} documents")
            
            return {
                "status": "success", 
                "results": results, 
                "source": "vector_db",
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error in vector DB lookup: {e}")
            return {"status": "error", "error": str(e), "results": [], "source": "vector_db", "time_taken": time.time() - start_time}

class PineconeIndexingTool(Tool):
    """Tool for indexing new content into Pinecone"""
    
    def __init__(self, embeddings):
        super().__init__(
            name="pinecone_indexer",
            description="Index new valuable content into the Pinecone vector database for future retrieval"
        )
        self.embeddings = embeddings
        self.pc = get_pinecone_client()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def run(self, content: List[Dict], index_name: str = "llama-text-embed-v2-index", namespace: str = "indian-law") -> Dict[str, Any]:
        """Index content into Pinecone"""
        start_time = time.time()
        try:
            # Convert to Document objects
            docs = []
            for i, item in enumerate(content):
                if not item or not item.get("content"):
                    continue
                    
                docs.append(
                    Document(
                        page_content=item["content"],
                        metadata={
                            "source": item.get("source", ""),
                            "title": item.get("title", ""),
                            "domain": item.get("domain", ""),
                            "type": item.get("type", "web"),
                            "doc_id": f"dynamic_{int(time.time())}_{i}"
                        }
                    )
                )
            
            if not docs:
                return {
                    "status": "error", 
                    "error": "No valid content to index", 
                    "indexed_count": 0,
                    "time_taken": time.time() - start_time
                }
            
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            
            # Process metadata
            for i, doc in enumerate(splits):
                if not doc.metadata.get('doc_id'):
                    doc.metadata['doc_id'] = f"doc_{int(time.time())}_{i}"
                doc.metadata['chunk_id'] = i
                doc.metadata['source_type'] = doc.metadata.get('type', 'web')
                doc.metadata['indexed_date'] = time.strftime("%Y-%m-%d")
            
            # Get the index
            if index_name not in self.pc.list_indexes().names():
                return {
                    "status": "error", 
                    "error": f"Index {index_name} does not exist", 
                    "indexed_count": 0,
                    "time_taken": time.time() - start_time
                }
                
            index = self.pc.Index(index_name)
            
            # Set up vector store
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace
            )
            
            # Add documents to vector store in batches
            batch_size = 50
            total_indexed = 0
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i: i+batch_size]
                vectorstore.add_documents(batch)
                total_indexed += len(batch)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Indexing completed in {elapsed_time:.2f}s - indexed {total_indexed} documents")
            
            return {
                "status": "success", 
                "indexed_count": total_indexed,
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error in indexing: {e}")
            return {"status": "error", "error": str(e), "indexed_count": 0, "time_taken": time.time() - start_time}

class ToolManager:
    """Manages tool registration, selection and execution"""
    
    def __init__(self, planning_llm):
        self.tools = {}
        self.planning_llm = planning_llm  # deepseek-r1 for planning
        self.evaluation_llm = ChatGroq(
            temperature=0.1,
            model_name="deepseek-r1-distill-llama-70b",
            max_tokens=2048
        )
        
        # Planning prompt template
        self.planning_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with planning how to answer a legal query.
        
        Available tools:
        {tools}
        
        User query: {query}
        
        First, analyze the query to understand what information is needed. Then create a plan for using the tools to get that information.
        
        For each tool, decide if it should be used, in what order, and with what parameters. Remember that:
        1. Always start with vector_db_lookup to check our existing knowledge base
        2. Only use indian_kanoon_search if the query explicitly relates to Indian law, cases, or legal precedents
        3. Only resort to web_search if the above tools don't yield sufficient information
        
        Your output must be a valid JSON object with this structure:
        {
            "plan": "Brief explanation of your approach",
            "tools": [
                {
                    "tool": "tool_name",
                    "parameters": {"param1": "value1"},
                    "reason": "Why you're using this tool"
                },
                ...
            ]
        }
        
        Return only the JSON, nothing else.
        """)
        
        # Evaluation prompt template
        self.evaluation_prompt = ChatPromptTemplate.from_template("""
        You need to evaluate if the information below would be valuable to add to our legal knowledge base.
        
        Information:
        {content}
        
        Source: {source}
        Type: {type}
        
        Evaluate the information based on:
        1. Accuracy & reliability - Is it from an authoritative source?
        2. Relevance to Indian law and legal topics
        3. Completeness and depth
        4. Uniqueness - Does it contain information not commonly available?
        
        Output just a JSON object with:
        {
            "should_index": true/false,
            "reason": "Brief explanation of your decision",
            "quality_score": score from 1-10
        }
        
        Return only the JSON, nothing else.
        """)
    
    def register_tool(self, tool: Tool):
        """Register a tool with the manager"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get a list of all registered tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Generate a formatted description of available tools"""
        result = ""
        for name, tool in self.tools.items():
            result += f"- {name}: {tool.description}\n"
        return result
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool by name"""
        if tool_name not in self.tools:
            return {"status": "error", "error": f"Tool '{tool_name}' not found"}
            
        try:
            return await self.tools[tool_name].run(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_plan(self, query: str) -> Dict[str, Any]:
        """Create a plan for answering the query"""
        try:
            prompt_val = self.planning_prompt.format_messages(
                tools=self.get_tools_description(),
                query=query
            )
            
            response = await self.planning_llm.ainvoke(prompt_val)
            plan_text = response.content
            
            # Extract JSON from response (in case the model adds extra text)
            import re
            json_match = re.search(r'({.*})', plan_text, re.DOTALL)
            if json_match:
                plan_text = json_match.group(1)
            
            # Parse the plan
            plan = json.loads(plan_text)
            return plan
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Return a default plan that just uses vector DB and web search
            return {
                "plan": "Fallback plan due to planning error",
                "tools": [
                    {
                        "tool": "vector_db_lookup",
                        "parameters": {"query": query, "k": 5},
                        "reason": "Check existing knowledge base"
                    },
                    {
                        "tool": "web_search",
                        "parameters": {"query": query, "max_results": 3},
                        "reason": "Find additional information online"
                    }
                ]
            }
    
    async def evaluate_content(self, content: Dict) -> Dict[str, Any]:
        """Evaluate if content should be indexed"""
        try:
            prompt_val = self.evaluation_prompt.format_messages(
                content=content.get("content", ""),
                source=content.get("source", "unknown"),
                type=content.get("type", "unknown")
            )
            
            response = await self.evaluation_llm.ainvoke(prompt_val)
            eval_text = response.content
            
            # Extract JSON from response
            import re
            json_match = re.search(r'({.*})', eval_text, re.DOTALL)
            if json_match:
                eval_text = json_match.group(1)
                
            # Parse the evaluation
            evaluation = json.loads(eval_text)
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            # Default to not indexing on error
            return {
                "should_index": False,
                "reason": f"Evaluation error: {str(e)}",
                "quality_score": 0
            }

class EnhancedLegalRAGSystem(LegalRAGSystem):
    """Advanced RAG system with tool-based architecture"""
    
    def __init__(self, vectorstore, enable_web=True):
        super().__init__(vectorstore)
        
        # Initialize tool-based system
        self.planning_llm = ChatGroq(
            temperature=0.1,
            model_name="deepseek-r1-distill-llama-70b",
            max_tokens=2048
        )
        
        self.tool_manager = ToolManager(self.planning_llm)
        
        # Register tools
        self.tool_manager.register_tool(VectorDBLookupTool(vectorstore))
        self.tool_manager.register_tool(IKAPITool())
        self.tool_manager.register_tool(WebSearchTool())
        self.tool_manager.register_tool(PineconeIndexingTool(embeddings))
        
        # Cache for results
        self.cache = {}
    
    async def query(self, question: str, use_web: bool = True):
        try:
            # 1. Planning phase - determine which tools to use
            plan = await self.tool_manager.create_plan(question)
            logger.info(f"Created plan: {plan['plan']}")
            
            # 2. Tool execution phase
            all_results = []
            valuable_content = []
            
            for tool_step in plan.get("tools", []):
                tool_name = tool_step.get("tool")
                parameters = tool_step.get("parameters", {})
                
                logger.info(f"Executing tool: {tool_name}")
                tool_result = await self.tool_manager.execute_tool(tool_name, **parameters)
                
                if tool_result.get("status") == "success":
                    results = tool_result.get("results", [])
                    
                    if results:
                        all_results.extend(results)
                        logger.info(f"Got {len(results)} results from {tool_name}")
                        
                        # Evaluate if results should be indexed (if not from vector DB)
                        if tool_name != "vector_db_lookup" and tool_result.get("source") != "vector_db":
                            for result in results:
                                # Skip empty or very short content
                                if not result.get("content") or len(result.get("content", "")) < 200:
                                    continue
                                    
                                evaluation = await self.tool_manager.evaluate_content(result)
                                if evaluation.get("should_index", False) and evaluation.get("quality_score", 0) >= 7:
                                    valuable_content.append(result)
                                    logger.info(f"Marked content from {result.get('source', 'unknown')} for indexing (score: {evaluation.get('quality_score', 0)})")
            
            # 3. Index valuable content if any
            if valuable_content:
                logger.info(f"Indexing {len(valuable_content)} valuable pieces of content")
                indexing_result = await self.tool_manager.execute_tool("pinecone_indexer", content=valuable_content)
                logger.info(f"Indexing result: {indexing_result}")
            
            # 4. Convert to Document objects for final response generation
            documents = []
            for result in all_results:
                if not result.get("content"):
                    continue
                    
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
            
            # 5. Manage token count
            token_budget = self.max_input_tokens
            question_tokens = self._count_tokens(question)
            token_budget -= question_tokens
            
            # Calculate tokens for prompt template (fixed approximation)
            system_prompt = "You are a legal assistant specializing in Indian law. Analyze the following legal context and provide a detailed, accurate response with proper citations."
            human_prompt_template = "Context:\n{context}\n\nQuestion: {input}"
            prompt_template_tokens = self._count_tokens(system_prompt) + self._count_tokens(human_prompt_template)
            token_budget -= prompt_template_tokens
            
            # Reserve tokens for the model's response
            token_budget -= 2000  # Response tokens
            
            # Select documents to include within token budget
            docs_for_context = self._select_docs_within_budget(documents, token_budget)
            
            # 6. Generate response
            logger.info(f"Generating response with {len(docs_for_context)} documents")
            return await self.qa_chain.ainvoke({
                "input": question,
                "context": docs_for_context
            })
        except Exception as e:
            logger.error(f"Error in tool-based query: {e}")
            # Fallback to simpler approach
            try:
                # Get some documents from vector DB
                docs = self.retriever.invoke(question)
                return await self.qa_chain.ainvoke({
                    "input": question,
                    "context": docs[:3]  # Use only top 3 docs
                })
            except Exception as e2:
                logger.error(f"Error in fallback query: {e2}")
                return {"content": f"I'm having trouble answering this question due to a technical error. Please try asking in a different way or contact support. Error: {str(e2)}"}
    
    def _select_docs_within_budget(self, documents: List[Document], token_budget: int) -> List[Document]:
        """Select documents to include within token budget"""
        docs_for_context = []
        current_tokens = 0
        
        # First sort documents by relevance/priority if metadata score exists
        documents.sort(
            key=lambda x: x.metadata.get("relevance_score", 0) + x.metadata.get("domain_score", 0), 
            reverse=True
        )
        
        for doc in documents:
            doc_tokens = self._count_tokens(doc.page_content)
            
            # Skip extremely large documents
            if doc_tokens > token_budget * 0.8:
                # Truncate very large documents
                try:
                    if self.tokenizer:
                        max_tokens = int(token_budget * 0.5)  # Use at most 50% of budget for a single doc
                        truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                        doc.page_content = truncated_content + "... [content truncated due to length]"
                    else:
                        # Simple truncation for fallback case
                        max_chars = int(token_budget * 0.5 * 4)  # Rough char estimate
                        doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                except Exception as e:
                    logger.warning(f"Error truncating document: {e}")
                    # Simple truncation fallback
                    doc.page_content = doc.page_content[:5000] + "... [content truncated]"
                
                doc_tokens = self._count_tokens(doc.page_content)
            
            # Check if adding this document would exceed our budget
            if current_tokens + doc_tokens > token_budget:
                # If we already have some docs, stop here
                if docs_for_context:
                    break
                # If this is the first doc, truncate it to fit
                else:
                    try:
                        if self.tokenizer:
                            max_tokens = token_budget - 100  # Leave a small buffer
                            truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                            doc.page_content = truncated_content + "... [content truncated due to length]"
                        else:
                            # Simple truncation for fallback case
                            max_chars = int((token_budget - 100) * 4)  # Rough char estimate
                            doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                    except Exception as e:
                        logger.warning(f"Error truncating document: {e}")
                        # Simple truncation fallback
                        doc.page_content = doc.page_content[:3000] + "... [content truncated]"
                    
                    doc_tokens = self._count_tokens(doc.page_content)
            
            docs_for_context.append(doc)
            current_tokens += doc_tokens
            
            # If we've exceeded 70% of our budget, stop adding more
            if current_tokens > token_budget * 0.7:
                break
        
        return docs_for_context

class IKAPIWrapper:
    """Wrapper for Indian Kanoon API to retrieve legal documents"""
    
    def __init__(self):
        token = os.getenv("INDIANKANOON_API_TOKEN")
        data_dir = os.getenv("INDIANKANOON_DATA_DIR", "/tmp/ik_data")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.storage = FileStorage(data_dir)
        
        # Configure minimal args
        class Args:
            token: str = None
            maxcites: int = 5
            maxcitedby: int = 5
            orig: bool = False
            maxpages: int = 1
            pathbysrc: bool = True
            addedtoday: bool = False
            numworkers: int = 1
            fromdate: Optional[str] = None
            todate: Optional[str] = None
            sortby: Optional[str] = None
        
        args = Args()
        args.token = token
        args.maxcites = 5
        args.maxcitedby = 5
        args.orig = False
        args.maxpages = 1
        args.pathbysrc = True
        args.addedtoday = False
        args.numworkers = 1
        args.fromdate = None
        args.todate = None
        args.sortby = None
        
        self.ikapi = IKApi(args, self.storage)
    
    def search_case_law(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search case law in Indian Kanoon"""
        try:
            results = self.ikapi.search(query, 0, 1)  # First page only
            result_obj = json.loads(results)
            
            if 'docs' not in result_obj:
                return []
                
            docs = result_obj['docs'][:max_results]
            processed_docs = []
            
            for doc in docs:
                doc_id = doc['tid']
                doc_json = self.ikapi.fetch_doc(doc_id)
                doc_data = json.loads(doc_json)
                
                processed_docs.append({
                    "title": doc_data.get('title', ''),
                    "content": doc_data.get('doc', ''),
                    "source": f"indiankanoon.org/doc/{doc_id}/",
                    "domain": "indiankanoon.org",
                    "docid": doc_id,
                    "citation": doc_data.get('citation', ''),
                    "type": "legal_document"
                })
                
            return processed_docs
        except Exception as e:
            logger.error(f"Error searching Indian Kanoon: {e}")
            return []

class WebRetriever:
    """Enhanced web retriever with multimedia support"""
    
    def __init__(self):
        # Increase timeout for better reliability
        self.http_client = httpx.AsyncClient(timeout=15, verify=False)  # Disable SSL verification for problematic sites
        self.ik_api = IKAPIWrapper()
        # Track already processed URLs to avoid duplicates
        self.processed_urls = set()
        # Set a minimum result count threshold before stopping search
        self.min_results_threshold = 5
        # Maximum number of search queries to try
        self.max_search_queries = 2
        # Maximum URLs to process per query
        self.max_urls_per_query = 3
    
    async def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform a web search using Serper API with Indian geolocation
        """
        # Add warnings filter to suppress SSL verification warnings
        import warnings
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        try:
            # Use Serper API as primary search method if available
            serper_api_key = os.getenv("SERPER_API_KEY")
            if (serper_api_key):
                logger.info(f"Searching with Serper API: {query}")
                url = "https://google.serper.dev/search"
                
                payload = json.dumps({
                    "q": query,
                    "gl": "in",  # Set geolocation to India
                    "num": max_results * 2  # Request more results to have backup options
                })
                
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                data = response.json()
                
                results = []
                
                # Process organic search results
                if "organic" in data:
                    for item in data["organic"][:max_results]:
                        # Skip sites that commonly cause SSL errors
                        if any(domain in item.get("link", "") for domain in ["rtionline.up.gov.in"]):
                            continue
                            
                        results.append({
                            "url": item.get("link", ""),
                            "title": item.get("title", ""),
                            "description": item.get("snippet", "")
                        })
                
                # Process knowledge graph if available (only if we need more results)
                if "knowledgeGraph" in data and len(results) < max_results:
                    kg = data["knowledgeGraph"]
                    if "descriptionLink" in kg:
                        results.append({
                            "url": kg.get("descriptionLink", ""),
                            "title": kg.get("title", ""),
                            "description": kg.get("description", "")
                        })
                
                return results[:max_results]  # Ensure we only return the requested number
                
            # Fall back to DuckDuckGo if Serper API key not available
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = await self.http_client.get(url)
            data = response.json()
            
            results = []
            # Extract organic results
            for result in data.get("Results", [])[:max_results]:
                results.append({
                    "url": result.get("FirstURL", ""),
                    "title": result.get("Text", ""),
                    "description": ""
                })
            
            # Also include related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "FirstURL" in topic:
                    results.append({
                        "url": topic.get("FirstURL", ""),
                        "title": topic.get("Text", ""),
                        "description": ""
                    })
            
            return results
                
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            # Fallback to a simple Google search page scraping as last resort
            try:
                search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = await self.http_client.get(search_url, headers=headers)
                
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                
                for g in soup.find_all('div', class_='g'):
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text if g.find('h3') else "No title"
                        results.append({
                            "url": link,
                            "title": title,
                            "description": ""
                        })
                        
                        if len(results) >= max_results:
                            break
                
                return results
            except Exception as e2:
                logger.error(f"Error in fallback search: {e2}")
                return []

    async def process_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Process different content types from the web"""
        try:
            # Determine content type
            if "youtube.com" in url or "youtu.be" in url:
                return await self._extract_youtube(url)
            elif url.endswith(".pdf"):
                return await self._extract_pdf(url)
            else:
                return await self._extract_webpage(url)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    async def _extract_youtube(self, url: str) -> Dict[str, Any]:
        """Extract content from YouTube videos"""
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True, language=["en", "hi"]
            )
            docs = loader.load()
            
            if not docs:
                # Fallback to pytube
                yt = YouTube(url)
                transcript = yt.captions.get_by_language_code('en')
                
                if transcript:
                    content = transcript.generate_srt_captions()
                else:
                    content = yt.description
                
                return {
                    "content": content,
                    "source": url,
                    "domain": "youtube.com",
                    "title": yt.title,
                    "type": "video"
                }
            
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            metadata = docs[0].metadata if docs else {}
            
            return {
                "content": combined_text,
                "source": url,
                "domain": "youtube.com",
                "title": metadata.get("title", "YouTube Video"),
                "type": "video"
            }
        except Exception as e:
            logger.error(f"Error extracting YouTube content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": "youtube.com",
                "title": "YouTube Video",
                "type": "video"
            }
    
    async def _extract_pdf(self, url: str) -> Dict[str, Any]:
        """Extract content from PDF documents with OCR"""
        try:
            response = await self.http_client.get(url)
            
            # Download the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            text_content = ""
            try:
                # First try to extract text directly
                doc = fitz.open(tmp_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text_content += page.get_text()
                doc.close()
                
                # If no text extracted, use OCR
                if not text_content.strip():
                    doc = fitz.open(tmp_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img = Image.open(BytesIO(pix.tobytes()))
                        text_content += pytesseract.image_to_string(img)
                    doc.close()
            finally:
                os.unlink(tmp_path)
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": text_content,
                "source": url,
                "domain": domain,
                "title": f"PDF Document from {domain}",
                "type": "pdf"
            }
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": tldextract.extract(url).domain,
                "title": "PDF Document",
                "type": "pdf"
            }
    
    async def _extract_webpage(self, url: str) -> Dict[str, Any]:
        """Extract content from webpages"""
        try:
            # Skip specific problematic domains
            if any(domain in url for domain in ["rtionline.up.gov.in"]):
                return None
                
            # Use a custom loader with SSL verification disabled for government sites
            loader = AsyncHtmlLoader([url], verify_ssl=False)
            docs = await loader.aload()
            
            if not docs:
                return None
                
            transformer = Html2TextTransformer()
            docs_transformed = transformer.transform_documents(docs)
            
            if not docs_transformed:
                return None
                
            content = docs_transformed[0].page_content
            metadata = docs_transformed[0].metadata
            
            # Extract title and other metadata using BeautifulSoup
            soup = BeautifulSoup(docs[0].page_content, "html.parser")
            title = soup.title.text if soup.title else url
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": content,
                "source": url,
                "domain": domain,
                "title": title,
                "type": "webpage"
            }
        except Exception as e:
            logger.error(f"Error extracting webpage content: {e}")
            return None

class LegalRAGSystem:
    """Base RAG system for legal queries"""
    
    def __init__(self, vectorstore):
        # Use vectorstore retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        # Configure models
        self.general_llm = ChatGroq(
            temperature=0.3,
            model_name="deepseek-r1-distill-llama-70b",
            max_tokens=4096
        )
        
        # Switch to OpenAI for GPT-4o with reduced token limits
        self.legal_specialist = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4o-mini",
            max_tokens=2000,  # Reduced from 4096 to leave more room for input
            request_timeout=120
        )
        
        # Setup improved prompt with better context handling
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal assistant specializing in Indian law. 
            Analyze the following legal context and provide a detailed, accurate response with proper citations.
            
            If information is not available in the context, clearly state that rather than making up an answer.
            
            Use markdown formatting for better readability. When citing cases, use proper citation format.
            Keep your response concise and focused on answering the user's question directly.
            """),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])
        
        # Create chain
        self.qa_chain = create_stuff_documents_chain(
            self.legal_specialist, 
            self.qa_prompt
        )
        
        # Set up token counter with fallback mechanism
        try:
            # Try to use the cl100k_base tokenizer which is more reliably available
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Using cl100k_base tokenizer")
        except (ImportError, Exception) as e:
            logger.warning(f"Error loading tiktoken: {e}")
            # Define a simple fallback tokenizer that estimates tokens
            self.tokenizer = None
            logger.info("Using fallback token estimator")
        
        # Set maximum tokens for input context (leaving room for model output and prompt)
        self.max_input_tokens = 15000  # Lower limit for safety
    
    # Helper function to count tokens in text
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting error: {e}")
                # Fall back to simple estimation
                return len(text.split()) * 1.5  # Rough estimate
        else:
            # Simple estimation: 1 token ~= 4 characters in English
            return len(text) // 4

class EnhancedLegalRAGSystem(LegalRAGSystem):
    """Advanced RAG system with web search and multimedia capabilities"""
    
    def __init__(self, vectorstore, enable_web=True):
        super().__init__(vectorstore)
        self.web_retriever = WebRetriever()
        self.enable_web = enable_web
        self.cache = {}
        self.ik_api = IKAPIWrapper()
    
    async def query(self, question: str, use_web: bool = True):
        # Check vector DB first - if we have enough results, we might not need web search
        local_docs = self.retriever.invoke(question)
        have_sufficient_local_docs = len(local_docs) >= 5
        
        web_docs = []
        if self.enable_web and use_web and not have_sufficient_local_docs:
            # Generate optimized search queries - but limit to 1-2 queries max
            try:
                # Check cache first for efficiency
                cache_key = question.lower()
                if cache_key in self.cache:
                    logger.info("Using cached web results")
                    web_docs = self.cache[cache_key]
                else:
                    # Create a more targeted search query
                    if "rti" in question.lower():
                        search_query = "RTI filing procedure India official"
                    else:
                        # Generate a single optimized query
                        search_prompt = ChatPromptTemplate.from_template(
                            "Create ONE specific search query to find information about: {question}. "
                            "Focus on Indian legal context. Return only the search query text, nothing else."
                        )
                        chain = search_prompt | self.general_llm
                        result = chain.invoke({"question": question})
                        search_query = result.content.strip().strip('"\'')
                    
                    logger.info(f"Using optimized search query: {search_query}")
                    
                    # Process the search query
                    results = await self.web_retriever.search_and_process(search_query)
                    
                    # If we don't have enough results, try a second query
                    if len(results) < 3:
                        # Create a more general fallback query
                        fallback_query = question.replace("?", "")
                        additional_results = await self.web_retriever.search_and_process(fallback_query)
                        
                        # Add new results only
                        seen_urls = {r.get("source") for r in results}
                        for res in additional_results:
                            if res.get("source") not in seen_urls:
                                results.append(res)
                                seen_urls.add(res.get("source"))
                    
                    # Cache the results
                    self.cache[cache_key] = results
                    web_docs = results
            except Exception as e:
                logger.error(f"Error in web retrieval: {e}")
                # Continue with whatever local docs we have
        
        # Convert web results to Document objects
        web_documents = []
        for doc in web_docs:
            web_documents.append(
                Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        "source": doc.get("source", ""),
                        "title": doc.get("title", ""),
                        "domain": doc.get("domain", ""),
                        "type": doc.get("type", "web")
                    }
                )
            )
        
        # Combine and rank all documents
        all_docs = local_docs + web_documents
        ranked_docs = self._rank_documents(question, all_docs)
        
        # NEW: Manage token count to avoid rate limits
        token_budget = self.max_input_tokens
        question_tokens = self._count_tokens(question)
        token_budget -= question_tokens
        
        # Calculate tokens for prompt template (fixed approximation)
        # Instead of trying to access message components directly, use a simpler approach
        system_prompt = "You are a legal assistant specializing in Indian law. Analyze the following legal context and provide a detailed, accurate response with proper citations."
        human_prompt_template = "Context:\n{context}\n\nQuestion: {input}"
        prompt_template_tokens = self._count_tokens(system_prompt) + self._count_tokens(human_prompt_template)
        token_budget -= prompt_template_tokens
        
        # Reserve tokens for the model's response
        token_budget -= 2000  # Response tokens
        
        # Calculate how many documents we can include
        docs_for_context = []
        current_tokens = 0
        
        for doc in ranked_docs:
            doc_tokens = self._count_tokens(doc.page_content)
            
            # Skip extremely large documents
            if doc_tokens > token_budget * 0.8:
                # Truncate very large documents
                try:
                    if self.tokenizer:
                        max_tokens = int(token_budget * 0.5)  # Use at most 50% of budget for a single doc
                        truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                        doc.page_content = truncated_content + "... [content truncated due to length]"
                    else:
                        # Simple truncation for fallback case
                        max_chars = int(token_budget * 0.5 * 4)  # Rough char estimate
                        doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                except Exception as e:
                    logger.warning(f"Error truncating document: {e}")
                    # Simple truncation fallback
                    doc.page_content = doc.page_content[:5000] + "... [content truncated]"
                
                doc_tokens = self._count_tokens(doc.page_content)
            
            # Check if adding this document would exceed our budget
            if current_tokens + doc_tokens > token_budget:
                # If we already have some docs, stop here
                if docs_for_context:
                    break
                # If this is the first doc, truncate it to fit
                else:
                    try:
                        if self.tokenizer:
                            max_tokens = token_budget - 100  # Leave a small buffer
                            truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                            doc.page_content = truncated_content + "... [content truncated due to length]"
                        else:
                            # Simple truncation for fallback case
                            max_chars = int((token_budget - 100) * 4)  # Rough char estimate
                            doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                    except Exception as e:
                        logger.warning(f"Error truncating document: {e}")
                        # Simple truncation fallback
                        doc.page_content = doc.page_content[:3000] + "... [content truncated]"
                    
                    doc_tokens = self._count_tokens(doc.page_content)
            
            docs_for_context.append(doc)
            current_tokens += doc_tokens
            
            # If we've exceeded 70% of our budget, stop adding more
            if current_tokens > token_budget * 0.7:
                break
        
        logger.info(f"Using {current_tokens} tokens for context out of {self.max_input_tokens} maximum")
        logger.info(f"Selected {len(docs_for_context)} documents out of {len(ranked_docs)} available")
        
        # Generate response with token-managed context
        try:
            return await self.qa_chain.ainvoke({
                "input": question,
                "context": docs_for_context
            })
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Attempt a more reliable fallback with even fewer docs if needed
            if len(docs_for_context) > 2:
                logger.info("Trying fallback with fewer documents")
                return await self.qa_chain.ainvoke({
                    "input": question,
                    "context": docs_for_context[:2]  # Use only top 2 docs
                })
            raise  # Re-raise the exception if we can't recover
    
    def _rank_documents(self, question: str, docs: List[Document]) -> List[Document]:
        """Rank documents by relevance and source authority"""
        domain_scores = {
            "indiankanoon.org": 5.0,
            "gov.in": 4.5,
            "nic.in": 4.0,
            "sci.gov.in": 5.0,  # Supreme Court
            "hc.gov.in": 4.5,   # High Courts
            "edu": 3.5,
            "org": 3.0,
            "com": 2.0
        }
        
        def get_domain_score(doc):
            source = doc.metadata.get("source", "")
            domain = doc.metadata.get("domain", "")
            
            # Check for exact domain matches
            for key, score in domain_scores.items():
                if key in source or key in domain:
                    return score
            
            # Get TLD score
            parts = domain.split(".")
            if len(parts) > 1:
                tld = parts[-1]
                if tld in domain_scores:
                    return domain_scores[tld]
            
            return 1.0
        
        # Calculate semantic similarity for each document
        for doc in docs:
            text = doc.page_content[:1000]  # limit text length for efficiency
            try:
                doc.metadata["relevance_score"] = self._semantic_similarity(question, text)
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                doc.metadata["relevance_score"] = 0.0
            
            doc.metadata["domain_score"] = get_domain_score(doc)
            doc.metadata["combined_score"] = (
                doc.metadata["relevance_score"] * 0.7 + 
                doc.metadata["domain_score"] * 0.3
            )
        
        # Sort by combined score
        ranked_docs = sorted(
            docs, 
            key=lambda x: x.metadata.get("combined_score", 0),
            reverse=True
        )
        
        # Keep top 10 documents
        return ranked_docs[:10]
    
    def _semantic_similarity(self, query: str, text: str) -> float:
        """Calculate semantic similarity between query and text"""
        try:
            query_embedding = embeddings.embed_query(query)  # type: ignore
            text_embedding = embeddings.embed_query(text)  # type: ignore
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error in semantic similarity: {e}")
            return 0.0
``` 