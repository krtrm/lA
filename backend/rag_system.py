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
from typing import List, Dict, Any, Optional
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
from langchain.chains.combine_documents import create_stuff_documents_chain  # Add missing import
import tiktoken  # Add this import for token counting

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
    def embed_query(self, text: str) -> list:
        vec = super().embed_query(text)
        return vec[:1024]
    def embed_documents(self, texts: list) -> list:
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
            pass
        
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

    async def process_content(self, url: str) -> Optional[Dict]:
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
    
    async def _extract_youtube(self, url: str) -> Dict:
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
    
    async def _extract_pdf(self, url: str) -> Dict:
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
    
    async def _extract_webpage(self, url: str) -> Dict:
        """Extract content from webpages"""
        try:
            # Skip specific problematic domains
            if any(domain in url for domain in ["rtionline.up.gov.in"]):
                return None
                
            # Use a custom loader with SSL verification disabled for government sites
            loader = AsyncHtmlLoader([url], verify_ssl=False)
            docs = loader.load()
            
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
    
    async def search_and_process(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web and process results with improved efficiency"""
        # Set a reasonable limit to avoid excessive processing
        max_pages_to_process = min(max_results, self.max_urls_per_query)
        
        # Get search results
        search_results = await self.search_web(query, max_pages_to_process)
        if not search_results:
            return []
            
        # Filter out URLs we've already processed
        filtered_results = [r for r in search_results if r["url"] not in self.processed_urls]
        if not filtered_results:
            return []
            
        # Update processed URLs set
        for result in filtered_results:
            self.processed_urls.add(result["url"])
        
        # Process each URL with a timeout
        tasks = []
        for result in filtered_results[:max_pages_to_process]:
            tasks.append(asyncio.wait_for(
                self.process_content(result["url"]), 
                timeout=10
            ))
        
        # Execute tasks in parallel with error handling
        results = []
        if tasks:
            try:
                # Use gather with return_exceptions to handle errors gracefully
                gathered_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out errors and None results
                for result in gathered_results:
                    if isinstance(result, Exception):
                        continue
                    if result is not None:
                        results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"Some content processing timed out for query: {query}")
        
        # Add legal documents from Indian Kanoon only for legal queries
        if any(term in query.lower() for term in ["law", "legal", "act", "section", "court", "rti", "right"]):
            try:
                legal_docs = self.ik_api.search_case_law(query, max_results=2)
                results.extend(legal_docs)
            except Exception as e:
                logger.error(f"Error retrieving legal documents: {e}")
        
        return results

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
            query_embedding = embeddings.embed_query(query)
            text_embedding = embeddings.embed_query(text)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error in semantic similarity: {e}")
            return 0.0