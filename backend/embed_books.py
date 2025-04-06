#!/usr/bin/env python
import os
import sys
import argparse
import logging
from dotenv import load_dotenv, find_dotenv

# Setup logging first to capture all issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file with explicit path for reliability
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
logger.info(f"Looking for .env file at: {env_path}")

if os.path.exists(env_path):
    loaded = load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded .env file: {loaded}")
else:
    # Try to find .env file automatically
    env_file = find_dotenv()
    if env_file:
        loaded = load_dotenv(dotenv_path=env_file)
        logger.info(f"Found and loaded .env file from: {env_file}")
    else:
        logger.warning("No .env file found. Will try to use environment variables.")

# Check if OPENAI_API_KEY is loaded
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    logger.error("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
    sys.exit(1)
else:
    masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
    logger.info(f"OPENAI_API_KEY is set (starts with {masked_key})")

# Verify other required keys
pinecone_key = os.getenv("PINECONE_API_KEY")
if not pinecone_key:
    logger.error("PINECONE_API_KEY environment variable is not set. Please check your .env file.")
    sys.exit(1)
else:
    masked_key = pinecone_key[:8] + "..." + pinecone_key[-4:] if len(pinecone_key) > 12 else "***"
    logger.info(f"PINECONE_API_KEY is set (starts with {masked_key})")

# Import modules after environment variables are confirmed to be loaded
from rag_system import create_vector_store

def main():
    parser = argparse.ArgumentParser(description='Embed books and other documents into the vector database')
    parser.add_argument('--books-folder', type=str, default='./books', 
                        help='Path to the books folder containing PDFs')
    parser.add_argument('--other-docs', type=str, nargs='*', default=[], 
                        help='Additional document paths to include')
    parser.add_argument('--index-name', type=str, default='legal-documents',
                        help='Name of the Pinecone index')
    parser.add_argument('--namespace', type=str, default='indian-law',
                        help='Namespace within the Pinecone index')
    
    args = parser.parse_args()
    
    # Convert books folder to absolute path for clarity in logs
    args.books_folder = os.path.abspath(args.books_folder)
    
    # Validate books folder exists
    if not os.path.exists(args.books_folder):
        logger.error(f"Books folder '{args.books_folder}' does not exist")
        # Create the directory instead of failing
        try:
            os.makedirs(args.books_folder)
            logger.info(f"Created books folder: {args.books_folder}")
        except Exception as e:
            logger.error(f"Failed to create books folder: {e}")
            return
    
    # Check if there are PDFs in the books folder
    pdf_count = len([f for f in os.listdir(args.books_folder) if f.lower().endswith('.pdf')])
    if pdf_count == 0:
        logger.warning(f"No PDF files found in books folder: {args.books_folder}")
    else:
        logger.info(f"Found {pdf_count} PDF files in books folder")
    
    # Create the vector store with books prioritized
    logger.info("Creating vector store with prioritized books...")
    try:
        vectorstore = create_vector_store(
            doc_paths=args.other_docs,
            index_name="llama-text-embed-v2-index",
            namespace="indian-law",
            books_folder="backend/books"
        )
        logger.info("Vector store creation complete!")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
