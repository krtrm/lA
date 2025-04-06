#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Set environment variables explicitly
export OPENAI_API_KEY=$(grep -o 'OPENAI_API_KEY=.*' .env | cut -d'=' -f2)
export PINECONE_API_KEY=$(grep -o 'PINECONE_API_KEY=.*' .env | cut -d'=' -f2)
export PINECONE_REGION=$(grep -o 'PINECONE_REGION=.*' .env | cut -d'=' -f2)
export USER_AGENT="Vaqeel-Legal-AI-Assistant/1.0"

# Check if environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo "ERROR: PINECONE_API_KEY is not set in .env file"
    exit 1
fi

# Set default Pinecone region if not specified
if [ -z "$PINECONE_REGION" ]; then
    export PINECONE_REGION="gcp-starter"
    echo "Using default Pinecone region: gcp-starter"
fi

# Automatically create books directory if it doesn't exist
BOOKS_DIR="./backend/books"
mkdir -p $BOOKS_DIR
echo "Books directory: $BOOKS_DIR"

# First test if env variables are loaded properly
echo "Testing environment variables..."
python3 test_env.py

if [ $? -ne 0 ]; then
    echo "Environment variable test failed. Please check your .env file"
    exit 1
fi

# Run embedding script
echo "Starting embedding process..."
python3 backend/embed_books.py --books-folder $BOOKS_DIR

echo "If you want to specify a different books folder, use:"
echo "python3 backend/embed_books.py --books-folder /path/to/your/books"
