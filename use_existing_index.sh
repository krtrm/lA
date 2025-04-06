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

# Run the initialization script to test connection to existing index
echo "Initializing RAG system with existing index..."
python3 init_rag.py

# Make the script executable
chmod +x init_rag.py
