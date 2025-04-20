#!/bin/bash

# start_api.sh - Starts the backend API server with proper environment setup

# Change to the project root directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Check for .env file and create from template if not found
if [ ! -f .env ]; then
    echo "No .env file found. Creating from .env.example template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys and configuration!"
    exit 1
fi

# Ensure virtual environment is active or create it
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r backend/requirements.txt
else
    source venv/bin/activate
fi

# Check for required packages
if ! pip list | grep -q "fastapi"; then
    echo "Installing dependencies..."
    pip install -r backend/requirements.txt
fi

# Set up environment variables
export USER_AGENT="Vaqeel-Legal-AI-Assistant/1.0"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Verify needed directories exist
mkdir -p $(pwd)/books
mkdir -p $(pwd)/ik_data

# Make sure the ik_data directory exists
mkdir -p ik_data

# Start the API server
echo "Starting Vaqeel.app backend API server..."
cd backend
python api.py
