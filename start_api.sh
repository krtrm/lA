#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Set up environment variables
export USER_AGENT="Vaqeel-Legal-AI-Assistant/1.0"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Verify needed directories exist
mkdir -p $(pwd)/books
mkdir -p $(pwd)/ik_data

# Start the FastAPI server
echo "Starting Vaqeel.app API server..."
python3 main.py
