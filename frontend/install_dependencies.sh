#!/bin/bash

echo "Installing Vaqeel frontend dependencies..."

# Check if npm is available
if command -v npm &> /dev/null; then
    echo "Using npm to install dependencies..."
    npm install react-hot-toast axios
    echo "Dependencies installed successfully!"
# If npm not available, try yarn
elif command -v yarn &> /dev/null; then
    echo "Using yarn to install dependencies..."
    yarn add react-hot-toast axios
    echo "Dependencies installed successfully!"
else
    echo "Error: Neither npm nor yarn found. Please install Node.js and npm/yarn first."
    exit 1
fi

echo "Run 'npm run dev' or 'yarn dev' to start the development server"
