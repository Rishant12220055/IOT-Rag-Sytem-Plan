#!/bin/bash

# HuggingFace Spaces deployment setup script
echo "🚀 Setting up IoT Sensor Data RAG System for HuggingFace Spaces..."

# Install required packages
pip install -r requirements.txt

# Create necessary directories
mkdir -p chroma_db
mkdir -p data/documents

# Initialize the system
echo "✅ Installation complete!"
echo "🏢 IoT Sensor Data RAG System is ready!"

# Start the application
streamlit run streamlit_app.py --server.headless true --server.enableCORS false --server.enableXsrfProtection false
