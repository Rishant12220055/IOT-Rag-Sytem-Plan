#!/bin/bash

echo "Starting IoT Sensor Data RAG System..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Streamlit application..."
streamlit run streamlit_app.py
