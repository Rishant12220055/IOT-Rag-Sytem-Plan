---
title: IoT Sensor Data RAG for Smart Buildings
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
pinned: false
license: mit
python_version: 3.9
---

# IoT Sensor Data RAG for Smart Buildings

A comprehensive Retrieval-Augmented Generation (RAG) system that processes IoT sensor data, maintenance manuals, and building specifications to provide predictive maintenance insights and operational optimization for smart buildings.

## Features

- **Real-time IoT Sensor Data Processing**: Ingests and processes sensor data from HVAC, electrical, and security systems
- **Document Integration**: Processes maintenance manuals and building specifications  
- **Predictive Maintenance**: ML-powered failure prediction and maintenance scheduling
- **Anomaly Detection**: Real-time detection of unusual sensor patterns
- **Operational Optimization**: Energy efficiency and system performance recommendations
- **Interactive Dashboard**: Streamlit-based web interface for monitoring and querying

## Quick Start

The application will start automatically when deployed to HuggingFace Spaces. Simply interact with the web interface to:

1. **Generate IoT Data** - Create realistic sensor readings
2. **Run Anomaly Detection** - Identify unusual patterns
3. **Predict Maintenance** - Get maintenance recommendations
4. **Ask Questions** - Query the RAG system about building operations

## Architecture

- **Frontend**: Streamlit web application
- **Vector Database**: ChromaDB for document embeddings
- **Embeddings**: HuggingFace Sentence Transformers
- **ML Models**: Scikit-learn for anomaly detection and predictive maintenance
- **Data Processing**: Pandas and NumPy for sensor data analysis
