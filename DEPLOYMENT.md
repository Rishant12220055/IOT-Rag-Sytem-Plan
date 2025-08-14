# IoT Sensor Data RAG System - Deployment Guide

## 🚀 Quick Start

1. **Navigate to the project directory:**
   ```bash
   cd "C:\Users\Rishant\OneDrive\Desktop\my project\neversparks assignment\iot-sensor-rag"
   ```

2. **Activate the virtual environment:**
   - Windows: `.\venv\Scripts\Activate.ps1`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser to:** `http://localhost:8501`

## 📦 Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

**Required files for Streamlit Cloud:**
- ✅ `streamlit_app.py` (main application)
- ✅ `requirements.txt` (dependencies)
- ✅ `README.md` (documentation)

### 2. HuggingFace Spaces

1. Create account on [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit framework
3. Upload project files or connect GitHub repo
4. Space will auto-deploy!

### 3. Local Development

Use the provided scripts:
- **Windows:** Double-click `run.bat`
- **Linux/Mac:** Execute `./run.sh`

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** The system works perfectly without OpenAI API key using local models.

## 📊 Features Implemented

### ✅ IoT Data Ingestion & Processing
- Real-time sensor data simulation
- Multi-sensor data fusion
- Time-series data processing
- Zone-based monitoring

### ✅ Document Processing & RAG
- PDF/DOCX/TXT document ingestion
- Intelligent chunking strategies
- ChromaDB vector storage
- HuggingFace embeddings
- Context-aware retrieval

### ✅ Anomaly Detection
- Rule-based threshold detection
- ML-based anomaly detection (Isolation Forest)
- Real-time alert system
- Multi-criteria analysis

### ✅ Predictive Maintenance
- Equipment health scoring
- Failure probability prediction
- Maintenance scheduling
- Priority-based recommendations

### ✅ Interactive Dashboard
- Real-time sensor monitoring
- RAG query interface
- Anomaly detection alerts
- Maintenance recommendations
- Document search capabilities
- Analytics and metrics

## 🎯 Key Components

1. **Data Ingestion** (`src/data_ingestion.py`)
   - IoT sensor data generation
   - Real-time data streaming simulation
   - Multi-sensor correlation

2. **Document Processing** (`src/document_processor.py`)
   - Document parsing and chunking
   - Vector embedding generation
   - ChromaDB integration

3. **Anomaly Detection** (`src/anomaly_detection.py`)
   - Statistical and ML-based detection
   - Alert system management
   - Threshold configuration

4. **Predictive Maintenance** (`src/predictive_maintenance.py`)
   - Equipment health assessment
   - Failure prediction models
   - Maintenance scheduling

5. **RAG Pipeline** (`src/rag_pipeline.py`)
   - Query processing
   - Document retrieval
   - Response generation

6. **Streamlit App** (`streamlit_app.py`)
   - Interactive web interface
   - Real-time dashboards
   - User query handling

## 📈 Evaluation Metrics

The system tracks:
- **Retrieval Accuracy**: Document relevance scores
- **Response Latency**: Query processing time
- **Anomaly Detection**: Precision/recall rates
- **System Performance**: Data quality and uptime

## 🔗 Public Links

- **GitHub Repository:** [Your GitHub URL]
- **Live Demo:** [Your Streamlit Cloud URL]
- **Documentation:** Complete README with setup instructions

## 🛠️ Technical Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.8+
- **Vector DB:** ChromaDB
- **ML Framework:** Scikit-learn
- **Embeddings:** HuggingFace Sentence Transformers
- **Visualization:** Plotly
- **Document Processing:** PyPDF2, python-docx

## 🎉 System Highlights

1. **Complete RAG Implementation**
   - Document ingestion and chunking
   - Vector storage and retrieval
   - Context-aware response generation

2. **Real-time IoT Processing**
   - Sensor data simulation and ingestion
   - Multi-sensor data fusion
   - Real-time anomaly detection

3. **Predictive Analytics**
   - Equipment failure prediction
   - Maintenance optimization
   - Operational efficiency insights

4. **Production-Ready**
   - Clean, modular code structure
   - Comprehensive documentation
   - Easy deployment options
   - Scalable architecture

## 📞 Support

For questions or issues:
1. Check the README.md for detailed setup instructions
2. Review the code documentation in each module
3. Test the system with provided sample data

**System Status:** ✅ Ready for deployment and demonstration!
