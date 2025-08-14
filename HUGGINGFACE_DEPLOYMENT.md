# HuggingFace Spaces Deployment Instructions

## 🚀 Deploy to HuggingFace Spaces

### Step 1: Create HuggingFace Spaces Repository

1. **Go to HuggingFace Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Fill in the details**:
   - **Space name**: `iot-sensor-rag-system`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Visibility**: Public
4. **Click "Create Space"**

### Step 2: Clone and Setup

```bash
# Clone your new HF Spaces repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/iot-sensor-rag-system
cd iot-sensor-rag-system

# Copy all files from your local project (except .git folder)
# Copy everything from: C:\Users\Rishant\OneDrive\Desktop\my project\neversparks assignment\iot-sensor-rag\
```

### Step 3: Required Files for HuggingFace Spaces

1. **Rename README_HF.md to README.md** (HF Spaces uses README.md for the app card)
2. **Ensure these files are present**:
   - `streamlit_app.py` (main app file)
   - `requirements.txt` (dependencies)
   - `README.md` (app description with metadata)
   - All `src/` files
   - All `data/` files

### Step 4: Update Requirements for HuggingFace

The current `requirements.txt` should work, but ensure these versions are compatible:

```
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.24.3
scikit-learn>=1.3.0
chromadb>=0.4.15
sentence-transformers>=2.2.2
plotly>=5.17.0
pypdf>=3.17.0
python-docx>=0.8.11
python-dotenv>=1.0.0
```

### Step 5: Push to HuggingFace Spaces

```bash
# Add all files
git add .

# Commit with descriptive message
git commit -m "Deploy IoT Sensor Data RAG System to HuggingFace Spaces

- Complete Streamlit application with IoT sensor processing
- RAG pipeline with ChromaDB and HuggingFace embeddings  
- Predictive maintenance and anomaly detection
- Interactive dashboard with real-time visualizations
- 15/16 tests passing with comprehensive documentation"

# Push to HuggingFace Spaces
git push
```

### Step 6: Monitor Deployment

1. **Go to your Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/iot-sensor-rag-system`
2. **Check the "Logs" tab** for any deployment issues
3. **Wait for the build to complete** (usually 2-5 minutes)
4. **Test the application** once it's live

### Step 7: Update README with Live Link

Once deployed, update your main project README.md with:

```markdown
## 🌐 Live Demo

**Try the live application**: https://huggingface.co/spaces/YOUR_USERNAME/iot-sensor-rag-system
```

---

## Alternative: Streamlit Cloud Deployment

If you prefer Streamlit Cloud:

1. **Go to**: https://share.streamlit.io/
2. **Connect your GitHub repository**: `https://github.com/Rishant12220055/IOT-Rag-Sytem-Plan`
3. **Set main file path**: `streamlit_app.py`
4. **Deploy automatically**

---

## 🎯 Expected Result

Your deployed application will have:

- ✅ **Interactive IoT Dashboard** with real-time sensor data
- ✅ **RAG-powered Q&A System** for building operations
- ✅ **Anomaly Detection Alerts** with visual indicators
- ✅ **Predictive Maintenance Recommendations** 
- ✅ **Professional UI** with charts and data visualization

---

## 🔧 Troubleshooting

### Common Issues:

1. **Build Timeout**: Reduce model sizes or use lighter dependencies
2. **Memory Issues**: Optimize ChromaDB usage and data loading
3. **Import Errors**: Check all file paths are relative and correct
4. **Missing Files**: Ensure all `src/` and `data/` files are included

### Solutions Implemented:

- ✅ **Lightweight embeddings** (all-MiniLM-L6-v2)
- ✅ **Efficient data loading** with error handling
- ✅ **Relative imports** throughout the codebase
- ✅ **Comprehensive error handling** for missing files

---

**Your application is ready for deployment! 🚀**
