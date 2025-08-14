"""
Test script to verify the IoT Sensor RAG system components
"""
import sys
import os
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data_ingestion import data_generator, data_processor
        print("✅ Data ingestion module imported successfully")
    except Exception as e:
        print(f"❌ Data ingestion import failed: {e}")
        return False
    
    try:
        from src.document_processor import DocumentProcessor
        print("✅ Document processor module imported successfully")
    except Exception as e:
        print(f"❌ Document processor import failed: {e}")
        return False
    
    try:
        from src.anomaly_detection import anomaly_detector, alert_system
        print("✅ Anomaly detection module imported successfully")
    except Exception as e:
        print(f"❌ Anomaly detection import failed: {e}")
        return False
    
    try:
        from src.predictive_maintenance import predictive_maintenance
        print("✅ Predictive maintenance module imported successfully")
    except Exception as e:
        print(f"❌ Predictive maintenance import failed: {e}")
        return False
    
    try:
        from src.rag_pipeline import RAGPipeline
        print("✅ RAG pipeline module imported successfully")
    except Exception as e:
        print(f"❌ RAG pipeline import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality"""
    print("\nTesting data generation...")
    
    try:
        from src.data_ingestion import data_generator, data_processor, simulate_real_time_data
        
        # Generate sample data
        readings = data_generator.generate_batch_readings(10)
        print(f"✅ Generated {len(readings)} sensor readings")
        
        # Add to processor
        for reading in readings:
            data_processor.add_reading(reading)
        
        # Get latest readings
        latest = data_processor.get_latest_readings(5)
        print(f"✅ Retrieved {len(latest)} latest readings")
        
        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_document_processing():
    """Test document processing functionality"""
    print("\nTesting document processing...")
    
    try:
        from src.document_processor import DocumentProcessor, create_sample_documents
        
        # Create sample documents
        hvac_manual, building_specs = create_sample_documents()
        print("✅ Sample documents created")
        
        # Initialize processor
        doc_processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        return True
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("\nTesting anomaly detection...")
    
    try:
        from src.data_ingestion import data_generator
        from src.anomaly_detection import anomaly_detector
        import pandas as pd
        
        # Generate test data
        historical_data = data_generator.generate_historical_data(days=7)
        print(f"✅ Generated {len(historical_data)} historical data points")
        
        # Test anomaly detection
        anomaly_detector.set_static_thresholds()
        detected_anomalies = anomaly_detector.detect_threshold_anomalies(historical_data)
        print("✅ Anomaly detection completed")
        
        return True
    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline functionality"""
    print("\nTesting RAG pipeline...")
    
    try:
        from src.document_processor import DocumentProcessor
        from src.rag_pipeline import RAGPipeline
        
        # Initialize components
        doc_processor = DocumentProcessor()
        rag_pipeline = RAGPipeline(doc_processor)
        print("✅ RAG pipeline initialized")
        
        return True
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 IoT Sensor RAG System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_document_processing,
        test_anomaly_detection,
        test_rag_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        print("\n🚀 To run the application:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()
