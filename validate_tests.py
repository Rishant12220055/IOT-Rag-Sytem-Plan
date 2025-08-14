"""
Test Runner for IoT Sensor Data RAG System
Runs a subset of tests to validate system functionality
"""
import sys
import os
sys.path.append('src')

def run_basic_validation():
    """Run basic validation of the test system"""
    print("🧪 Running Basic Test Validation")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.data_ingestion import data_generator, data_processor
        from src.document_processor import DocumentProcessor
        from src.anomaly_detection import anomaly_detector
        from src.rag_pipeline import RAGPipeline
        print("✅ All imports successful")
        
        # Test data generation
        print("2. Testing data generation...")
        readings = data_generator.generate_batch_readings(5)
        assert len(readings) == 5, "Data generation failed"
        print("✅ Data generation working")
        
        # Test document processing
        print("3. Testing document processor...")
        doc_processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test RAG pipeline
        print("4. Testing RAG pipeline...")
        rag_pipeline = RAGPipeline(doc_processor, use_openai=False)
        print("✅ RAG pipeline initialized")
        
        print("\n🎉 Basic validation complete - Test system ready!")
        print("\n📝 To run full test suite:")
        print("   pytest test_smart_building_rag.py -v")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = run_basic_validation()
    sys.exit(0 if success else 1)
