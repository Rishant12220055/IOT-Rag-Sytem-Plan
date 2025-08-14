#!/usr/bin/env python3
"""
Test script to verify rag_pipeline.py imports and initialization work correctly
"""

import os
import sys

# Add project root to path for proper package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_rag_pipeline_imports():
    """Test that rag_pipeline imports work correctly"""
    print("Testing RAG pipeline imports...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        print("✓ RAGPipeline import successful")
        
        # Test that OpenAI import handling works
        from src.rag_pipeline import OPENAI_AVAILABLE
        print(f"✓ OpenAI availability check: {OPENAI_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_rag_pipeline_initialization():
    """Test that RAG pipeline can be initialized"""
    print("Testing RAG pipeline initialization...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        from src.document_processor import DocumentProcessor
        
        # Create a mock document processor
        doc_processor = DocumentProcessor()
        
        # Test initialization without OpenAI
        rag_pipeline = RAGPipeline(doc_processor, use_openai=False)
        print("✓ RAGPipeline initialized successfully without OpenAI")
        
        # Test initialization with OpenAI (should fallback to local)
        rag_pipeline_with_openai = RAGPipeline(doc_processor, use_openai=True)
        print(f"✓ RAGPipeline initialized with use_openai=True (actual use_openai: {rag_pipeline_with_openai.use_openai})")
        
        # Cleanup
        doc_processor.cleanup()
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running RAG pipeline validation tests...\n")
    
    success = True
    success &= test_rag_pipeline_imports()
    print()
    success &= test_rag_pipeline_initialization()
    
    print()
    if success:
        print("✅ All RAG pipeline tests passed!")
        print("The RAG pipeline is ready for use.")
    else:
        print("❌ Some RAG pipeline tests failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
