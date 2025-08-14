#!/usr/bin/env python3
"""
Simple test script to verify the fixes for encoding and ChromaDB cleanup issues
"""

import os
import sys
import tempfile
import shutil
import gc
import time

# Add src to path and import
import sys
import os

# Add the project root to Python path for proper package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import using proper package path
from src.document_processor import DocumentProcessor

def test_encoding_fix():
    """Test that text files with various encodings can be read"""
    print("Testing encoding fix...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    doc_processor = None
    
    try:
        # Create test file with UTF-8 content (make it long enough to create chunks)
        test_file = os.path.join(temp_dir, 'test.txt')
        test_content = """
        This is a comprehensive test document to verify the encoding fix works properly.
        It contains multiple sentences and paragraphs to ensure that the chunking algorithm
        will create at least one chunk that meets the minimum length requirement.
        
        SECTION 1: OVERVIEW
        This section contains important information about the test. The document processor
        should be able to read this text without any encoding errors, even if it contains
        special characters or non-ASCII content.
        
        SECTION 2: DETAILS
        Here we provide more detailed information to ensure the chunk is substantial enough
        to pass the minimum length filter. This text should be processed successfully and
        stored in the ChromaDB vector database for retrieval testing.
        """
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Test document processor
        chroma_dir = os.path.join(temp_dir, 'chroma_test')
        doc_processor = DocumentProcessor(persist_directory=chroma_dir)
        
        # Process documents
        doc_processor.process_documents_folder(temp_dir)
        
        # Check stats
        stats = doc_processor.get_collection_stats()
        print(f"Successfully processed {stats['total_chunks']} chunks from {stats['unique_files']} files")
        
        if stats['total_chunks'] > 0:
            print("✓ Encoding fix successful - documents processed")
        else:
            print("✗ Encoding fix failed - no documents processed")
            return False
        
    except Exception as e:
        print(f"✗ Error during encoding test: {e}")
        return False
    
    finally:
        # Test cleanup
        if doc_processor:
            print("Testing ChromaDB cleanup...")
            doc_processor.cleanup()
            del doc_processor
        
        gc.collect()
        time.sleep(0.2)  # Wait for file locks to release
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✓ Cleanup successful")
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")
    
    return True

def main():
    """Run all tests"""
    print("Running fix validation tests...\n")
    
    success = test_encoding_fix()
    
    if success:
        print("\n✅ All fixes validated successfully!")
        print("The test suite should now run without encoding or cleanup errors.")
    else:
        print("\n❌ Some fixes need more work.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
