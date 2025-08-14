import os
import re
import hashlib
from typing import List, Dict, Any
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

class DocumentProcessor:
    """Processes maintenance manuals and building specifications for RAG"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="building_documents",
            metadata={"description": "Building maintenance manuals and specifications"}
        )
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            try:
                # Try UTF-8 first, then fallback to other encodings
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        return file.read()
                except Exception as e:
                    try:
                        with open(file_path, 'r', encoding='cp1252') as file:
                            return file.read()
                    except Exception as e2:
                        print(f"Error reading TXT {file_path}: {e2}")
                        return ""
            except Exception as e:
                print(f"Error reading TXT {file_path}: {e}")
                return ""
        else:
            print(f"Unsupported file format: {file_extension}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text_by_sentences(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Chunk text by sentences with overlap"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def chunk_text_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by sections/headings for technical documents"""
        chunks = []
        
        # Split by common section patterns
        section_patterns = [
            r'\n\s*(?:CHAPTER|Chapter)\s+\d+',
            r'\n\s*(?:SECTION|Section)\s+\d+',
            r'\n\s*\d+\.\s+[A-Z][A-Za-z\s]+',
            r'\n\s*[A-Z][A-Z\s]{5,}(?:\n|$)',  # ALL CAPS headings
        ]
        
        sections = [text]  # Start with full text
        
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            sections = new_sections
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 100:  # Only process substantial sections
                section_chunks = self.chunk_text_by_sentences(section, max_chunk_size=800)
                for j, chunk in enumerate(section_chunks):
                    chunks.append({
                        'text': chunk,
                        'section_id': f"section_{i}",
                        'chunk_id': f"section_{i}_chunk_{j}",
                        'length': len(chunk)
                    })
        
        return chunks
    
    def process_document(self, file_path: str, doc_type: str = "manual") -> List[Dict[str, Any]]:
        """Process a single document and return chunks with metadata"""
        print(f"Processing document: {file_path}")
        
        # Extract text
        text = self.extract_text_from_file(file_path)
        if not text:
            return []
        
        # Clean text
        text = self.clean_text(text)
        
        # Chunk text
        chunks_data = self.chunk_text_by_sections(text)
        
        # Add metadata
        file_name = os.path.basename(file_path)
        file_hash = hashlib.md5(text.encode()).hexdigest()
        
        processed_chunks = []
        for chunk_data in chunks_data:
            chunk_id = f"{file_hash}_{chunk_data['chunk_id']}"
            processed_chunks.append({
                'id': chunk_id,
                'text': chunk_data['text'],
                'source_file': file_name,
                'file_path': file_path,
                'doc_type': doc_type,
                'section_id': chunk_data['section_id'],
                'chunk_length': chunk_data['length'],
                'file_hash': file_hash
            })
        
        return processed_chunks
    
    def add_documents_to_vectordb(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to ChromaDB"""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [{
            'source_file': chunk['source_file'],
            'doc_type': chunk['doc_type'],
            'section_id': chunk['section_id'],
            'chunk_length': chunk['chunk_length']
        } for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def process_documents_folder(self, folder_path: str):
        """Process all documents in a folder"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        supported_extensions = ['.pdf', '.docx', '.txt']
        all_chunks = []
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in supported_extensions):
                doc_type = "manual" if "manual" in file_name.lower() else "specification"
                chunks = self.process_document(file_path, doc_type)
                all_chunks.extend(chunks)
        
        if all_chunks:
            self.add_documents_to_vectordb(all_chunks)
            print(f"Processed {len(all_chunks)} total chunks from {folder_path}")
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                # Calculate relevance score: higher score = more relevant
                # Use exponential decay to map distance to 0-1 range
                relevance_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
                
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'relevance_score': relevance_score
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            if count > 0:
                sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
                
                doc_types = {}
                source_files = set()
                
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    source_files.add(metadata.get('source_file', 'unknown'))
                
                return {
                    'total_chunks': count,
                    'document_types': doc_types,
                    'unique_files': len(source_files),
                    'source_files': list(source_files)
                }
            
            return {'total_chunks': 0, 'document_types': {}, 'unique_files': 0, 'source_files': []}
        
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup and close ChromaDB client properly"""
        try:
            # Force garbage collection of chromadb resources
            if hasattr(self, 'collection'):
                del self.collection
            if hasattr(self, 'client'):
                del self.client
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def create_sample_documents():
    """Create sample maintenance manuals and specifications"""
    
    # Sample HVAC Manual
    hvac_manual = """
    HVAC SYSTEM MAINTENANCE MANUAL
    
    CHAPTER 1: SYSTEM OVERVIEW
    The HVAC system consists of air handling units, chillers, boilers, and ductwork distribution. Regular maintenance is critical for optimal performance and energy efficiency.
    
    CHAPTER 2: PREVENTIVE MAINTENANCE SCHEDULE
    
    MONTHLY TASKS:
    - Check and replace air filters if dirty
    - Inspect belts for wear and proper tension
    - Verify thermostat calibration
    - Clean condenser coils
    - Check refrigerant levels
    
    QUARTERLY TASKS:
    - Lubricate motor bearings
    - Inspect electrical connections
    - Test safety controls and alarms
    - Check ductwork for leaks
    - Calibrate temperature sensors
    
    ANNUAL TASKS:
    - Complete system performance analysis
    - Replace worn belts and bearings
    - Clean entire ductwork system
    - Test emergency shutdown procedures
    - Update system documentation
    
    CHAPTER 3: TROUBLESHOOTING GUIDE
    
    HIGH ENERGY CONSUMPTION:
    - Check for dirty filters (replace if needed)
    - Verify proper insulation
    - Inspect for ductwork leaks
    - Check thermostat settings
    - Analyze compressor performance
    
    POOR AIR QUALITY:
    - Replace air filters immediately
    - Check ventilation rates
    - Inspect for mold or contamination
    - Verify CO2 sensor calibration
    - Clean air handling units
    
    TEMPERATURE FLUCTUATIONS:
    - Calibrate temperature sensors
    - Check damper operation
    - Verify control system settings
    - Inspect heating/cooling coils
    - Test zone control valves
    
    CHAPTER 4: SAFETY PROCEDURES
    Always follow lockout/tagout procedures before maintenance. Wear appropriate PPE including safety glasses and gloves. Ensure proper ventilation when working with refrigerants.
    """
    
    # Sample Building Specifications
    building_specs = """
    SMART BUILDING SPECIFICATIONS
    
    SECTION 1: BUILDING OVERVIEW
    This 5-story commercial building features integrated IoT sensors, automated building management systems, and energy-efficient design.
    
    SECTION 2: SENSOR SPECIFICATIONS
    
    TEMPERATURE SENSORS:
    - Range: -40°C to +85°C
    - Accuracy: ±0.5°C
    - Response time: <30 seconds
    - Installation: Every 500 sq ft
    
    HUMIDITY SENSORS:
    - Range: 0-100% RH
    - Accuracy: ±2% RH
    - Calibration: Annual
    - Alert thresholds: <30% or >70%
    
    AIR QUALITY SENSORS:
    - CO2 range: 0-5000 ppm
    - Accuracy: ±30 ppm
    - Alert threshold: >1000 ppm
    - Ventilation trigger: >800 ppm
    
    SECTION 3: ENERGY MANAGEMENT
    
    LIGHTING SYSTEM:
    - LED fixtures with daylight sensors
    - Occupancy-based dimming
    - Target efficiency: 95 lumens/watt
    - Maintenance: Replace at 80% output
    
    ELECTRICAL SYSTEM:
    - Smart meters on each floor
    - Real-time consumption monitoring
    - Demand response capability
    - Peak load management
    
    SECTION 4: MAINTENANCE REQUIREMENTS
    
    SENSOR CALIBRATION:
    - Temperature sensors: Every 6 months
    - Humidity sensors: Annually
    - Air quality sensors: Every 3 months
    - Energy meters: Annually
    
    SYSTEM UPDATES:
    - Firmware updates quarterly
    - Security patches monthly
    - Performance optimization annually
    - Hardware refresh every 5 years
    
    SECTION 5: ALERT THRESHOLDS
    
    CRITICAL ALERTS:
    - Temperature >30°C or <15°C
    - Humidity >80% or <20%
    - CO2 >1500 ppm
    - Energy consumption >120% of baseline
    
    WARNING ALERTS:
    - Temperature >28°C or <18°C
    - Humidity >70% or <30%
    - CO2 >1000 ppm
    - Energy consumption >110% of baseline
    """
    
    return hvac_manual, building_specs
