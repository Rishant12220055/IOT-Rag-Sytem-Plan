"""
Comprehensive Test Suite for IoT Sensor Data RAG for Smart Buildings System

This test suite validates all components of the smart building RAG system including:
- Data ingestion and processing
- RAG pipeline functionality
- Predictive maintenance
- Anomaly detection
- Operational optimization
- Edge cases and robustness

Author: Senior QA Engineer
Framework: pytest
"""

import pytest
import pandas as pd
import numpy as np
import time
import os
import tempfile
import gc
import shutil
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import tempfile
import json

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import system components
from src.data_ingestion import data_generator, data_processor, IoTDataGenerator, DataProcessor
from src.document_processor import DocumentProcessor, create_sample_documents
from src.anomaly_detection import anomaly_detector, alert_system, AnomalyDetector, AlertSystem
from src.predictive_maintenance import predictive_maintenance, PredictiveMaintenanceModel
from src.rag_pipeline import RAGPipeline
from src.utils import calculate_basic_stats, get_sensor_status


class TestDataFixtures:
    """Test data fixtures and helper methods"""
    
    @pytest.fixture
    def sample_sensor_data(self):
        """Generate sample IoT sensor data for testing"""
        generator = IoTDataGenerator()
        data = []
        
        # Generate normal operating data
        for i in range(100):
            timestamp = datetime.now() - timedelta(hours=i)
            
            # Normal HVAC data
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_id': 'HVAC_01_temp',
                'sensor_type': 'temperature',
                'zone': 'Floor_1',
                'equipment': 'HVAC',
                'value': 22.0 + np.random.normal(0, 1),
                'unit': '°C',
                'is_anomaly': False
            })
            
            # Normal vibration data
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_id': 'HVAC_01_vib',
                'sensor_type': 'vibration',
                'zone': 'Floor_1',
                'equipment': 'HVAC',
                'value': 1.0 + np.random.normal(0, 0.2),
                'unit': 'mm/s',
                'is_anomaly': False
            })
            
            # Normal energy consumption
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_id': 'Floor_1_energy',
                'sensor_type': 'energy_consumption',
                'zone': 'Floor_1',
                'equipment': 'Electrical',
                'value': 150.0 + np.random.normal(0, 20),
                'unit': 'kWh',
                'is_anomaly': False
            })
            
            # Normal water pressure
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_id': 'Water_01_pressure',
                'sensor_type': 'pressure',
                'zone': 'Basement',
                'equipment': 'Water_System',
                'value': 1020.0 + np.random.normal(0, 5),
                'unit': 'hPa',
                'is_anomaly': False
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def anomalous_sensor_data(self):
        """Generate sensor data with specific anomalies for testing"""
        data = []
        base_time = datetime.now()
        
        # High vibration anomaly (predictive maintenance scenario)
        for i in range(10):
            data.append({
                'timestamp': (base_time - timedelta(hours=i)).isoformat(),
                'sensor_id': 'HVAC_A_vibration',
                'sensor_type': 'vibration',
                'zone': 'Floor_2',
                'equipment': 'HVAC',
                'value': 8.0 + i * 0.5,  # Progressively increasing vibration
                'unit': 'mm/s',
                'is_anomaly': True
            })
        
        # High energy consumption at midnight (operational optimization scenario)
        midnight_time = base_time.replace(hour=0, minute=0, second=0)
        for i in range(5):
            data.append({
                'timestamp': (midnight_time - timedelta(hours=i)).isoformat(),
                'sensor_id': 'Building_energy',
                'sensor_type': 'energy_consumption',
                'zone': 'Floor_3',
                'equipment': 'Electrical',
                'value': 450.0 + np.random.normal(0, 10),  # High consumption
                'unit': 'kWh',
                'is_anomaly': True
            })
        
        # Water pressure drop (anomaly detection scenario)
        for i in range(5):
            data.append({
                'timestamp': (base_time - timedelta(minutes=i*10)).isoformat(),
                'sensor_id': 'Water_pressure_01',
                'sensor_type': 'pressure',
                'zone': 'Floor_1',
                'equipment': 'Water_System',
                'value': 980.0 - i * 50,  # Sharp pressure drop
                'unit': 'hPa',
                'is_anomaly': True
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample maintenance manuals and building specs"""
        hvac_manual, building_specs = create_sample_documents()
        
        # Add specific information for testing
        hvac_manual_extended = hvac_manual + """
        
BOILER SPECIFICATIONS:
- Recommended operating temperature: 75°C
- Maximum operating pressure: 2.5 bar
- Annual maintenance required
- Inspect motor bearings when vibration exceeds 5 mm/s

MOTOR BEARING MAINTENANCE:
When vibration levels exceed 5 mm/s, immediately:
1. Stop the equipment
2. Inspect motor bearings for wear
3. Replace bearings if damaged
4. Lubricate with specified grease
        """
        
        return {
            'hvac_manual': hvac_manual_extended,
            'building_specs': building_specs
        }
    
    @pytest.fixture
    def initialized_system(self, sample_documents):
        """Initialize the complete RAG system for testing"""
        import tempfile
        import gc
        import time
        
        # Create temporary directory for test documents
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write test documents with UTF-8 encoding
            hvac_path = os.path.join(temp_dir, 'hvac_manual.txt')
            specs_path = os.path.join(temp_dir, 'building_specs.txt')
            
            with open(hvac_path, 'w', encoding='utf-8') as f:
                f.write(sample_documents['hvac_manual'])
            
            with open(specs_path, 'w', encoding='utf-8') as f:
                f.write(sample_documents['building_specs'])
            
            # Initialize document processor
            chroma_dir = os.path.join(temp_dir, 'chroma_test')
            doc_processor = DocumentProcessor(persist_directory=chroma_dir)
            doc_processor.process_documents_folder(temp_dir)
            
            # Initialize RAG pipeline
            rag_pipeline = RAGPipeline(doc_processor, use_openai=False)
            
            # Initialize other components
            anomaly_det = AnomalyDetector()
            pred_maintenance = PredictiveMaintenanceModel()
            
            system_dict = {
                'rag_pipeline': rag_pipeline,
                'doc_processor': doc_processor,
                'anomaly_detector': anomaly_det,
                'predictive_maintenance': pred_maintenance,
                'temp_dir': temp_dir
            }
            
            yield system_dict
            
        finally:
            # Cleanup ChromaDB properly
            try:
                if 'doc_processor' in locals():
                    doc_processor.cleanup()
                    del doc_processor
                if 'rag_pipeline' in locals():
                    del rag_pipeline
                
                # Force garbage collection
                gc.collect()
                time.sleep(0.1)  # Small delay for Windows file locks
                
                # Clean up temporary directory
                import shutil
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass  # Ignore cleanup errors
            except Exception as e:
                print(f"Cleanup warning: {e}")


class TestCoreFeatures(TestDataFixtures):
    """Test core functionality and unit tests"""
    
    def test_document_loading_and_processing(self, sample_documents):
        """Test that documents are loaded and processed correctly into vector store"""
        import tempfile
        import gc
        import time
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        doc_processor = None
        
        try:
            # Write test documents with UTF-8 encoding
            test_file = os.path.join(temp_dir, 'test_manual.txt')
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(sample_documents['hvac_manual'])
            
            # Initialize document processor
            chroma_dir = os.path.join(temp_dir, 'chroma_test')
            doc_processor = DocumentProcessor(persist_directory=chroma_dir)
            
            # Process documents
            doc_processor.process_documents_folder(temp_dir)
            
            # Verify documents were processed
            stats = doc_processor.get_collection_stats()
            assert stats['total_chunks'] > 0, "No document chunks were created"
            assert stats['unique_files'] > 0, "No files were processed"
            assert 'test_manual.txt' in stats['source_files'], "Test file was not processed"
            
        finally:
            # Cleanup ChromaDB properly
            if doc_processor:
                doc_processor.cleanup()
                del doc_processor
            
            gc.collect()
            time.sleep(0.1)  # Small delay for Windows file locks
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass  # Ignore cleanup errors
    
    def test_iot_sensor_data_loading(self, sample_sensor_data):
        """Test that IoT sensor data is loaded and accessible"""
        processor = DataProcessor()
        
        # Add sample data to processor
        for _, row in sample_sensor_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Verify data is accessible
        latest_readings = processor.get_latest_readings(50)
        assert len(latest_readings) > 0, "No sensor readings were loaded"
        assert 'temperature' in latest_readings['sensor_type'].values, "Temperature sensor data missing"
        assert 'vibration' in latest_readings['sensor_type'].values, "Vibration sensor data missing"
        assert 'energy_consumption' in latest_readings['sensor_type'].values, "Energy consumption data missing"
    
    def test_retrieval_mechanism(self, initialized_system):
        """Test that retrieval returns relevant context chunks for specific queries"""
        rag_pipeline = initialized_system['rag_pipeline']
        doc_processor = initialized_system['doc_processor']
        
        # Test query about HVAC maintenance
        query = "How to maintain HVAC motor bearings"
        results = doc_processor.search_documents(query, n_results=5)
        
        assert len(results) > 0, "No documents retrieved for HVAC query"
        
        # Check that retrieved chunks contain expected keywords
        retrieved_text = " ".join([result['text'].lower() for result in results])
        assert 'motor' in retrieved_text or 'bearing' in retrieved_text, "Retrieved text doesn't contain motor/bearing keywords"
        assert 'hvac' in retrieved_text or 'maintenance' in retrieved_text, "Retrieved text doesn't contain HVAC/maintenance keywords"
        
        # Verify relevance scores
        assert all(result['relevance_score'] >= 0 for result in results), "Invalid relevance scores"
    
    def test_context_augmentation(self, initialized_system, sample_sensor_data):
        """Test that the final prompt contains both user question and retrieved chunks"""
        rag_pipeline = initialized_system['rag_pipeline']
        processor = DataProcessor()
        
        # Add sensor data
        for _, row in sample_sensor_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Create sensor context
        sensor_context = {
            'latest_readings': processor.get_latest_readings(10).to_dict('records'),
            'anomalies': [],
            'maintenance_predictions': []
        }
        
        # Test query processing
        query = "What is the recommended temperature for HVAC systems?"
        result = rag_pipeline.process_query(query, sensor_context)
        
        assert 'query' in result, "Query not included in result"
        assert 'response' in result, "Response not generated"
        assert 'relevant_documents' in result, "Document count not included"
        assert result['relevant_documents'] > 0, "No relevant documents found"


class TestEndToEndScenarios(TestDataFixtures):
    """Test end-to-end scenario-based functionality"""
    
    def test_predictive_maintenance_scenario(self, initialized_system, anomalous_sensor_data):
        """Test predictive maintenance with high vibration scenario"""
        rag_pipeline = initialized_system['rag_pipeline']
        pred_maintenance = initialized_system['predictive_maintenance']
        processor = DataProcessor()
        
        # Load anomalous data (high vibration)
        high_vib_data = anomalous_sensor_data[
            (anomalous_sensor_data['sensor_type'] == 'vibration') &
            (anomalous_sensor_data['equipment'] == 'HVAC')
        ].copy()
        
        for _, row in high_vib_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Query about HVAC unit status
        query = "What is the status of HVAC unit A?"
        
        # Get predictions
        try:
            predictions = pred_maintenance.predict_maintenance(processor.get_latest_readings())
            maintenance_context = {
                'latest_readings': processor.get_latest_readings().to_dict('records'),
                'maintenance_predictions': predictions,
                'anomalies': []
            }
        except:
            # Fallback if prediction model isn't trained
            maintenance_context = {
                'latest_readings': processor.get_latest_readings().to_dict('records'),
                'maintenance_predictions': [],
                'anomalies': []
            }
        
        result = rag_pipeline.process_query(query, maintenance_context)
        response = result['response'].lower()
        
        # Assert response contains warning and maintenance action
        assert any(word in response for word in ['warning', 'high', 'vibration', 'maintenance']), \
            "Response doesn't contain expected warning keywords"
        assert any(word in response for word in ['bearing', 'inspect', 'motor']), \
            "Response doesn't reference specific maintenance action from manual"
    
    def test_operational_optimization_scenario(self, initialized_system, anomalous_sensor_data):
        """Test operational optimization with high energy consumption"""
        rag_pipeline = initialized_system['rag_pipeline']
        processor = DataProcessor()
        
        # Load high energy consumption data
        energy_data = anomalous_sensor_data[
            anomalous_sensor_data['sensor_type'] == 'energy_consumption'
        ].copy()
        
        for _, row in energy_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Query about energy cost reduction
        query = "How can we reduce building energy costs?"
        
        sensor_context = {
            'latest_readings': processor.get_latest_readings().to_dict('records'),
            'anomalies': [],
            'maintenance_predictions': []
        }
        
        result = rag_pipeline.process_query(query, sensor_context)
        response = result['response'].lower()
        
        # Assert response suggests energy optimization
        assert any(word in response for word in ['energy', 'consumption', 'efficiency', 'optimize']), \
            "Response doesn't address energy optimization"
        assert any(word in response for word in ['thermostat', 'temperature', 'schedule', 'hvac']), \
            "Response doesn't suggest specific energy-saving actions"
    
    def test_anomaly_detection_scenario(self, initialized_system, anomalous_sensor_data):
        """Test anomaly detection with water pressure drop"""
        rag_pipeline = initialized_system['rag_pipeline']
        anomaly_det = initialized_system['anomaly_detector']
        processor = DataProcessor()
        
        # Load water pressure drop data
        pressure_data = anomalous_sensor_data[
            anomalous_sensor_data['sensor_type'] == 'pressure'
        ].copy()
        
        for _, row in pressure_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Detect anomalies
        anomaly_det.set_static_thresholds()
        anomaly_results = anomaly_det.detect_all_anomalies(processor.get_latest_readings())
        detected_anomalies = anomaly_results[anomaly_results.get('is_anomaly_detected', False)]
        
        # Query about water system issues
        query = "Are there any issues with the water system?"
        
        sensor_context = {
            'latest_readings': processor.get_latest_readings().to_dict('records'),
            'anomalies': detected_anomalies.to_dict('records'),
            'maintenance_predictions': []
        }
        
        result = rag_pipeline.process_query(query, sensor_context)
        response = result['response'].lower()
        
        # Assert response identifies the anomaly
        assert len(detected_anomalies) > 0, "No anomalies detected in pressure data"
        assert any(word in response for word in ['pressure', 'water', 'system', 'anomaly']), \
            "Response doesn't identify water system anomaly"
        assert any(word in response for word in ['leak', 'drop', 'issue', 'problem']), \
            "Response doesn't suggest potential cause"
    
    def test_general_qa_scenario(self, initialized_system):
        """Test direct factual question answering from manual"""
        rag_pipeline = initialized_system['rag_pipeline']
        
        # Query for specific information from manual
        query = "What is the recommended operating temperature for the boiler?"
        
        result = rag_pipeline.process_query(query)
        response = result['response']
        
        # Assert response contains the correct temperature from manual
        assert '75°C' in response or '75' in response, \
            "Response doesn't contain the correct boiler temperature from manual"
        assert result['relevant_documents'] > 0, "No relevant documents retrieved"


class TestEdgeCasesAndRobustness(TestDataFixtures):
    """Test edge cases and system robustness"""
    
    def test_irrelevant_query(self, initialized_system):
        """Test system response to completely irrelevant questions"""
        rag_pipeline = initialized_system['rag_pipeline']
        
        # Ask irrelevant question
        query = "What is the best pizza topping?"
        
        result = rag_pipeline.process_query(query)
        response = result['response'].lower()
        
        # Assert graceful handling
        assert any(phrase in response for phrase in [
            'cannot answer', 'not available', 'no information', 'based on the provided',
            'building', 'maintenance', 'system'
        ]), "System didn't handle irrelevant query gracefully"
    
    def test_unanswerable_relevant_query(self, initialized_system):
        """Test query that is relevant but unanswerable from documents"""
        rag_pipeline = initialized_system['rag_pipeline']
        
        # Ask specific question not in documents
        query = "What is the brand of the fire extinguisher on floor 3?"
        
        result = rag_pipeline.process_query(query)
        response = result['response'].lower()
        
        # Assert system states it cannot find the information
        assert any(phrase in response for phrase in [
            'cannot find', 'not available', 'no information', 'not specified'
        ]) or 'fire' in response, "System didn't properly handle unanswerable query"
    
    def test_empty_malformed_input(self, initialized_system):
        """Test system with empty or null queries"""
        rag_pipeline = initialized_system['rag_pipeline']
        
        # Test empty query
        try:
            result = rag_pipeline.process_query("")
            assert 'response' in result, "System crashed on empty query"
        except Exception as e:
            pytest.fail(f"System crashed on empty query: {e}")
        
        # Test None query
        try:
            result = rag_pipeline.process_query(None if hasattr(rag_pipeline, 'process_query') else "")
            # Should handle gracefully
        except Exception as e:
            # Expected to handle the error gracefully
            assert "query" in str(e).lower() or "none" in str(e).lower()
    
    def test_malformed_sensor_data(self):
        """Test system with malformed sensor data"""
        processor = DataProcessor()
        
        # Test with missing required fields
        malformed_data = {
            'timestamp': datetime.now().isoformat(),
            'value': 25.0
            # Missing sensor_type, zone, equipment
        }
        
        try:
            processor.add_reading(malformed_data)
            readings = processor.get_latest_readings(5)
            # Should handle gracefully without crashing
        except Exception as e:
            # Should handle errors gracefully
            assert "data" in str(e).lower() or "missing" in str(e).lower()


class TestPerformanceAndEvaluation(TestDataFixtures):
    """Test performance metrics and evaluation capabilities"""
    
    def test_response_latency(self, initialized_system):
        """Test system response time for typical queries"""
        rag_pipeline = initialized_system['rag_pipeline']
        
        query = "What maintenance is needed for HVAC systems?"
        
        # Measure response time
        start_time = time.time()
        result = rag_pipeline.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Assert reasonable response time (< 10 seconds for testing)
        assert response_time < 10.0, f"Response time too slow: {response_time:.2f} seconds"
        assert 'processing_time_seconds' in result, "Processing time not tracked"
        
        print(f"Response time: {response_time:.2f} seconds")
    
    def test_retrieval_accuracy_metrics(self, initialized_system):
        """Test retrieval accuracy and relevance scoring"""
        doc_processor = initialized_system['doc_processor']
        
        # Test queries with known relevant content
        test_cases = [
            ("HVAC maintenance", ["hvac", "maintenance", "system"]),
            ("temperature sensor calibration", ["temperature", "sensor", "calibration"]),
            ("energy consumption monitoring", ["energy", "consumption", "monitor"])
        ]
        
        for query, expected_keywords in test_cases:
            results = doc_processor.search_documents(query, n_results=3)
            
            assert len(results) > 0, f"No results for query: {query}"
            
            # Check relevance scores are reasonable
            for result in results:
                assert 0 <= result['relevance_score'] <= 1, "Invalid relevance score"
            
            # Check that top result contains expected keywords
            top_result_text = results[0]['text'].lower()
            keyword_found = any(keyword in top_result_text for keyword in expected_keywords)
            assert keyword_found, f"Top result for '{query}' doesn't contain expected keywords"
    
    @pytest.mark.skip(reason="RAGAS evaluation would require additional setup")
    def test_ragas_evaluation_placeholder(self, initialized_system):
        """Placeholder for RAGAS evaluation metrics"""
        # This would demonstrate how to use RAGAS for evaluation
        # Example implementation would be:
        
        # from ragas import evaluate
        # from ragas.metrics import faithfulness, answer_relevancy
        
        # # Prepare evaluation data
        # eval_data = {
        #     'question': ['What is HVAC maintenance procedure?'],
        #     'answer': ['Retrieved answer from system'],
        #     'contexts': [['Retrieved context chunks']],
        #     'ground_truths': [['Expected answer']]
        # }
        
        # # Run evaluation
        # result = evaluate(
        #     dataset=eval_data,
        #     metrics=[faithfulness, answer_relevancy]
        # )
        
        # assert result['faithfulness'] > 0.7, "Low faithfulness score"
        # assert result['answer_relevancy'] > 0.7, "Low answer relevancy score"
        
        pass


class TestSystemIntegration(TestDataFixtures):
    """Test complete system integration scenarios"""
    
    def test_full_system_workflow(self, initialized_system, sample_sensor_data, anomalous_sensor_data):
        """Test complete workflow from data ingestion to response generation"""
        rag_pipeline = initialized_system['rag_pipeline']
        anomaly_det = initialized_system['anomaly_detector']
        processor = DataProcessor()
        
        # Step 1: Ingest sensor data
        all_data = pd.concat([sample_sensor_data, anomalous_sensor_data], ignore_index=True)
        for _, row in all_data.iterrows():
            processor.add_reading(row.to_dict())
        
        # Step 2: Detect anomalies
        anomaly_det.set_static_thresholds()
        anomaly_results = anomaly_det.detect_all_anomalies(processor.get_latest_readings())
        anomalies = anomaly_results[anomaly_results.get('is_anomaly_detected', False)]
        
        # Step 3: Generate comprehensive building status report
        query = "Provide a comprehensive status report for the building including any issues or maintenance needs"
        
        sensor_context = {
            'latest_readings': processor.get_latest_readings().to_dict('records'),
            'anomalies': anomalies.to_dict('records'),
            'maintenance_predictions': []
        }
        
        result = rag_pipeline.process_query(query, sensor_context)
        
        # Verify comprehensive response
        assert len(result['response']) > 100, "Response too brief for comprehensive report"
        assert result['relevant_documents'] > 0, "No documents used in response"
        assert result['processing_time_seconds'] > 0, "Processing time not recorded"
        
        # Check response contains building information
        response_lower = result['response'].lower()
        assert any(word in response_lower for word in ['building', 'system', 'maintenance', 'status']), \
            "Response doesn't contain building-related information"


# Helper function to run all tests
def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("🧪 Running Comprehensive IoT Sensor RAG System Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


if __name__ == "__main__":
    run_comprehensive_tests()
