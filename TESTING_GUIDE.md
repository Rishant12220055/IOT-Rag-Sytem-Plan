# 🧪 Comprehensive Testing Suite for IoT Sensor Data RAG System

## Overview

This document describes the comprehensive testing framework for the "IoT Sensor Data RAG for Smart Buildings" system. The test suite validates all critical components and scenarios using pytest framework.

## 📁 Test Files

- **`test_smart_building_rag.py`** - Main comprehensive test suite
- **`validate_tests.py`** - Basic validation runner
- **`test_system.py`** - Original system component tests

## 🎯 Test Categories

### 1. Core Functionality & Unit Tests

#### Document Processing Tests
- **`test_document_loading_and_processing`**: Validates that PDFs, DOCX, and text files are properly loaded and processed into the ChromaDB vector store
- **`test_retrieval_mechanism`**: Ensures that document retrieval returns relevant context chunks with proper keywords and relevance scores
- **`test_context_augmentation`**: Verifies that the final prompt contains both user questions and retrieved document chunks

#### Data Ingestion Tests
- **`test_iot_sensor_data_loading`**: Validates that IoT sensor data (temperature, vibration, energy, pressure) is properly loaded and accessible
- Tests CSV loading, data structure validation, and multi-sensor data handling

### 2. End-to-End Scenario Tests

#### Predictive Maintenance Scenario
```python
def test_predictive_maintenance_scenario():
    # Setup: Simulate progressively increasing vibration for HVAC unit
    # Test: Query "What is the status of HVAC unit A?"
    # Assert: Response contains warning about high vibration and references 
    #         specific maintenance action from manual (inspect motor bearings)
```

#### Operational Optimization Scenario
```python
def test_operational_optimization_scenario():
    # Setup: Simulate high energy consumption during off-peak hours
    # Test: Query "How can we reduce building energy costs?"
    # Assert: Response suggests thermostat adjustments and references 
    #         time-based energy data
```

#### Anomaly Detection Scenario
```python
def test_anomaly_detection_scenario():
    # Setup: Simulate sharp water pressure drop
    # Test: Query "Are there any issues with the water system?"
    # Assert: System identifies pressure anomaly and suggests potential 
    #         causes (leak, pipe burst)
```

#### General Q&A Scenario
```python
def test_general_qa_scenario():
    # Test: Ask "What is the recommended operating temperature for the boiler?"
    # Assert: Response contains correct temperature (75°C) from manual
```

### 3. Edge Cases & Robustness Tests

#### Invalid Input Handling
- **`test_irrelevant_query`**: Tests response to completely unrelated questions ("What is the best pizza topping?")
- **`test_unanswerable_relevant_query`**: Tests handling of relevant but unanswerable questions
- **`test_empty_malformed_input`**: Tests system with empty, null, or malformed inputs
- **`test_malformed_sensor_data`**: Tests handling of incomplete or invalid sensor data

### 4. Performance & Evaluation Tests

#### Performance Metrics
- **`test_response_latency`**: Measures query processing time (target: < 10 seconds)
- **`test_retrieval_accuracy_metrics`**: Validates relevance scoring and keyword matching

#### Quality Evaluation (RAGAS Integration Ready)
- **`test_ragas_evaluation_placeholder`**: Framework for integrating RAGAS metrics
  - Faithfulness evaluation
  - Answer relevancy scoring
  - Context precision measurement

### 5. System Integration Tests

#### Complete Workflow Validation
- **`test_full_system_workflow`**: Tests entire pipeline from data ingestion to response generation
- Validates integration between all components
- Tests comprehensive building status reporting

## 🚀 Running Tests

### Quick Validation
```bash
python validate_tests.py
```

### Full Test Suite
```bash
pytest test_smart_building_rag.py -v
```

### Specific Test Categories
```bash
# Core functionality tests
pytest test_smart_building_rag.py::TestCoreFeatures -v

# End-to-end scenarios
pytest test_smart_building_rag.py::TestEndToEndScenarios -v

# Edge cases
pytest test_smart_building_rag.py::TestEdgeCasesAndRobustness -v

# Performance tests
pytest test_smart_building_rag.py::TestPerformanceAndEvaluation -v
```

### Generate HTML Report
```bash
pytest test_smart_building_rag.py --html=test_report.html --self-contained-html
```

## 📊 Test Data Fixtures

### Sample Sensor Data
- **Normal Operating Data**: Temperature (22°C ±1), vibration (1.0 ±0.2 mm/s), energy (150 ±20 kWh)
- **Anomalous Data**: High vibration (8+ mm/s), high energy consumption (450+ kWh), pressure drops

### Document Fixtures
- **HVAC Manual**: Extended with boiler specifications and bearing maintenance procedures
- **Building Specifications**: Sensor thresholds, calibration schedules, alert settings

### System Initialization
- Temporary ChromaDB instance for isolated testing
- Pre-loaded documents and vector embeddings
- Configured RAG pipeline with local models

## 🎯 Test Assertions & Validation

### Response Quality Checks
- **Keyword Presence**: Verifies relevant technical terms in responses
- **Context Utilization**: Ensures retrieved documents are referenced
- **Actionable Content**: Validates that responses contain specific maintenance actions

### Data Integrity Checks
- **Sensor Data Validation**: Confirms all required fields are present
- **Anomaly Detection**: Validates threshold-based and ML-based detection
- **Document Processing**: Ensures proper chunking and embedding generation

### Performance Benchmarks
- **Response Time**: < 10 seconds for standard queries
- **Retrieval Accuracy**: Relevance scores > 0.5 for top results
- **System Availability**: Graceful error handling without crashes

## 🔧 Test Configuration

### Environment Setup
```python
# Test fixtures automatically handle:
- Temporary file creation
- ChromaDB initialization
- Test data generation
- System component setup
```

### Mock Data Generation
- **IoT Sensors**: Realistic time-series data with patterns
- **Anomalies**: Controlled abnormal readings for testing
- **Documents**: Extended manuals with test-specific information

## 📈 Expected Test Results

### Success Criteria
- **All Core Tests**: Must pass (document processing, data ingestion, retrieval)
- **Scenario Tests**: Must demonstrate correct behavior for all use cases
- **Edge Cases**: Must handle gracefully without crashes
- **Performance**: Must meet latency requirements

### Sample Test Output
```
🧪 IoT Sensor RAG System - Comprehensive Tests
=============================================
TestCoreFeatures::test_document_loading_and_processing ✅ PASSED
TestCoreFeatures::test_iot_sensor_data_loading ✅ PASSED
TestCoreFeatures::test_retrieval_mechanism ✅ PASSED
TestEndToEndScenarios::test_predictive_maintenance_scenario ✅ PASSED
TestEndToEndScenarios::test_operational_optimization_scenario ✅ PASSED
TestEndToEndScenarios::test_anomaly_detection_scenario ✅ PASSED
TestEdgeCasesAndRobustness::test_irrelevant_query ✅ PASSED
TestPerformanceAndEvaluation::test_response_latency ✅ PASSED

📊 Results: 20/20 tests passed
🎉 System validation complete!
```

## 🛠️ Extending the Test Suite

### Adding New Test Cases
1. **Inherit from TestDataFixtures** for access to sample data
2. **Use pytest fixtures** for setup and teardown
3. **Follow naming convention**: `test_[feature]_[scenario]`
4. **Include clear assertions** with descriptive error messages

### Custom Test Data
```python
@pytest.fixture
def custom_sensor_data(self):
    # Generate specific test scenarios
    return custom_data_frame
```

### Integration with CI/CD
- Tests are designed for automated execution
- Generate reports in multiple formats (HTML, XML, JSON)
- Compatible with GitHub Actions, Jenkins, etc.

## 📚 Testing Best Practices

### Test Isolation
- Each test runs independently with fresh fixtures
- Temporary directories ensure no state pollution
- Mock external dependencies when needed

### Comprehensive Coverage
- **Happy Path**: Normal system operation
- **Error Scenarios**: Invalid inputs and edge cases
- **Performance**: Latency and throughput validation
- **Integration**: End-to-end workflow testing

### Documentation
- Clear test descriptions and expected outcomes
- Inline comments explaining complex assertions
- Fixture documentation for reusability

## 🎯 Quality Assurance Metrics

The test suite provides comprehensive validation of:
- ✅ **Functional Requirements**: All features work as specified
- ✅ **Performance Requirements**: Response times meet targets
- ✅ **Reliability Requirements**: Graceful error handling
- ✅ **Integration Requirements**: Components work together
- ✅ **Usability Requirements**: Clear and actionable responses

This testing framework ensures the IoT Sensor Data RAG system meets all requirements and performs reliably in production environments.
