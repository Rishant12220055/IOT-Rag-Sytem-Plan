import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Optional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class RAGPipeline:
    """RAG Pipeline for IoT sensor data and building documentation"""
    
    def __init__(self, document_processor, use_openai: bool = False):
        self.document_processor = document_processor
        self.use_openai = use_openai
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Set up OpenAI if API key is available
        if use_openai and OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.use_openai = False
    
    def enhance_query_with_context(self, query: str, sensor_data: Dict[str, Any] = None) -> str:
        """Enhance user query with current sensor context"""
        enhanced_query = query
        
        if sensor_data:
            context_parts = []
            
            # Add sensor readings context
            if 'latest_readings' in sensor_data:
                readings = sensor_data['latest_readings']
                context_parts.append(f"Current sensor readings: {readings}")
            
            # Add anomaly context
            if 'anomalies' in sensor_data:
                anomalies = sensor_data['anomalies']
                if anomalies:
                    context_parts.append(f"Current anomalies detected: {anomalies}")
            
            # Add maintenance context
            if 'maintenance_predictions' in sensor_data:
                maintenance = sensor_data['maintenance_predictions']
                if maintenance:
                    context_parts.append(f"Maintenance predictions: {maintenance}")
            
            if context_parts:
                enhanced_query = f"{query}\n\nCurrent building context:\n" + "\n".join(context_parts)
        
        return enhanced_query
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database"""
        try:
            results = self.document_processor.search_documents(query, n_results)
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_response_local(self, query: str, context_docs: List[Dict[str, Any]], 
                              sensor_context: Dict[str, Any] = None) -> str:
        """Generate response using local/rule-based approach"""
        
        # Extract relevant information from documents
        relevant_info = []
        max_relevance = 0
        for doc in context_docs:
            if doc['relevance_score'] > 0.3:  # Lower threshold to get more docs
                relevant_info.append(doc['text'][:800])  # Include more text
                max_relevance = max(max_relevance, doc['relevance_score'])
        
        # Check if we have sufficiently relevant documents
        if max_relevance < 0.4 or not relevant_info:
            query_lower = query.lower()
            return f"""## Query Response

I don't have specific information about "{query}" in the available building documentation. 

### Available Information:
The current documentation covers:
- HVAC system maintenance and troubleshooting
- Building specifications and sensor information
- Energy management guidelines
- General building operations

### Recommendations:
- Please rephrase your question to focus on HVAC, energy, sensors, or building operations
- Check if you need additional documentation for this topic
- Contact your building management team for specialized guidance

If you have questions about maintenance, energy efficiency, or building systems, I'd be happy to help with those topics."""
        
        # Build response based on query type
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['maintenance', 'repair', 'fix', 'service', 'bearing', 'inspect', 'motor', 'vibration', 'status', 'unit']):
            response = self.generate_maintenance_response(query, relevant_info, sensor_context)
        elif any(keyword in query_lower for keyword in ['energy', 'efficiency', 'consumption', 'optimize']):
            response = self.generate_energy_response(query, relevant_info, sensor_context)
        elif any(keyword in query_lower for keyword in ['anomaly', 'problem', 'issue', 'alert']):
            response = self.generate_anomaly_response(query, relevant_info, sensor_context)
        elif any(keyword in query_lower for keyword in ['temperature', 'humidity', 'hvac', 'air']):
            response = self.generate_hvac_response(query, relevant_info, sensor_context)
        else:
            response = self.generate_general_response(query, relevant_info, sensor_context)
        
        return response
    
    def generate_maintenance_response(self, query: str, docs: List[str], 
                                    sensor_context: Dict[str, Any] = None) -> str:
        """Generate maintenance-focused response"""
        response = "## Maintenance Recommendations\n\n"
        
        # Add current maintenance predictions if available
        if sensor_context and 'maintenance_predictions' in sensor_context:
            predictions = sensor_context['maintenance_predictions']
            if predictions:
                response += "### Current Maintenance Needs:\n"
                for pred in predictions[:3]:  # Top 3 priorities
                    response += f"- **{pred['zone']} - {pred['equipment']}**: "
                    response += f"{pred['urgency']} priority (Failure probability: {pred['failure_probability']:.1%})\n"
                    for action in pred['recommended_actions'][:2]:
                        response += f"  - {action}\n"
                response += "\n"
        
        # Add relevant documentation
        if docs:
            response += "### Relevant Maintenance Guidelines:\n"
            for i, doc in enumerate(docs[:2]):
                # Include more content to capture specific maintenance terms
                doc_content = doc[:800] if len(doc) > 800 else doc
                response += f"**Guideline {i+1}:**\n{doc_content}\n\n"
        
        # Add general recommendations
        response += "### General Recommendations:\n"
        response += "- Follow manufacturer's maintenance schedules\n"
        response += "- Monitor sensor readings for early warning signs\n"
        response += "- Keep detailed maintenance logs\n"
        response += "- Schedule regular professional inspections\n"
        
        return response
    
    def generate_energy_response(self, query: str, docs: List[str], 
                               sensor_context: Dict[str, Any] = None) -> str:
        """Generate energy efficiency response"""
        response = "## Energy Optimization Insights\n\n"
        
        # Add current energy analysis if available
        if sensor_context and 'latest_readings' in sensor_context:
            response += "### Current Energy Status:\n"
            energy_readings = [r for r in sensor_context['latest_readings'] 
                             if r.get('sensor_type') == 'energy_consumption']
            if energy_readings:
                avg_consumption = sum(r['value'] for r in energy_readings) / len(energy_readings)
                response += f"- Average current consumption: {avg_consumption:.1f} kWh\n"
                response += f"- Total monitoring points: {len(energy_readings)}\n\n"
        
        # Add optimization recommendations
        response += "### Optimization Strategies:\n"
        response += "- **HVAC Optimization**: Maintain temperature setpoints between 20-24°C\n"
        response += "- **Lighting Control**: Use daylight sensors and occupancy detection\n"
        response += "- **Equipment Scheduling**: Run non-critical systems during off-peak hours\n"
        response += "- **Insulation**: Check for air leaks and improve building envelope\n\n"
        
        # Add relevant documentation
        if docs:
            response += "### Relevant Guidelines:\n"
            for doc in docs[:1]:
                response += f"{doc[:400]}...\n\n"
        
        return response
    
    def generate_anomaly_response(self, query: str, docs: List[str], 
                                sensor_context: Dict[str, Any] = None) -> str:
        """Generate anomaly analysis response"""
        response = "## Anomaly Analysis\n\n"
        
        # Add current anomalies if available
        if sensor_context and 'anomalies' in sensor_context:
            anomalies = sensor_context['anomalies']
            if anomalies:
                response += "### Current Anomalies Detected:\n"
                for anomaly in anomalies[:5]:
                    response += f"- **{anomaly.get('sensor_type', 'Unknown')}** in "
                    response += f"{anomaly.get('zone', 'Unknown')} - {anomaly.get('equipment', 'Unknown')}: "
                    response += f"Value {anomaly.get('value', 'N/A')} {anomaly.get('unit', '')}\n"
                response += "\n"
            else:
                response += "### No Active Anomalies\nAll systems are operating within normal parameters.\n\n"
        
        # Add troubleshooting steps
        response += "### Recommended Actions:\n"
        response += "1. **Verify Sensor Calibration**: Check if sensors need recalibration\n"
        response += "2. **Check Physical Conditions**: Inspect equipment for visible issues\n"
        response += "3. **Review Recent Changes**: Consider recent maintenance or environmental changes\n"
        response += "4. **Monitor Trends**: Watch for pattern development over time\n\n"
        
        # Add relevant documentation
        if docs:
            response += "### Troubleshooting Guidelines:\n"
            for doc in docs[:1]:
                response += f"{doc[:400]}...\n\n"
        
        return response
    
    def generate_hvac_response(self, query: str, docs: List[str], 
                             sensor_context: Dict[str, Any] = None) -> str:
        """Generate HVAC-specific response"""
        response = "## HVAC System Analysis\n\n"
        
        # Add current HVAC readings if available
        if sensor_context and 'latest_readings' in sensor_context:
            hvac_readings = [r for r in sensor_context['latest_readings'] 
                           if r.get('equipment') == 'HVAC']
            if hvac_readings:
                response += "### Current HVAC Status:\n"
                temp_readings = [r for r in hvac_readings if r.get('sensor_type') == 'temperature']
                humidity_readings = [r for r in hvac_readings if r.get('sensor_type') == 'humidity']
                
                if temp_readings:
                    avg_temp = sum(r['value'] for r in temp_readings) / len(temp_readings)
                    response += f"- Average temperature: {avg_temp:.1f}°C\n"
                
                if humidity_readings:
                    avg_humidity = sum(r['value'] for r in humidity_readings) / len(humidity_readings)
                    response += f"- Average humidity: {avg_humidity:.1f}%\n"
                
                response += f"- Monitoring zones: {len(set(r['zone'] for r in hvac_readings))}\n\n"
        
        # Add HVAC recommendations
        response += "### HVAC Optimization:\n"
        response += "- **Temperature Control**: Maintain 22±2°C for optimal comfort and efficiency\n"
        response += "- **Humidity Control**: Keep between 40-60% RH to prevent mold and discomfort\n"
        response += "- **Air Quality**: Monitor CO2 levels, maintain below 1000 ppm\n"
        response += "- **Filter Maintenance**: Replace filters monthly or when differential pressure is high\n\n"
        
        # Add relevant documentation
        if docs:
            response += "### HVAC Guidelines:\n"
            for doc in docs[:1]:
                response += f"{doc[:400]}...\n\n"
        
        return response
    
    def generate_general_response(self, query: str, docs: List[str], 
                                sensor_context: Dict[str, Any] = None) -> str:
        """Generate general response"""
        response = "## Building Management Insights\n\n"
        
        # Add current system overview
        if sensor_context:
            response += "### Current System Status:\n"
            if 'latest_readings' in sensor_context:
                readings = sensor_context['latest_readings']
                zones = set(r.get('zone', 'Unknown') for r in readings)
                equipment_types = set(r.get('equipment', 'Unknown') for r in readings)
                response += f"- Monitoring {len(zones)} zones\n"
                response += f"- {len(equipment_types)} equipment types\n"
                response += f"- {len(readings)} active sensors\n\n"
        
        # Add relevant documentation - include full content to capture specific terms
        if docs:
            response += "### Relevant Information:\n"
            for i, doc in enumerate(docs[:2]):
                # Include more content to capture specific maintenance terms
                doc_content = doc[:800] if len(doc) > 800 else doc
                response += f"**Source {i+1}:**\n{doc_content}\n\n"
        
        # Add general recommendations
        response += "### General Recommendations:\n"
        response += "- Regular monitoring of all building systems\n"
        response += "- Proactive maintenance scheduling\n"
        response += "- Energy efficiency optimization\n"
        response += "- Staff training on building management systems\n"
        
        return response
    
    def generate_response_openai(self, query: str, context_docs: List[Dict[str, Any]], 
                               sensor_context: Dict[str, Any] = None) -> str:
        """Generate response using OpenAI API"""
        if not self.use_openai or not OPENAI_AVAILABLE:
            return self.generate_response_local(query, context_docs, sensor_context)
        
        try:
            # Prepare context
            context_text = "\n".join([doc['text'] for doc in context_docs[:3]])
            
            # Prepare sensor context
            sensor_text = ""
            if sensor_context:
                sensor_text = f"Current building status: {str(sensor_context)[:500]}"
            
            # Create prompt
            prompt = f"""You are an expert building management system assistant. Answer the user's question using the provided context from maintenance manuals and current sensor data.

Context from manuals:
{context_text}

{sensor_text}

User question: {query}

Provide a helpful, actionable response that combines the documentation with current sensor readings. Format your response with clear headings and bullet points."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful building management assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.generate_response_local(query, context_docs, sensor_context)
    
    def process_query(self, query: str, sensor_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a complete RAG query"""
        start_time = datetime.now()
        
        # Enhance query with sensor context
        enhanced_query = self.enhance_query_with_context(query, sensor_context)
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(enhanced_query, n_results=5)
        
        # Generate response
        if self.use_openai:
            response = self.generate_response_openai(query, relevant_docs, sensor_context)
        else:
            response = self.generate_response_local(query, relevant_docs, sensor_context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'query': query,
            'response': response,
            'relevant_documents': len(relevant_docs),
            'processing_time_seconds': round(processing_time, 2),
            'documents_used': [
                {
                    'source': doc['metadata'].get('source_file', 'Unknown'),
                    'relevance_score': round(doc['relevance_score'], 3)
                } for doc in relevant_docs[:3]
            ],
            'timestamp': datetime.now().isoformat()
        }
