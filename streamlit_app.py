import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.data_ingestion import data_generator, data_processor, simulate_real_time_data, load_sample_data
from src.document_processor import DocumentProcessor, create_sample_documents
from src.anomaly_detection import anomaly_detector, alert_system
from src.predictive_maintenance import predictive_maintenance
from src.rag_pipeline import RAGPipeline
from src.utils import (
    calculate_basic_stats, format_sensor_reading, filter_data_by_timerange, 
    get_sensor_status, create_sample_documents_files, load_environment_variables
)

# Page configuration
st.set_page_config(
    page_title="IoT Sensor Data RAG for Smart Buildings",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_environment_variables()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.doc_processor = None
    st.session_state.rag_pipeline = None
    st.session_state.last_update = datetime.now()

def initialize_system():
    """Initialize the RAG system"""
    with st.spinner("Initializing IoT RAG system..."):
        # Load sample data
        historical_df = load_sample_data()
        
        # Create sample documents
        create_sample_documents_files()
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Process sample documents
        hvac_manual, building_specs = create_sample_documents()
        
        # Create temporary files and process them
        os.makedirs("temp_docs", exist_ok=True)
        
        with open("temp_docs/hvac_manual.txt", "w") as f:
            f.write(hvac_manual)
        with open("temp_docs/building_specs.txt", "w") as f:
            f.write(building_specs)
        
        doc_processor.process_documents_folder("temp_docs")
        doc_processor.process_documents_folder("data/documents")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(doc_processor, use_openai=bool(os.getenv('OPENAI_API_KEY')))
        
        # Train anomaly detection model
        if not historical_df.empty:
            anomaly_detector.train_isolation_forest(historical_df)
        
        # Train predictive maintenance model
        if not historical_df.empty:
            predictive_maintenance.train_models(historical_df)
        
        st.session_state.doc_processor = doc_processor
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.initialized = True
        
        st.success("System initialized successfully!")

def main():
    """Main application"""
    st.title("🏢 IoT Sensor Data RAG for Smart Buildings")
    st.markdown("### Predictive Maintenance & Operational Optimization System")
    
    # Initialize system if not done
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
        else:
            st.info("Click 'Initialize System' to start the IoT RAG system.")
            return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            simulate_real_time_data()
            st.rerun()
        
        # System stats
        st.header("📊 System Stats")
        doc_stats = st.session_state.doc_processor.get_collection_stats()
        st.metric("Documents Processed", doc_stats.get('total_chunks', 0))
        st.metric("Unique Files", doc_stats.get('unique_files', 0))
        
        latest_data = data_processor.get_latest_readings(10)
        st.metric("Latest Readings", len(latest_data))
        
        anomalies = data_processor.get_anomaly_readings()
        st.metric("Active Anomalies", len(anomalies))
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Update data
        if (datetime.now() - st.session_state.last_update).seconds >= refresh_interval:
            simulate_real_time_data()
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Show countdown
        time_since_update = (datetime.now() - st.session_state.last_update).seconds
        time_to_refresh = refresh_interval - time_since_update
        if time_to_refresh > 0:
            st.sidebar.info(f"Next refresh in: {time_to_refresh}s")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Real-time Dashboard", 
        "🔍 RAG Query Interface", 
        "⚠️ Anomaly Detection", 
        "🔧 Predictive Maintenance", 
        "📚 Document Search",
        "📈 Analytics"
    ])
    
    with tab1:
        show_realtime_dashboard()
    
    with tab2:
        show_rag_interface()
    
    with tab3:
        show_anomaly_detection()
    
    with tab4:
        show_predictive_maintenance()
    
    with tab5:
        show_document_search()
    
    with tab6:
        show_analytics()

def show_realtime_dashboard():
    """Show real-time sensor dashboard"""
    st.header("📊 Real-time Sensor Dashboard")
    
    # Get latest data
    latest_data = data_processor.get_latest_readings(100)
    
    if latest_data.empty:
        st.warning("No sensor data available. Click 'Refresh Data' to generate sample data.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sensors = len(latest_data)
        st.metric("Active Sensors", total_sensors)
    
    with col2:
        zones = latest_data['zone'].nunique()
        st.metric("Monitoring Zones", zones)
    
    with col3:
        anomaly_count = latest_data.get('is_anomaly', pd.Series(False)).sum()
        st.metric("Current Anomalies", anomaly_count, delta=None)
    
    with col4:
        equipment_types = latest_data['equipment'].nunique()
        st.metric("Equipment Types", equipment_types)
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Readings by Zone")
        temp_data = latest_data[latest_data['sensor_type'] == 'temperature']
        if not temp_data.empty:
            fig = px.box(temp_data, x='zone', y='value', title="Temperature Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No temperature data available")
    
    with col2:
        st.subheader("Energy Consumption by Equipment")
        energy_data = latest_data[latest_data['sensor_type'] == 'energy_consumption']
        if not energy_data.empty:
            fig = px.bar(energy_data, x='equipment', y='value', color='zone',
                        title="Energy Consumption by Equipment Type")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No energy data available")
    
    # Recent readings table
    st.subheader("Recent Sensor Readings")
    display_data = latest_data.head(20).copy()
    if not display_data.empty:
        display_data['status'] = display_data.apply(
            lambda row: get_sensor_status(row['value'], row['sensor_type']), axis=1
        )
        st.dataframe(
            display_data[['timestamp', 'zone', 'equipment', 'sensor_type', 'value', 'unit', 'status']],
            use_container_width=True
        )

def show_rag_interface():
    """Show RAG query interface"""
    st.header("🔍 RAG Query Interface")
    st.markdown("Ask questions about building maintenance, operations, and current sensor status.")
    
    # Query examples
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - "What maintenance is needed for HVAC system based on current readings?"
        - "How can I optimize energy consumption in Floor_2?"
        - "What should I do about high CO2 levels in the building?"
        - "Explain the troubleshooting steps for temperature control issues"
        - "What are the recommended maintenance intervals for electrical systems?"
        """)
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What maintenance is needed for HVAC system based on current sensor readings?",
        height=100
    )
    
    if st.button("🔍 Ask Question", type="primary") and user_query:
        with st.spinner("Processing your query..."):
            # Get current sensor context
            latest_readings = data_processor.get_latest_readings(50).to_dict('records')
            anomalies = data_processor.get_anomaly_readings().to_dict('records')
            
            # Get maintenance predictions
            maintenance_predictions = []
            if not data_processor.get_latest_readings().empty:
                try:
                    maintenance_predictions = predictive_maintenance.predict_maintenance(
                        data_processor.get_latest_readings()
                    )
                except:
                    pass
            
            sensor_context = {
                'latest_readings': latest_readings,
                'anomalies': anomalies,
                'maintenance_predictions': maintenance_predictions
            }
            
            # Process query
            result = st.session_state.rag_pipeline.process_query(user_query, sensor_context)
            
            # Display results
            st.markdown("### 💬 Response")
            st.markdown(result['response'])
            
            # Show metadata
            with st.expander("📊 Query Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents Used", result['relevant_documents'])
                with col2:
                    st.metric("Processing Time", f"{result['processing_time_seconds']}s")
                with col3:
                    st.metric("Sources", len(result['documents_used']))
                
                if result['documents_used']:
                    st.subheader("Sources Used:")
                    for doc in result['documents_used']:
                        st.write(f"- {doc['source']} (Relevance: {doc['relevance_score']:.3f})")

def show_anomaly_detection():
    """Show anomaly detection interface"""
    st.header("⚠️ Anomaly Detection")
    
    # Get latest data
    latest_data = data_processor.get_latest_readings(200)
    
    if latest_data.empty:
        st.warning("No data available for anomaly detection.")
        return
    
    # Detect anomalies
    anomaly_data = anomaly_detector.detect_all_anomalies(latest_data)
    anomaly_summary = anomaly_detector.get_anomaly_summary(anomaly_data)
    
    # Process alerts
    alert_system.process_anomaly_alerts(anomaly_data)
    alert_summary = alert_system.get_alert_summary()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Readings", anomaly_summary['total_readings'])
    
    with col2:
        st.metric("Anomalies Detected", anomaly_summary['total_anomalies'])
    
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate']:.1f}%")
    
    with col4:
        st.metric("Active Alerts", alert_summary['total_active_alerts'])
    
    # Anomaly visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anomalies by Sensor Type")
        if anomaly_summary['anomalies_by_sensor_type']:
            fig = px.pie(
                values=list(anomaly_summary['anomalies_by_sensor_type'].values()),
                names=list(anomaly_summary['anomalies_by_sensor_type'].keys()),
                title="Anomaly Distribution by Sensor Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies by sensor type to display")
    
    with col2:
        st.subheader("Anomalies by Zone")
        if anomaly_summary['anomalies_by_zone']:
            fig = px.bar(
                x=list(anomaly_summary['anomalies_by_zone'].keys()),
                y=list(anomaly_summary['anomalies_by_zone'].values()),
                title="Anomalies by Building Zone"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies by zone to display")
    
    # Active alerts
    st.subheader("🚨 Active Alerts")
    active_alerts = alert_system.get_active_alerts(20)
    
    if active_alerts:
        for alert in active_alerts:
            severity_color = {
                'critical': '🔴',
                'warning': '🟡',
                'info': '🔵'
            }.get(alert['severity'], '⚪')
            
            with st.expander(f"{severity_color} {alert['message'][:100]}..."):
                st.write(f"**Type:** {alert['type']}")
                st.write(f"**Severity:** {alert['severity']}")
                st.write(f"**Time:** {alert['timestamp']}")
                st.write(f"**Message:** {alert['message']}")
                
                if alert['sensor_data']:
                    st.json(alert['sensor_data'])
                
                if st.button(f"Acknowledge Alert {alert['id']}", key=f"ack_{alert['id']}"):
                    alert_system.acknowledge_alert(alert['id'])
                    st.success("Alert acknowledged!")
                    st.rerun()
    else:
        st.success("No active alerts. All systems operating normally.")
    
    # Recent anomalous readings
    st.subheader("Recent Anomalous Readings")
    current_anomalies = anomaly_data[anomaly_data.get('is_anomaly_detected', False)]
    
    if not current_anomalies.empty:
        st.dataframe(
            current_anomalies[['timestamp', 'zone', 'equipment', 'sensor_type', 'value', 'unit']].head(10),
            use_container_width=True
        )
    else:
        st.info("No anomalous readings in current dataset.")

def show_predictive_maintenance():
    """Show predictive maintenance interface"""
    st.header("🔧 Predictive Maintenance")
    
    # Get latest data
    latest_data = data_processor.get_latest_readings(100)
    
    if latest_data.empty:
        st.warning("No data available for predictive maintenance.")
        return
    
    # Get maintenance predictions
    try:
        predictions = predictive_maintenance.predict_maintenance(latest_data)
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        predictions = []
    
    if not predictions:
        st.warning("No maintenance predictions available. System may need more training data.")
        return
    
    # Summary metrics
    high_priority = len([p for p in predictions if p['urgency'] in ['critical', 'high']])
    avg_health = np.mean([p['health_score'] for p in predictions])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Equipment Monitored", len(predictions))
    
    with col2:
        st.metric("High Priority Items", high_priority)
    
    with col3:
        st.metric("Average Health Score", f"{avg_health:.2f}")
    
    with col4:
        avg_failure_prob = np.mean([p['failure_probability'] for p in predictions])
        st.metric("Avg Failure Probability", f"{avg_failure_prob:.1%}")
    
    # Priority maintenance items
    st.subheader("🚨 Priority Maintenance Items")
    
    urgent_items = [p for p in predictions if p['urgency'] in ['critical', 'high']]
    
    if urgent_items:
        for item in urgent_items[:5]:
            urgency_color = {'critical': '🔴', 'high': '🟡'}.get(item['urgency'], '🟢')
            
            with st.expander(f"{urgency_color} {item['zone']} - {item['equipment']} (Urgency: {item['urgency']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Health Score", f"{item['health_score']:.3f}")
                    st.metric("Failure Probability", f"{item['failure_probability']:.1%}")
                
                with col2:
                    st.metric("Remaining Life", f"{item['remaining_life_days']:.1f} days")
                    st.metric("Next Maintenance", item['next_maintenance_date'])
                
                st.subheader("Recommended Actions:")
                for action in item['recommended_actions']:
                    st.write(f"• {action}")
    else:
        st.success("No urgent maintenance items. All equipment operating within normal parameters.")
    
    # All equipment health overview
    st.subheader("📊 Equipment Health Overview")
    
    # Create health score chart
    equipment_names = [f"{p['zone']} - {p['equipment']}" for p in predictions]
    health_scores = [p['health_score'] for p in predictions]
    failure_probs = [p['failure_probability'] for p in predictions]
    urgencies = [p['urgency'] for p in predictions]
    
    fig = px.scatter(
        x=health_scores,
        y=failure_probs,
        color=urgencies,
        hover_name=equipment_names,
        title="Equipment Health vs Failure Probability",
        labels={'x': 'Health Score', 'y': 'Failure Probability'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed maintenance schedule
    st.subheader("📅 Detailed Maintenance Schedule")
    
    maintenance_df = pd.DataFrame(predictions)
    maintenance_df = maintenance_df.sort_values('failure_probability', ascending=False)
    
    st.dataframe(
        maintenance_df[['zone', 'equipment', 'health_score', 'failure_probability', 
                       'remaining_life_days', 'urgency', 'next_maintenance_date']],
        use_container_width=True
    )

def show_document_search():
    """Show document search interface"""
    st.header("📚 Document Search")
    
    # Search interface
    search_query = st.text_input(
        "Search building documents:",
        placeholder="e.g., HVAC maintenance, filter replacement, temperature control"
    )
    
    if search_query:
        with st.spinner("Searching documents..."):
            results = st.session_state.doc_processor.search_documents(search_query, n_results=10)
        
        if results:
            st.subheader(f"Found {len(results)} relevant documents")
            
            for i, result in enumerate(results):
                relevance_score = result['relevance_score']
                color = "🟢" if relevance_score > 0.7 else "🟡" if relevance_score > 0.5 else "🔴"
                
                with st.expander(f"{color} Result {i+1} - Relevance: {relevance_score:.3f}"):
                    st.write(f"**Source:** {result['metadata'].get('source_file', 'Unknown')}")
                    st.write(f"**Document Type:** {result['metadata'].get('doc_type', 'Unknown')}")
                    st.write(f"**Section:** {result['metadata'].get('section_id', 'Unknown')}")
                    st.write("**Content:**")
                    st.write(result['text'])
        else:
            st.info("No documents found for your search query.")
    
    # Document collection statistics
    st.subheader("📊 Document Collection Statistics")
    doc_stats = st.session_state.doc_processor.get_collection_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Document Chunks", doc_stats.get('total_chunks', 0))
    
    with col2:
        st.metric("Unique Source Files", doc_stats.get('unique_files', 0))
    
    with col3:
        if doc_stats.get('document_types'):
            most_common_type = max(doc_stats['document_types'].items(), key=lambda x: x[1])
            st.metric("Most Common Type", f"{most_common_type[0]} ({most_common_type[1]})")
    
    # Document types breakdown
    if doc_stats.get('document_types'):
        st.subheader("Document Types")
        fig = px.pie(
            values=list(doc_stats['document_types'].values()),
            names=list(doc_stats['document_types'].keys()),
            title="Document Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Source files list
    if doc_stats.get('source_files'):
        st.subheader("Source Files")
        for file_name in doc_stats['source_files']:
            st.write(f"📄 {file_name}")

def show_analytics():
    """Show analytics and metrics"""
    st.header("📈 System Analytics")
    
    # Get data for analysis
    latest_data = data_processor.get_latest_readings(500)
    
    if latest_data.empty:
        st.warning("No data available for analytics.")
        return
    
    # Time range selector
    time_range = st.selectbox(
        "Select time range:",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "All Data"]
    )
    
    if time_range == "Last Hour":
        filtered_data = filter_data_by_timerange(latest_data, 1)
    elif time_range == "Last 6 Hours":
        filtered_data = filter_data_by_timerange(latest_data, 6)
    elif time_range == "Last 24 Hours":
        filtered_data = filter_data_by_timerange(latest_data, 24)
    else:
        filtered_data = latest_data
    
    if filtered_data.empty:
        st.warning(f"No data available for {time_range.lower()}.")
        return
    
    # Performance metrics
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # System uptime (simulated)
        uptime_percentage = 99.2  # Simulated uptime
        st.metric("System Uptime", f"{uptime_percentage:.1f}%")
    
    with col2:
        # Data quality score
        valid_readings = len(filtered_data[filtered_data['value'].notna()])
        total_readings = len(filtered_data)
        data_quality = (valid_readings / total_readings) * 100 if total_readings > 0 else 0
        st.metric("Data Quality", f"{data_quality:.1f}%")
    
    with col3:
        # Response time (simulated)
        avg_response_time = 0.85  # Simulated response time in seconds
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    # Sensor trends
    st.subheader("📊 Sensor Trends")
    
    # Temperature trend
    temp_data = filtered_data[filtered_data['sensor_type'] == 'temperature']
    if not temp_data.empty:
        fig = px.line(
            temp_data, 
            x='timestamp', 
            y='value', 
            color='zone',
            title="Temperature Trends by Zone"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Energy consumption analysis
    energy_data = filtered_data[filtered_data['sensor_type'] == 'energy_consumption']
    if not energy_data.empty:
        st.subheader("⚡ Energy Consumption Analysis")
        
        # Energy by zone
        energy_by_zone = energy_data.groupby('zone')['value'].sum().reset_index()
        fig = px.bar(
            energy_by_zone,
            x='zone',
            y='value',
            title="Total Energy Consumption by Zone"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Sensor Correlations")
    correlations = data_processor.detect_correlations()
    
    if correlations:
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Sensor Pair', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        fig = px.bar(
            corr_df.head(10),
            x='Correlation',
            y='Sensor Pair',
            orientation='h',
            title="Top 10 Sensor Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No correlation data available.")
    
    # Raw data export
    st.subheader("💾 Data Export")
    if st.button("Download Current Dataset"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
