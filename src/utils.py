import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics for a dataframe"""
    if df.empty:
        return {}
    
    return {
        'count': len(df),
        'mean': df.select_dtypes(include=[np.number]).mean().to_dict(),
        'std': df.select_dtypes(include=[np.number]).std().to_dict(),
        'min': df.select_dtypes(include=[np.number]).min().to_dict(),
        'max': df.select_dtypes(include=[np.number]).max().to_dict()
    }

def format_sensor_reading(reading: Dict[str, Any]) -> str:
    """Format a sensor reading for display"""
    return f"{reading.get('sensor_type', 'Unknown')}: {reading.get('value', 'N/A')} {reading.get('unit', '')}"

def filter_data_by_timerange(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Filter dataframe to last N hours"""
    if df.empty or 'timestamp' not in df.columns:
        return df
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
    
    return df[df['timestamp'] >= cutoff_time]

def get_sensor_status(value: float, sensor_type: str) -> str:
    """Get status for a sensor reading"""
    thresholds = {
        'temperature': {'min': 18, 'max': 26},
        'humidity': {'min': 30, 'max': 70},
        'co2': {'min': 400, 'max': 1000},
        'energy_consumption': {'min': 50, 'max': 400}
    }
    
    if sensor_type in thresholds:
        t = thresholds[sensor_type]
        if value < t['min']:
            return '🔵 Low'
        elif value > t['max']:
            return '🔴 High'
        else:
            return '🟢 Normal'
    
    return '⚪ Unknown'

def create_sample_documents_files():
    """Create sample document files"""
    import os
    
    # Create data/documents directory
    docs_dir = "data/documents"
    os.makedirs(docs_dir, exist_ok=True)
    
    # HVAC Manual content
    hvac_content = """HVAC SYSTEM MAINTENANCE MANUAL

OVERVIEW
This manual covers maintenance procedures for heating, ventilation, and air conditioning systems in smart buildings.

MONTHLY MAINTENANCE TASKS
1. Replace air filters if dirty or clogged
2. Check thermostat calibration and settings
3. Inspect belts for wear and proper tension
4. Clean condenser and evaporator coils
5. Check refrigerant levels and look for leaks
6. Verify proper operation of dampers and controls

QUARTERLY MAINTENANCE TASKS
1. Lubricate all motor bearings and moving parts
2. Inspect and tighten electrical connections
3. Test all safety controls and alarm systems
4. Check ductwork for leaks and obstructions
5. Calibrate temperature and humidity sensors
6. Verify proper ventilation rates and air quality

ANNUAL MAINTENANCE TASKS
1. Complete comprehensive system performance analysis
2. Replace worn belts, bearings, and other components
3. Clean entire ductwork system and air handling units
4. Test emergency shutdown and safety procedures
5. Update system documentation and maintenance records

TROUBLESHOOTING GUIDE
High Energy Consumption:
- Check for dirty or clogged filters
- Verify proper building insulation
- Inspect ductwork for leaks
- Check thermostat settings and schedules
- Analyze compressor performance and efficiency

Poor Indoor Air Quality:
- Replace air filters immediately
- Check ventilation rates and outside air intake
- Inspect for mold, contamination, or odors
- Verify CO2 sensor calibration
- Clean air handling units and ductwork

Temperature Control Issues:
- Calibrate temperature sensors and thermostats
- Check damper operation and positioning
- Verify control system settings and schedules
- Inspect heating and cooling coils
- Test zone control valves and actuators"""
    
    # Building Specifications content
    building_specs_content = """SMART BUILDING SPECIFICATIONS

BUILDING OVERVIEW
This document outlines the technical specifications for a 5-story smart commercial building with integrated IoT sensors and automated building management systems.

SENSOR SPECIFICATIONS

Temperature Sensors:
- Operating range: -40°C to +85°C
- Accuracy: ±0.5°C at 25°C
- Response time: Less than 30 seconds
- Installation density: One sensor per 500 square feet
- Calibration frequency: Every 6 months

Humidity Sensors:
- Operating range: 0-100% RH
- Accuracy: ±2% RH (10-90% RH)
- Calibration frequency: Annually
- Alert thresholds: Less than 30% or greater than 70%

Air Quality Sensors:
- CO2 measurement range: 0-5000 ppm
- Accuracy: ±30 ppm + 3% of reading
- Alert threshold: Greater than 1000 ppm
- Ventilation trigger: Greater than 800 ppm

Energy Monitoring:
- Smart meters on each floor and major equipment
- Real-time consumption monitoring capability
- Demand response and peak load management
- Power quality monitoring and analysis

SYSTEM REQUIREMENTS

Lighting Control System:
- LED fixtures with integrated daylight sensors
- Occupancy-based automatic dimming
- Target efficiency: Minimum 95 lumens per watt
- Maintenance trigger: Replace when output drops to 80%

HVAC Control System:
- Variable air volume with economizer controls
- Demand-controlled ventilation based on occupancy
- Energy recovery ventilation systems
- Predictive maintenance monitoring

MAINTENANCE SCHEDULES

Sensor Calibration:
- Temperature sensors: Every 6 months
- Humidity sensors: Annually
- Air quality sensors: Every 3 months
- Energy meters: Annually by certified technician

System Updates:
- Firmware updates: Quarterly
- Security patches: Monthly
- Performance optimization: Annually
- Hardware refresh cycle: Every 5 years

ALERT AND THRESHOLD SETTINGS

Critical Alerts (Immediate Response):
- Temperature above 30°C or below 15°C
- Humidity above 80% or below 20%
- CO2 levels above 1500 ppm
- Energy consumption exceeding 120% of baseline

Warning Alerts (Response within 4 hours):
- Temperature above 28°C or below 18°C
- Humidity above 70% or below 30%
- CO2 levels above 1000 ppm
- Energy consumption exceeding 110% of baseline"""
    
    # Write files
    with open(os.path.join(docs_dir, "hvac_maintenance_manual.txt"), 'w') as f:
        f.write(hvac_content)
    
    with open(os.path.join(docs_dir, "building_specifications.txt"), 'w') as f:
        f.write(building_specs_content)
    
    return True

def load_environment_variables():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, skip loading .env file

def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback"""
    try:
        return __import__(module_name)
    except ImportError:
        print(f"Warning: {module_name} not installed, using fallback")
        return fallback
