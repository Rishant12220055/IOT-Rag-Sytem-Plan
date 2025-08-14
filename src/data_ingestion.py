import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

class IoTDataGenerator:
    """Generates realistic IoT sensor data for smart buildings"""
    
    def __init__(self):
        self.sensor_types = {
            'temperature': {'min': 18.0, 'max': 28.0, 'unit': '°C'},
            'humidity': {'min': 30.0, 'max': 70.0, 'unit': '%'},
            'pressure': {'min': 1010.0, 'max': 1030.0, 'unit': 'hPa'},
            'co2': {'min': 300.0, 'max': 1200.0, 'unit': 'ppm'},
            'energy_consumption': {'min': 50.0, 'max': 500.0, 'unit': 'kWh'},
            'vibration': {'min': 0.1, 'max': 5.0, 'unit': 'mm/s'},
            'noise_level': {'min': 30.0, 'max': 80.0, 'unit': 'dB'},
            'air_flow': {'min': 500.0, 'max': 2000.0, 'unit': 'CFM'}
        }
        
        self.building_zones = ['Floor_1', 'Floor_2', 'Floor_3', 'Basement', 'Rooftop']
        self.equipment_types = ['HVAC', 'Electrical', 'Security', 'Lighting', 'Fire_Safety']
        
    def generate_sensor_reading(self, sensor_type: str, zone: str, equipment: str, 
                              anomaly_prob: float = 0.05) -> Dict[str, Any]:
        """Generate a single sensor reading"""
        base_range = self.sensor_types[sensor_type]
        
        # Add some time-based patterns (day/night cycles, etc.)
        hour = datetime.now().hour
        time_factor = 1.0
        
        if sensor_type == 'temperature':
            # Higher temps during day, lower at night
            time_factor = 0.9 + 0.2 * (np.sin(hour * np.pi / 12) + 1) / 2
        elif sensor_type == 'energy_consumption':
            # Higher consumption during business hours
            time_factor = 0.7 + 0.6 * (1 if 8 <= hour <= 18 else 0.3)
        elif sensor_type == 'co2':
            # Higher CO2 during occupied hours
            time_factor = 0.8 + 0.4 * (1 if 7 <= hour <= 19 else 0.2)
            
        # Base value with time factor
        base_value = (base_range['min'] + base_range['max']) / 2 * time_factor
        noise = np.random.normal(0, (base_range['max'] - base_range['min']) * 0.1)
        value = base_value + noise
        
        # Add anomalies randomly
        is_anomaly = random.random() < anomaly_prob
        if is_anomaly:
            if random.random() > 0.5:
                value *= 1.5  # High anomaly
            else:
                value *= 0.5  # Low anomaly
                
        # Ensure value stays within realistic bounds
        value = max(base_range['min'] * 0.5, min(base_range['max'] * 1.5, value))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sensor_id': f"{zone}_{equipment}_{sensor_type}",
            'sensor_type': sensor_type,
            'zone': zone,
            'equipment': equipment,
            'value': round(value, 2),
            'unit': base_range['unit'],
            'is_anomaly': is_anomaly,
            'status': 'normal' if not is_anomaly else 'anomaly'
        }
    
    def generate_batch_readings(self, num_readings: int = 100) -> List[Dict[str, Any]]:
        """Generate a batch of sensor readings"""
        readings = []
        
        for _ in range(num_readings):
            zone = random.choice(self.building_zones)
            equipment = random.choice(self.equipment_types)
            sensor_type = random.choice(list(self.sensor_types.keys()))
            
            reading = self.generate_sensor_reading(sensor_type, zone, equipment)
            readings.append(reading)
            
        return readings
    
    def generate_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Generate historical sensor data for training models"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate readings every hour
            for hour in range(24):
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                
                for zone in self.building_zones:
                    for equipment in self.equipment_types:
                        for sensor_type in list(self.sensor_types.keys()):
                            # Simulate some sensors being offline occasionally
                            if random.random() > 0.95:
                                continue
                                
                            reading = self.generate_sensor_reading(sensor_type, zone, equipment)
                            reading['timestamp'] = timestamp.isoformat()
                            data.append(reading)
        
        return pd.DataFrame(data)

class DataProcessor:
    """Processes and analyzes IoT sensor data"""
    
    def __init__(self):
        self.data_buffer = []
        self.max_buffer_size = 1000
        
    def add_reading(self, reading: Dict[str, Any]):
        """Add a new sensor reading to the buffer"""
        self.data_buffer.append(reading)
        
        # Keep buffer size manageable
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
    
    def get_latest_readings(self, limit: int = 50) -> pd.DataFrame:
        """Get the latest sensor readings"""
        if not self.data_buffer:
            return pd.DataFrame()
        
        recent_data = self.data_buffer[-limit:]
        return pd.DataFrame(recent_data)
    
    def get_readings_by_zone(self, zone: str) -> pd.DataFrame:
        """Get readings for a specific building zone"""
        zone_data = [r for r in self.data_buffer if r.get('zone') == zone]
        return pd.DataFrame(zone_data)
    
    def get_readings_by_equipment(self, equipment: str) -> pd.DataFrame:
        """Get readings for specific equipment type"""
        equipment_data = [r for r in self.data_buffer if r.get('equipment') == equipment]
        return pd.DataFrame(equipment_data)
    
    def get_anomaly_readings(self) -> pd.DataFrame:
        """Get all anomalous readings"""
        anomaly_data = [r for r in self.data_buffer if r.get('is_anomaly', False)]
        return pd.DataFrame(anomaly_data)
    
    def calculate_zone_stats(self, zone: str) -> Dict[str, Any]:
        """Calculate statistics for a specific zone"""
        zone_df = self.get_readings_by_zone(zone)
        
        if zone_df.empty:
            return {}
        
        stats = {}
        for sensor_type in zone_df['sensor_type'].unique():
            sensor_data = zone_df[zone_df['sensor_type'] == sensor_type]['value']
            stats[sensor_type] = {
                'mean': sensor_data.mean(),
                'std': sensor_data.std(),
                'min': sensor_data.min(),
                'max': sensor_data.max(),
                'latest': sensor_data.iloc[-1] if not sensor_data.empty else None
            }
        
        return stats
    
    def detect_correlations(self) -> Dict[str, float]:
        """Detect correlations between different sensor types"""
        df = self.get_latest_readings(200)
        
        if df.empty or len(df) < 10:
            return {}
        
        # Pivot data to have sensor types as columns
        pivot_df = df.pivot_table(
            index='timestamp', 
            columns='sensor_type', 
            values='value', 
            aggfunc='mean'
        )
        
        correlations = {}
        sensor_types = pivot_df.columns.tolist()
        
        for i, sensor1 in enumerate(sensor_types):
            for sensor2 in sensor_types[i+1:]:
                try:
                    corr = pivot_df[sensor1].corr(pivot_df[sensor2])
                    if not np.isnan(corr):
                        correlations[f"{sensor1}_vs_{sensor2}"] = round(corr, 3)
                except:
                    continue
        
        return correlations

# Global instances for the application
data_generator = IoTDataGenerator()
data_processor = DataProcessor()

def simulate_real_time_data():
    """Simulate real-time data generation"""
    readings = data_generator.generate_batch_readings(5)
    for reading in readings:
        data_processor.add_reading(reading)
    return readings

def load_sample_data():
    """Load sample historical data for analysis"""
    historical_df = data_generator.generate_historical_data(days=7)
    
    # Add to processor buffer (sample)
    sample_data = historical_df.tail(100).to_dict('records')
    for record in sample_data:
        data_processor.add_reading(record)
    
    return historical_df
