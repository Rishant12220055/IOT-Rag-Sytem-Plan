import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime, timedelta

class AnomalyDetector:
    """Anomaly detection for IoT sensor data"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.anomaly_thresholds = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        if df.empty:
            return df
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create pivot table with sensor types as columns
        feature_df = df.pivot_table(
            index=['timestamp', 'zone', 'equipment'],
            columns='sensor_type',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Fill missing values with median
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Add time-based features
        if 'timestamp' in feature_df.columns:
            feature_df['hour'] = feature_df['timestamp'].dt.hour
            feature_df['day_of_week'] = feature_df['timestamp'].dt.dayofweek
            feature_df['is_weekend'] = feature_df['day_of_week'].isin([5, 6]).astype(int)
            feature_df['is_business_hours'] = ((feature_df['hour'] >= 8) & (feature_df['hour'] <= 18)).astype(int)
        
        return feature_df
    
    def set_static_thresholds(self):
        """Set static thresholds for rule-based anomaly detection"""
        self.anomaly_thresholds = {
            'temperature': {'min': 15.0, 'max': 32.0},
            'humidity': {'min': 20.0, 'max': 80.0},
            'co2': {'min': 250.0, 'max': 1500.0},
            'energy_consumption': {'min': 10.0, 'max': 600.0},
            'pressure': {'min': 1000.0, 'max': 1040.0},
            'vibration': {'min': 0.0, 'max': 8.0},
            'noise_level': {'min': 25.0, 'max': 90.0},
            'air_flow': {'min': 300.0, 'max': 2500.0}
        }
    
    def detect_threshold_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using static thresholds"""
        if df.empty:
            return df
        
        df = df.copy()
        df['threshold_anomaly'] = False
        
        for _, row in df.iterrows():
            sensor_type = row.get('sensor_type', '')
            value = row.get('value', 0)
            
            if sensor_type in self.anomaly_thresholds:
                thresholds = self.anomaly_thresholds[sensor_type]
                if value < thresholds['min'] or value > thresholds['max']:
                    df.loc[df.index == row.name, 'threshold_anomaly'] = True
        
        return df
    
    def train_isolation_forest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train isolation forest model for anomaly detection"""
        feature_df = self.prepare_features(df)
        
        if feature_df.empty:
            return {'error': 'No data available for training'}
        
        # Select only numeric columns for training
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_columns if col not in ['timestamp']]
        
        if len(self.feature_columns) == 0:
            return {'error': 'No numeric features available for training'}
        
        # Prepare training data
        X = feature_df[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        
        # Get anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        
        # Calculate statistics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        return {
            'status': 'success',
            'training_samples': len(X),
            'features_used': self.feature_columns,
            'anomalies_detected': int(n_anomalies),
            'anomaly_rate': round(anomaly_rate, 2),
            'mean_anomaly_score': round(np.mean(anomaly_scores), 3),
            'std_anomaly_score': round(np.std(anomaly_scores), 3)
        }
    
    def detect_ml_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using trained ML model"""
        if not self.is_trained or df.empty:
            df['ml_anomaly'] = False
            df['anomaly_score'] = 0.0
            return df
        
        feature_df = self.prepare_features(df)
        
        if feature_df.empty:
            df['ml_anomaly'] = False
            df['anomaly_score'] = 0.0
            return df
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_columns) - set(feature_df.columns)
        for feature in missing_features:
            feature_df[feature] = 0
        
        X = feature_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Map back to original dataframe
        df = df.copy()
        df['ml_anomaly'] = False
        df['anomaly_score'] = 0.0
        
        # This is a simplified mapping - in practice, you'd need to properly align indices
        for i, (_, row) in enumerate(df.iterrows()):
            if i < len(predictions):
                df.loc[df.index == row.name, 'ml_anomaly'] = predictions[i] == -1
                df.loc[df.index == row.name, 'anomaly_score'] = anomaly_scores[i]
        
        return df
    
    def detect_all_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using all available methods"""
        if df.empty:
            return df
        
        # Set thresholds if not already set
        if not self.anomaly_thresholds:
            self.set_static_thresholds()
        
        # Apply threshold-based detection
        df = self.detect_threshold_anomalies(df)
        
        # Apply ML-based detection if model is trained
        df = self.detect_ml_anomalies(df)
        
        # Combine anomaly indicators
        df['is_anomaly_detected'] = df.get('threshold_anomaly', False) | df.get('ml_anomaly', False)
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        if df.empty:
            return {'total_readings': 0, 'anomalies': 0, 'anomaly_rate': 0}
        
        total_readings = len(df)
        threshold_anomalies = df.get('threshold_anomaly', pd.Series(False)).sum()
        ml_anomalies = df.get('ml_anomaly', pd.Series(False)).sum()
        total_anomalies = df.get('is_anomaly_detected', pd.Series(False)).sum()
        
        anomaly_by_sensor = {}
        if 'sensor_type' in df.columns and 'is_anomaly_detected' in df.columns:
            anomaly_by_sensor = df[df['is_anomaly_detected']]['sensor_type'].value_counts().to_dict()
        
        anomaly_by_zone = {}
        if 'zone' in df.columns and 'is_anomaly_detected' in df.columns:
            anomaly_by_zone = df[df['is_anomaly_detected']]['zone'].value_counts().to_dict()
        
        return {
            'total_readings': total_readings,
            'threshold_anomalies': int(threshold_anomalies),
            'ml_anomalies': int(ml_anomalies),
            'total_anomalies': int(total_anomalies),
            'anomaly_rate': round(total_anomalies / total_readings * 100, 2) if total_readings > 0 else 0,
            'anomalies_by_sensor_type': anomaly_by_sensor,
            'anomalies_by_zone': anomaly_by_zone
        }

class AlertSystem:
    """Alert system for anomalies and maintenance needs"""
    
    def __init__(self):
        self.alerts = []
        self.alert_history = []
        self.max_alerts = 100
        
    def create_alert(self, alert_type: str, severity: str, message: str, 
                    sensor_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new alert"""
        alert = {
            'id': len(self.alert_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,  # 'anomaly', 'maintenance', 'system'
            'severity': severity,  # 'critical', 'warning', 'info'
            'message': message,
            'sensor_data': sensor_data or {},
            'status': 'active',
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        return alert
    
    def process_anomaly_alerts(self, anomaly_df: pd.DataFrame):
        """Process anomalies and create alerts"""
        if anomaly_df.empty:
            return
        
        # Get current anomalies
        current_anomalies = anomaly_df[anomaly_df.get('is_anomaly_detected', False)]
        
        for _, row in current_anomalies.iterrows():
            sensor_type = row.get('sensor_type', 'unknown')
            zone = row.get('zone', 'unknown')
            equipment = row.get('equipment', 'unknown')
            value = row.get('value', 0)
            
            # Determine severity based on sensor type and value
            severity = self.determine_severity(sensor_type, value)
            
            message = f"Anomaly detected in {sensor_type} sensor at {zone} - {equipment}. Value: {value}"
            
            self.create_alert(
                alert_type='anomaly',
                severity=severity,
                message=message,
                sensor_data={
                    'sensor_type': sensor_type,
                    'zone': zone,
                    'equipment': equipment,
                    'value': value,
                    'sensor_id': row.get('sensor_id', '')
                }
            )
    
    def determine_severity(self, sensor_type: str, value: float) -> str:
        """Determine alert severity based on sensor type and value"""
        critical_thresholds = {
            'temperature': {'min': 10.0, 'max': 35.0},
            'humidity': {'min': 15.0, 'max': 85.0},
            'co2': {'min': 200.0, 'max': 2000.0},
            'energy_consumption': {'min': 5.0, 'max': 700.0}
        }
        
        if sensor_type in critical_thresholds:
            thresholds = critical_thresholds[sensor_type]
            if value < thresholds['min'] or value > thresholds['max']:
                return 'critical'
        
        return 'warning'
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active alerts"""
        active_alerts = [alert for alert in self.alerts if alert['status'] == 'active']
        return sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert['severity'] == severity and alert['status'] == 'active']
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['status'] = 'acknowledged'
                return True
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        active_alerts = [alert for alert in self.alerts if alert['status'] == 'active']
        
        severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
        type_counts = {'anomaly': 0, 'maintenance': 0, 'system': 0}
        
        for alert in active_alerts:
            severity_counts[alert['severity']] = severity_counts.get(alert['severity'], 0) + 1
            type_counts[alert['type']] = type_counts.get(alert['type'], 0) + 1
        
        return {
            'total_active_alerts': len(active_alerts),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'latest_alert_time': active_alerts[0]['timestamp'] if active_alerts else None
        }

# Global instances
anomaly_detector = AnomalyDetector()
alert_system = AlertSystem()
