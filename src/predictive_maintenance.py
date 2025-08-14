import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import joblib
import os

class PredictiveMaintenanceModel:
    """Predictive maintenance model for building equipment"""
    
    def __init__(self):
        self.failure_prediction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.remaining_life_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.is_trained = False
        self.feature_columns = []
        self.equipment_types = []
        
        # Maintenance schedules and thresholds
        self.maintenance_schedules = {
            'HVAC': {
                'filter_change': 30,  # days
                'coil_cleaning': 90,
                'belt_inspection': 60,
                'full_service': 365
            },
            'Electrical': {
                'connection_check': 90,
                'breaker_test': 180,
                'panel_cleaning': 365,
                'grounding_check': 365
            },
            'Security': {
                'camera_cleaning': 30,
                'sensor_calibration': 90,
                'battery_replacement': 365,
                'system_update': 180
            },
            'Lighting': {
                'bulb_replacement': 180,
                'fixture_cleaning': 90,
                'ballast_check': 365,
                'sensor_calibration': 180
            },
            'Fire_Safety': {
                'detector_test': 30,
                'battery_check': 180,
                'system_inspection': 365,
                'alarm_test': 90
            }
        }
    
    def generate_synthetic_maintenance_data(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic maintenance data for training"""
        if sensor_df.empty:
            return pd.DataFrame()
        
        maintenance_records = []
        
        # Get unique equipment combinations
        equipment_combos = sensor_df[['zone', 'equipment']].drop_duplicates()
        
        for _, combo in equipment_combos.iterrows():
            zone = combo['zone']
            equipment = combo['equipment']
            
            # Get sensor data for this equipment
            equipment_data = sensor_df[
                (sensor_df['zone'] == zone) & (sensor_df['equipment'] == equipment)
            ].copy()
            
            if equipment_data.empty:
                continue
            
            # Generate maintenance events over time
            start_date = datetime.now() - timedelta(days=365)
            
            for days_ago in range(0, 365, 30):  # Monthly intervals
                event_date = start_date + timedelta(days=days_ago)
                
                # Get sensor readings around this date
                date_range_data = equipment_data[
                    pd.to_datetime(equipment_data['timestamp']).dt.date == event_date.date()
                ]
                
                if date_range_data.empty:
                    # Use average values if no data for specific date
                    avg_values = equipment_data.groupby('sensor_type')['value'].mean()
                else:
                    avg_values = date_range_data.groupby('sensor_type')['value'].mean()
                
                # Simulate equipment health score based on sensor values
                health_score = self.calculate_health_score(avg_values, equipment)
                
                # Predict failure probability based on health score and age
                equipment_age_days = 365 - days_ago
                failure_probability = self.simulate_failure_probability(health_score, equipment_age_days)
                
                # Determine if maintenance is needed
                needs_maintenance = failure_probability > 0.7 or health_score < 0.3
                
                # Calculate remaining useful life
                if failure_probability > 0.9:
                    remaining_life = np.random.uniform(1, 15)  # Critical
                elif failure_probability > 0.7:
                    remaining_life = np.random.uniform(15, 45)  # Warning
                else:
                    remaining_life = np.random.uniform(45, 180)  # Normal
                
                record = {
                    'timestamp': event_date.isoformat(),
                    'zone': zone,
                    'equipment': equipment,
                    'health_score': health_score,
                    'failure_probability': failure_probability,
                    'needs_maintenance': needs_maintenance,
                    'remaining_life_days': remaining_life,
                    'equipment_age_days': equipment_age_days
                }
                
                # Add sensor values as features
                for sensor_type, value in avg_values.items():
                    record[f'avg_{sensor_type}'] = value
                
                maintenance_records.append(record)
        
        return pd.DataFrame(maintenance_records)
    
    def calculate_health_score(self, sensor_values: pd.Series, equipment_type: str) -> float:
        """Calculate equipment health score based on sensor values"""
        if sensor_values.empty:
            return 0.5
        
        # Equipment-specific health calculations
        if equipment_type == 'HVAC':
            temp_score = 1.0
            if 'temperature' in sensor_values:
                temp = sensor_values['temperature']
                temp_score = 1.0 - abs(temp - 22) / 10  # Optimal temp around 22°C
            
            humidity_score = 1.0
            if 'humidity' in sensor_values:
                humidity = sensor_values['humidity']
                humidity_score = 1.0 - abs(humidity - 50) / 50  # Optimal humidity around 50%
            
            energy_score = 1.0
            if 'energy_consumption' in sensor_values:
                energy = sensor_values['energy_consumption']
                energy_score = max(0, 1.0 - (energy - 100) / 400)  # Higher energy = lower health
            
            vibration_score = 1.0
            if 'vibration' in sensor_values:
                vibration = sensor_values['vibration']
                vibration_score = max(0, 1.0 - vibration / 5)  # Higher vibration = lower health
            
            health_score = np.mean([temp_score, humidity_score, energy_score, vibration_score])
        
        elif equipment_type == 'Electrical':
            energy_score = 1.0
            if 'energy_consumption' in sensor_values:
                energy = sensor_values['energy_consumption']
                energy_score = max(0, 1.0 - abs(energy - 200) / 300)
            
            # Electrical equipment health primarily based on energy patterns
            health_score = energy_score
        
        else:
            # Default health calculation for other equipment types
            health_score = np.random.uniform(0.3, 0.9)
        
        return max(0.0, min(1.0, health_score))
    
    def simulate_failure_probability(self, health_score: float, age_days: int) -> float:
        """Simulate failure probability based on health and age"""
        # Base probability from health score
        health_factor = 1.0 - health_score
        
        # Age factor (equipment degrades over time)
        age_factor = min(1.0, age_days / 1825)  # 5 years max
        
        # Combine factors
        failure_prob = 0.1 + 0.7 * health_factor + 0.2 * age_factor
        
        # Add some randomness
        failure_prob += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, failure_prob))
    
    def prepare_features(self, maintenance_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features for model training"""
        if maintenance_df.empty:
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        # Select feature columns (exclude target variables)
        feature_cols = [col for col in maintenance_df.columns 
                       if col not in ['timestamp', 'needs_maintenance', 'remaining_life_days']]
        
        X = maintenance_df[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Target variables
        y_classification = maintenance_df['needs_maintenance']
        y_regression = maintenance_df['remaining_life_days']
        
        self.feature_columns = X.columns.tolist()
        
        return X, y_classification, y_regression
    
    def train_models(self, sensor_df: pd.DataFrame) -> Dict[str, Any]:
        """Train predictive maintenance models"""
        # Generate synthetic maintenance data
        maintenance_df = self.generate_synthetic_maintenance_data(sensor_df)
        
        if maintenance_df.empty:
            return {'error': 'No data available for training'}
        
        # Prepare features
        X, y_classification, y_regression = self.prepare_features(maintenance_df)
        
        if X.empty:
            return {'error': 'No features available for training'}
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classification model (failure prediction)
        self.failure_prediction_model.fit(X_train_scaled, y_class_train)
        class_predictions = self.failure_prediction_model.predict(X_test_scaled)
        class_accuracy = accuracy_score(y_class_test, class_predictions)
        
        # Train regression model (remaining life prediction)
        self.remaining_life_model.fit(X_train_scaled, y_reg_train)
        reg_predictions = self.remaining_life_model.predict(X_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, reg_predictions)
        
        self.is_trained = True
        
        return {
            'status': 'success',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': self.feature_columns,
            'classification_accuracy': round(class_accuracy, 3),
            'regression_mse': round(reg_mse, 3),
            'classification_report': classification_report(y_class_test, class_predictions, output_dict=True)
        }
    
    def predict_maintenance(self, current_sensor_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict maintenance needs for current sensor data"""
        if not self.is_trained or current_sensor_data.empty:
            return []
        
        predictions = []
        
        # Group by equipment
        equipment_groups = current_sensor_data.groupby(['zone', 'equipment'])
        
        for (zone, equipment), group in equipment_groups:
            # Calculate average sensor values
            avg_values = group.groupby('sensor_type')['value'].mean()
            
            # Calculate health score
            health_score = self.calculate_health_score(avg_values, equipment)
            
            # Prepare features for prediction
            feature_dict = {
                'zone': zone,
                'equipment': equipment,
                'health_score': health_score,
                'equipment_age_days': 365,  # Assume 1 year old
            }
            
            # Add sensor averages
            for sensor_type, value in avg_values.items():
                feature_dict[f'avg_{sensor_type}'] = value
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([feature_dict])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_columns]
            
            # Encode categorical variables
            categorical_cols = feature_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in feature_df.columns:
                    feature_df[col] = pd.Categorical(feature_df[col]).codes
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make predictions
            failure_prob = self.failure_prediction_model.predict_proba(features_scaled)[0][1]
            remaining_life = self.remaining_life_model.predict(features_scaled)[0]
            
            # Determine maintenance urgency
            if failure_prob > 0.8:
                urgency = 'critical'
            elif failure_prob > 0.6:
                urgency = 'high'
            elif failure_prob > 0.4:
                urgency = 'medium'
            else:
                urgency = 'low'
            
            prediction = {
                'zone': zone,
                'equipment': equipment,
                'health_score': round(health_score, 3),
                'failure_probability': round(failure_prob, 3),
                'remaining_life_days': max(1, round(remaining_life, 1)),
                'urgency': urgency,
                'recommended_actions': self.get_maintenance_recommendations(equipment, urgency),
                'next_maintenance_date': (datetime.now() + timedelta(days=remaining_life)).strftime('%Y-%m-%d')
            }
            
            predictions.append(prediction)
        
        return sorted(predictions, key=lambda x: x['failure_probability'], reverse=True)
    
    def get_maintenance_recommendations(self, equipment_type: str, urgency: str) -> List[str]:
        """Get maintenance recommendations based on equipment type and urgency"""
        base_actions = {
            'HVAC': [
                'Check and replace air filters',
                'Inspect and clean coils',
                'Check belt tension and alignment',
                'Verify thermostat calibration',
                'Test safety controls'
            ],
            'Electrical': [
                'Inspect electrical connections',
                'Test circuit breakers',
                'Check grounding systems',
                'Clean electrical panels',
                'Verify load balancing'
            ],
            'Security': [
                'Clean camera lenses',
                'Test motion sensors',
                'Check battery levels',
                'Update firmware',
                'Verify network connectivity'
            ],
            'Lighting': [
                'Replace failed bulbs',
                'Clean light fixtures',
                'Check ballasts and drivers',
                'Test daylight sensors',
                'Verify timer settings'
            ],
            'Fire_Safety': [
                'Test smoke detectors',
                'Check alarm systems',
                'Inspect sprinkler heads',
                'Test emergency lighting',
                'Verify evacuation systems'
            ]
        }
        
        actions = base_actions.get(equipment_type, ['Perform general inspection'])
        
        if urgency == 'critical':
            actions = ['URGENT: ' + action for action in actions[:3]]
            actions.append('Schedule immediate technician visit')
        elif urgency == 'high':
            actions = actions[:4]
            actions.append('Schedule maintenance within 1 week')
        elif urgency == 'medium':
            actions = actions[:3]
            actions.append('Schedule maintenance within 1 month')
        else:
            actions = actions[:2]
            actions.append('Include in next routine maintenance')
        
        return actions
    
    def get_maintenance_schedule(self, equipment_type: str) -> Dict[str, Any]:
        """Get maintenance schedule for equipment type"""
        if equipment_type not in self.maintenance_schedules:
            return {}
        
        schedule = self.maintenance_schedules[equipment_type]
        next_maintenance = {}
        
        for task, interval_days in schedule.items():
            next_date = datetime.now() + timedelta(days=interval_days)
            next_maintenance[task] = {
                'interval_days': interval_days,
                'next_due_date': next_date.strftime('%Y-%m-%d'),
                'days_until_due': interval_days
            }
        
        return next_maintenance

# Global instance
predictive_maintenance = PredictiveMaintenanceModel()
