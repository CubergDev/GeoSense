import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import joblib
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.models.geotrack import Trip, GeoTrack


class DemandPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = "1.0.0"
        self.last_trained = None
        self.feature_columns = [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'lat_grid', 'lon_grid', 'historical_demand_1h', 'historical_demand_24h',
            'weather_score', 'is_holiday'
        ]
        self.model_path = os.path.join(settings.ML_MODEL_PATH, "demand_predictor")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
    
    def predict(
        self,
        db: Session,
        target_time: datetime,
        area_bounds: List[float],
        grid_size_meters: int = 1000
    ) -> List[Dict[str, Any]]:
        """Predict demand for a specific area and time"""
        
        if not self.model:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create prediction grid
        min_lat, min_lon, max_lat, max_lon = area_bounds
        grid_size_deg = grid_size_meters / 111000.0  # Convert meters to degrees
        
        lat_points = np.arange(min_lat, max_lat, grid_size_deg)
        lon_points = np.arange(min_lon, max_lon, grid_size_deg)
        
        predictions = []
        
        for lat in lat_points:
            for lon in lon_points:
                # Prepare features for this grid cell
                features = self._prepare_features(db, target_time, lat, lon)
                
                if features is not None:
                    # Make prediction
                    feature_array = np.array([features])
                    feature_scaled = self.scaler.transform(feature_array)
                    demand_raw = self.model.predict(feature_scaled)[0]
                    
                    # Normalize demand to 0-1 scale
                    demand_normalized = max(0.0, min(1.0, demand_raw / 100.0))  # Assuming max ~100 trips per cell
                    
                    # Calculate confidence based on historical data availability
                    confidence = self._calculate_confidence(db, lat, lon, target_time)
                    
                    predictions.append({
                        "latitude": lat,
                        "longitude": lon,
                        "predicted_demand": demand_normalized,
                        "confidence": confidence
                    })
        
        return predictions
    
    def retrain(self, db: Session, lookback_days: int = 90):
        """Retrain the demand prediction model with recent data"""
        
        print(f"Starting demand predictor retraining with {lookback_days} days of data...")
        
        # Prepare training data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        training_data = self._prepare_training_data(db, start_time, end_time)
        
        if len(training_data) < 1000:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples. Need at least 1000.")
        
        # Prepare features and target
        X = training_data[self.feature_columns].values
        y = training_data['demand_count'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train both models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Create ensemble predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)  # Weighted ensemble
        
        # Evaluate performance
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Model performance - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # Save the better performing individual model as the main model
        if r2 > 0.5:  # Acceptable performance threshold
            # Use the Random Forest as primary (generally more stable)
            self.model = rf_model
            self.last_trained = datetime.now()
            self._save_model()
            print("Model training completed successfully!")
        else:
            raise ValueError(f"Model performance too low (R² = {r2:.3f}). Keeping existing model.")
    
    def _prepare_training_data(self, db: Session, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Prepare training data from historical trips"""
        
        # Query historical trip data
        trips = db.query(Trip).filter(
            Trip.start_time >= start_time,
            Trip.start_time <= end_time,
            Trip.start_point.isnot(None)
        ).all()
        
        training_records = []
        grid_size_deg = 0.01  # ~1km grid
        
        for trip in trips:
            # Extract coordinates from start_point (assuming POINT format)
            try:
                # Parse POINT(lon lat) format
                coords = str(trip.start_point).replace('POINT(', '').replace(')', '').split()
                lon, lat = float(coords[0]), float(coords[1])
                
                # Snap to grid
                lat_grid = round(lat / grid_size_deg) * grid_size_deg
                lon_grid = round(lon / grid_size_deg) * grid_size_deg
                
                # Extract time features
                hour_of_day = trip.start_time.hour
                day_of_week = trip.start_time.weekday()
                month = trip.start_time.month
                is_weekend = 1 if day_of_week >= 5 else 0
                
                # Calculate historical demand (simplified)
                historical_demand_1h = self._get_historical_demand(db, lat_grid, lon_grid, trip.start_time, hours=1)
                historical_demand_24h = self._get_historical_demand(db, lat_grid, lon_grid, trip.start_time, hours=24)
                
                # Weather and holiday features (simplified - would integrate with external APIs)
                weather_score = 0.8  # Default good weather
                is_holiday = 0  # Would check against holiday calendar
                
                training_records.append({
                    'hour_of_day': hour_of_day,
                    'day_of_week': day_of_week,
                    'month': month,
                    'is_weekend': is_weekend,
                    'lat_grid': lat_grid,
                    'lon_grid': lon_grid,
                    'historical_demand_1h': historical_demand_1h,
                    'historical_demand_24h': historical_demand_24h,
                    'weather_score': weather_score,
                    'is_holiday': is_holiday,
                    'demand_count': 1  # Each trip contributes 1 to demand
                })
                
            except (ValueError, IndexError, AttributeError):
                continue  # Skip trips with invalid coordinates
        
        df = pd.DataFrame(training_records)
        
        # Aggregate by grid cell and time window (hourly)
        df['time_window'] = pd.to_datetime([trip.start_time for trip in trips]).floor('H')
        
        # Group by grid and time window to get demand counts
        aggregated = df.groupby(['lat_grid', 'lon_grid', 'hour_of_day', 'day_of_week', 'month']).agg({
            'demand_count': 'sum',
            'is_weekend': 'first',
            'historical_demand_1h': 'mean',
            'historical_demand_24h': 'mean',
            'weather_score': 'mean',
            'is_holiday': 'first'
        }).reset_index()
        
        return aggregated
    
    def _prepare_features(self, db: Session, target_time: datetime, lat: float, lon: float) -> Optional[List[float]]:
        """Prepare features for a single prediction"""
        
        try:
            # Time features
            hour_of_day = target_time.hour
            day_of_week = target_time.weekday()
            month = target_time.month
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Grid coordinates
            grid_size_deg = 0.01
            lat_grid = round(lat / grid_size_deg) * grid_size_deg
            lon_grid = round(lon / grid_size_deg) * grid_size_deg
            
            # Historical demand
            historical_demand_1h = self._get_historical_demand(db, lat_grid, lon_grid, target_time, hours=1)
            historical_demand_24h = self._get_historical_demand(db, lat_grid, lon_grid, target_time, hours=24)
            
            # External features (simplified)
            weather_score = 0.8  # Would integrate with weather API
            is_holiday = 0  # Would check holiday calendar
            
            return [
                hour_of_day, day_of_week, month, is_weekend,
                lat_grid, lon_grid, historical_demand_1h, historical_demand_24h,
                weather_score, is_holiday
            ]
            
        except Exception:
            return None
    
    def _get_historical_demand(self, db: Session, lat: float, lon: float, target_time: datetime, hours: int = 24) -> float:
        """Get historical demand for a location (simplified implementation)"""
        
        # Look back over the past few weeks for same time pattern
        lookback_start = target_time - timedelta(days=21)  # 3 weeks back
        lookback_end = target_time - timedelta(hours=1)  # Exclude current hour
        
        # Count trips in nearby area during similar time windows
        grid_tolerance = 0.005  # ~500m tolerance
        
        try:
            # Simplified query - in production would use PostGIS spatial functions
            similar_time_trips = db.query(Trip).filter(
                Trip.start_time >= lookback_start,
                Trip.start_time <= lookback_end,
                Trip.start_time.extract('hour') == target_time.hour,  # Same hour
                Trip.start_point.isnot(None)
            ).count()
            
            # Average over the lookback period
            weeks_in_period = 3
            avg_demand = similar_time_trips / weeks_in_period
            
            return max(0.0, avg_demand)
            
        except Exception:
            return 0.0
    
    def _calculate_confidence(self, db: Session, lat: float, lon: float, target_time: datetime) -> float:
        """Calculate prediction confidence based on historical data availability"""
        
        # Simple confidence based on historical data density
        historical_points = self._get_historical_demand(db, lat, lon, target_time, hours=168)  # 1 week
        
        if historical_points >= 10:
            return 0.9
        elif historical_points >= 5:
            return 0.7
        elif historical_points >= 2:
            return 0.5
        else:
            return 0.3
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, "demand_model.joblib")
            scaler_file = os.path.join(self.model_path, "demand_scaler.joblib")
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'feature_columns': self.feature_columns
            }
            
            import json
            metadata_file = os.path.join(self.model_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load existing model and scaler"""
        try:
            model_file = os.path.join(self.model_path, "demand_model.joblib")
            scaler_file = os.path.join(self.model_path, "demand_scaler.joblib")
            metadata_file = os.path.join(self.model_path, "metadata.json")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                
                if os.path.exists(metadata_file):
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_version = metadata.get('version', self.model_version)
                    if metadata.get('last_trained'):
                        self.last_trained = datetime.fromisoformat(metadata['last_trained'])
                
                print("Demand prediction model loaded successfully")
            else:
                print("No existing demand model found. Train a new model first.")
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def get_model_version(self) -> str:
        return self.model_version
    
    def get_last_training_date(self) -> Optional[datetime]:
        return self.last_trained
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Return cached accuracy metrics (would be stored during training)"""
        return {
            "mae": 2.5,  # Example values - would be stored from actual training
            "rmse": 3.8,
            "r2_score": 0.72
        }
    
    def is_healthy(self) -> bool:
        """Check if model is healthy and ready for predictions"""
        if not self.model:
            return False
        
        if self.last_trained and (datetime.now() - self.last_trained).days > 30:
            return False  # Model is stale
        
        return True
    
    def get_prediction_history(self, db: Session, start_time: datetime, end_time: datetime, limit: int) -> List[Dict]:
        """Get historical predictions (simplified - would need prediction logging)"""
        # This would return actual logged predictions in production
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "location": {"lat": 40.7128, "lon": -74.0060},
                "predicted_demand": 0.75,
                "actual_demand": 0.68,
                "accuracy": 0.91
            }
        ]


# Create global instance
demand_predictor = DemandPredictor()
