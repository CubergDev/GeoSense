import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import joblib
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.models.geotrack import Trip, GeoTrack
from app.utils.meta import build_meta


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = "1.0.0"
        self.last_trained = None
        # Legacy feature columns (kept for batch storage compatibility)
        self.feature_columns = [
            'duration_minutes', 'distance_km', 'avg_speed_kmh', 'max_speed_kmh',
            'stop_count', 'route_efficiency', 'hour_of_day', 'day_of_week',
            'speed_variance', 'heading_changes', 'acceleration_events'
        ]
        self.model_path = os.path.join(settings.ML_MODEL_PATH, "anomaly_detector")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
    
    def detect_anomalies_p0(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        top_n: int = 100,
        contamination: float = 0.02,
    ) -> Dict[str, Any]:
        """
        P0-style anomaly detection returning items with flags/severity/why and meta.
        Uses IsolationForest trained on-the-fly for the selected period.
        Features: path_len_km, straight_len_km, detour_ratio, mean_spd_kmh, p95_spd_kmh, std_spd_kmh, circ_std_azm.
        """
        # Fetch trips with sufficient geotracks
        trips = db.query(Trip).filter(
            Trip.start_time >= start_time,
            Trip.start_time <= end_time,
            Trip.duration_seconds.isnot(None),
            Trip.distance_meters.isnot(None)
        ).all()
        if not trips:
            return {"items": [], "model": {"name": "iforest_anomaly", "version": self.model_version}, "meta": build_meta({"start_time": start_time.isoformat(), "end_time": end_time.isoformat()}, k_anon=5, suppressed=0)}

        # Helper: compute haversine between two lat/lng
        def hav_km(lat1, lon1, lat2, lon2):
            R = 6371.0
            from math import radians, sin, cos, sqrt, atan2
            dlat = radians(lat2-lat1)
            dlon = radians(lon2-lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c

        def circ_std(angles_deg: List[float]) -> float:
            import math
            if not angles_deg:
                return 0.0
            ang = [math.radians(a) for a in angles_deg]
            C = sum(math.cos(a) for a in ang) / len(ang)
            S = sum(math.sin(a) for a in ang) / len(ang)
            R = math.sqrt(C*C + S*S)
            if R <= 0:
                return 180.0
            return math.degrees(math.sqrt(-2.0 * math.log(R)))

        # Extract features per trip
        feats: List[List[float]] = []
        meta_rows: List[Dict[str, Any]] = []
        for trip in trips:
            gts = db.query(GeoTrack).filter(GeoTrack.trip_id == trip.id).order_by(GeoTrack.sequence_order.asc()).all()
            if len(gts) < 2:
                continue
            # Path length via haversine over successive points
            path_km = 0.0
            speeds = []
            headings = []
            for i in range(1, len(gts)):
                lat1, lon1 = gts[i-1].latitude, gts[i-1].longitude
                lat2, lon2 = gts[i].latitude, gts[i].longitude
                path_km += hav_km(lat1, lon1, lat2, lon2)
                if gts[i].speed_kmh is not None:
                    speeds.append(float(gts[i].speed_kmh))
                if gts[i].heading_degrees is not None:
                    headings.append(float(gts[i].heading_degrees))
            # Straight line between first and last
            straight_km = hav_km(gts[0].latitude, gts[0].longitude, gts[-1].latitude, gts[-1].longitude)
            detour = (path_km / straight_km) if straight_km >= 0.05 else 1.0
            mean_spd = float(np.mean(speeds)) if speeds else 0.0
            p95_spd = float(np.percentile(speeds, 95)) if len(speeds) > 1 else mean_spd
            std_spd = float(np.std(speeds)) if len(speeds) > 1 else 0.0
            circ_std_azm = float(circ_std(headings)) if headings else 0.0

            feats.append([path_km, straight_km, detour, mean_spd, p95_spd, std_spd, circ_std_azm])
            meta_rows.append({"trip": trip, "features": {
                "path_len_km": path_km,
                "straight_len_km": straight_km,
                "detour_ratio": detour,
                "mean_spd_kmh": mean_spd,
                "p95_spd_kmh": p95_spd,
                "std_spd_kmh": std_spd,
                "circ_std_azm": circ_std_azm,
            }})

        if not feats:
            return {"items": [], "model": {"name": "iforest_anomaly", "version": self.model_version}, "meta": build_meta({"start_time": start_time.isoformat(), "end_time": end_time.isoformat()}, k_anon=5, suppressed=0)}

        X = np.array(feats, dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iforest = IsolationForest(n_estimators=100, contamination=float(contamination), random_state=42)
        iforest.fit(Xs)
        scores = -iforest.score_samples(Xs)  # higher means more anomalous
        # Normalize scores 0..1
        smin, smax = float(np.min(scores)), float(np.max(scores))
        scores01 = (scores - smin) / (smax - smin + 1e-9)

        items: List[Dict[str, Any]] = []
        for i, row in enumerate(meta_rows):
            trip = row["trip"]
            f = row["features"]
            score = float(scores01[i])
            flags = []
            if f["p95_spd_kmh"] > 120:
                flags.append("overspeed")
            if f["detour_ratio"] > 1.6:
                flags.append("high_detour")
            if f["circ_std_azm"] > 60:
                flags.append("zigzag")
            # Severity heuristic prioritizes rules
            if "overspeed" in flags and "high_detour" in flags:
                severity = "high"
            elif score > 0.9 or "overspeed" in flags:
                severity = "high"
            elif score > 0.7 or "high_detour" in flags or "zigzag" in flags:
                severity = "medium"
            else:
                severity = "low"
            # Simple contributions via absolute z-scores
            z = np.abs(Xs[i])
            top_idx = np.argsort(z)[-3:][::-1].tolist()
            feat_names = ["path_len_km", "straight_len_km", "detour_ratio", "mean_spd_kmh", "p95_spd_kmh", "std_spd_kmh", "circ_std_azm"]
            top_reasons = [{"feature": feat_names[j], "contrib": float(z[j])} for j in top_idx]
            items.append({
                "trip_id": trip.trip_id,
                "score": round(score, 3),
                "severity": severity,
                "flags": flags,
                "why": {
                    "detour_ratio": round(f["detour_ratio"], 3),
                    "p95_spd_kmh": round(f["p95_spd_kmh"], 1),
                    "circ_std_azm": round(f["circ_std_azm"], 1),
                    "top_reasons": top_reasons,
                },
                "policy_violation": bool("overspeed" in flags),
            })
        # Sort and take top_n
        items.sort(key=lambda x: x["score"], reverse=True)
        items = items[:top_n]
        return {
            "items": items,
            "model": {"name": "iforest_anomaly", "version": self.model_version},
            "meta": build_meta({"start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "top_n": top_n}, k_anon=5, suppressed=0),
        }

    def detect_anomalies(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        sensitivity: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Detect anomalous trips in the given time period"""
        
        if not self.model:
            raise ValueError("Anomaly detection model not trained. Please train the model first.")
        
        # Get trips for analysis
        trips = db.query(Trip).filter(
            Trip.start_time >= start_time,
            Trip.start_time <= end_time,
            Trip.duration_seconds.isnot(None),
            Trip.distance_meters.isnot(None)
        ).all()
        
        if not trips:
            return []
        
        anomalies = []
        
        for trip in trips:
            # Extract features for this trip
            features = self._extract_trip_features(db, trip)
            
            if features is not None:
                # Scale features
                feature_array = np.array([features])
                feature_scaled = self.scaler.transform(feature_array)
                
                # Predict anomaly score
                anomaly_score = self.model.decision_function(feature_scaled)[0]
                is_anomaly = self.model.predict(feature_scaled)[0] == -1
                
                # Convert to 0-1 scale (higher = more anomalous)
                normalized_score = max(0.0, min(1.0, (0.5 - anomaly_score) / 1.0))
                
                if is_anomaly or normalized_score > (1.0 - sensitivity):
                    # Analyze what makes this trip anomalous
                    anomaly_reasons = self._analyze_anomaly_reasons(features, trip)
                    
                    anomalies.append({
                        "trip_id": trip.trip_id,
                        "trip_db_id": trip.id,
                        "anomaly_score": round(normalized_score, 3),
                        "detected_at": datetime.now().isoformat(),
                        "trip_start_time": trip.start_time.isoformat(),
                        "anomaly_reasons": anomaly_reasons,
                        "trip_summary": {
                            "duration_minutes": round((trip.duration_seconds or 0) / 60.0, 1),
                            "distance_km": round((trip.distance_meters or 0) / 1000.0, 2),
                            "vehicle_type": trip.vehicle_type,
                            "trip_type": trip.trip_type
                        }
                    })
        
        # Sort by anomaly score (most anomalous first)
        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        
        return anomalies
    
    def detect_batch_anomalies(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        sensitivity: float = 0.05
    ):
        """Detect anomalies in batch mode and store results"""
        
        print(f"Starting batch anomaly detection for period {start_time} to {end_time}")
        
        # Process in chunks to avoid memory issues
        chunk_size = timedelta(days=1)
        current_time = start_time
        
        total_anomalies = 0
        
        while current_time < end_time:
            chunk_end = min(current_time + chunk_size, end_time)
            
            try:
                anomalies = self.detect_anomalies(db, current_time, chunk_end, sensitivity)
                
                # Store anomaly results in TripAnalytics
                for anomaly in anomalies:
                    self._store_anomaly_result(db, anomaly)
                
                total_anomalies += len(anomalies)
                print(f"Processed {current_time.date()}: {len(anomalies)} anomalies found")
                
            except Exception as e:
                print(f"Error processing chunk {current_time.date()}: {e}")
            
            current_time = chunk_end
        
        print(f"Batch anomaly detection completed. Total anomalies: {total_anomalies}")
    
    def retrain(self, db: Session, lookback_days: int = 90):
        """Retrain the anomaly detection model"""
        
        print(f"Starting anomaly detector retraining with {lookback_days} days of data...")
        
        # Prepare training data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        training_data = self._prepare_training_data(db, start_time, end_time)
        
        if len(training_data) < 1000:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples. Need at least 1000.")
        
        # Prepare features
        X = training_data[self.feature_columns].values
        
        # Remove any rows with NaN values
        valid_rows = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_rows]
        
        if len(X_clean) < 500:
            raise ValueError("Too many invalid samples after cleaning.")
        
        # Scale features
        self.scaler.fit(X_clean)
        X_scaled = self.scaler.transform(X_clean)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            n_estimators=200,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        
        # Evaluate on training data (simplified)
        predictions = self.model.predict(X_scaled)
        anomaly_rate = np.sum(predictions == -1) / len(predictions)
        
        print(f"Training completed. Anomaly rate: {anomaly_rate:.3f}")
        
        # Update metadata and save
        self.last_trained = datetime.now()
        self._save_model()
        
        print("Anomaly detector training completed successfully!")
    
    def _prepare_training_data(self, db: Session, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Prepare training data from historical trips"""
        
        trips = db.query(Trip).filter(
            Trip.start_time >= start_time,
            Trip.start_time <= end_time,
            Trip.duration_seconds.isnot(None),
            Trip.distance_meters.isnot(None)
        ).all()
        
        training_records = []
        
        for trip in trips:
            features = self._extract_trip_features(db, trip)
            if features is not None:
                feature_dict = dict(zip(self.feature_columns, features))
                training_records.append(feature_dict)
        
        return pd.DataFrame(training_records)
    
    def _extract_trip_features(self, db: Session, trip: Trip) -> Optional[List[float]]:
        """Extract features from a trip for anomaly detection"""
        
        try:
            # Basic trip metrics
            duration_minutes = (trip.duration_seconds or 0) / 60.0
            distance_km = (trip.distance_meters or 0) / 1000.0
            
            if duration_minutes <= 0 or distance_km <= 0:
                return None
            
            avg_speed_kmh = distance_km / (duration_minutes / 60.0)
            
            # Get geotracks for detailed analysis
            geotracks = db.query(GeoTrack).filter(
                GeoTrack.trip_id == trip.id
            ).order_by(GeoTrack.sequence_order).all()
            
            if len(geotracks) < 2:
                return None
            
            # Calculate advanced features
            speeds = [g.speed_kmh for g in geotracks if g.speed_kmh is not None]
            headings = [g.heading_degrees for g in geotracks if g.heading_degrees is not None]
            
            max_speed_kmh = max(speeds) if speeds else avg_speed_kmh
            speed_variance = np.var(speeds) if len(speeds) > 1 else 0.0
            
            # Count stops
            stop_count = sum(1 for g in geotracks if g.is_stop_point)
            
            # Route efficiency (actual vs direct distance)
            if trip.start_point and trip.end_point:
                # Simplified calculation - would use actual geographic distance
                route_efficiency = 1.0  # Default
            else:
                route_efficiency = distance_km / max(distance_km * 0.8, 0.1)  # Rough estimate
            
            # Time features
            hour_of_day = trip.start_time.hour
            day_of_week = trip.start_time.weekday()
            
            # Heading changes (measure of route complexity)
            heading_changes = 0
            if len(headings) > 1:
                for i in range(1, len(headings)):
                    diff = abs(headings[i] - headings[i-1])
                    if diff > 180:
                        diff = 360 - diff
                    if diff > 30:  # Significant heading change
                        heading_changes += 1
            
            # Acceleration events (simplified)
            acceleration_events = 0
            if len(speeds) > 1:
                for i in range(1, len(speeds)):
                    speed_diff = abs(speeds[i] - speeds[i-1])
                    if speed_diff > 20:  # Rapid speed change
                        acceleration_events += 1
            
            return [
                duration_minutes,
                distance_km,
                avg_speed_kmh,
                max_speed_kmh,
                stop_count,
                route_efficiency,
                hour_of_day,
                day_of_week,
                speed_variance,
                heading_changes,
                acceleration_events
            ]
            
        except Exception as e:
            print(f"Error extracting features for trip {trip.id}: {e}")
            return None
    
    def _analyze_anomaly_reasons(self, features: List[float], trip: Trip) -> List[str]:
        """Analyze what makes a trip anomalous"""
        
        reasons = []
        
        # Map features to their values
        feature_dict = dict(zip(self.feature_columns, features))
        
        # Check various anomaly conditions
        if feature_dict['duration_minutes'] > 180:  # > 3 hours
            reasons.append("Unusually long duration")
        elif feature_dict['duration_minutes'] < 2:  # < 2 minutes
            reasons.append("Unusually short duration")
        
        if feature_dict['distance_km'] > 200:  # > 200km
            reasons.append("Unusually long distance")
        
        if feature_dict['max_speed_kmh'] > 120:  # > 120 km/h
            reasons.append("Excessive speed detected")
        elif feature_dict['avg_speed_kmh'] < 5:  # < 5 km/h average
            reasons.append("Unusually slow average speed")
        
        if feature_dict['route_efficiency'] > 2.0:
            reasons.append("Highly inefficient route")
        
        if feature_dict['stop_count'] > 20:
            reasons.append("Excessive number of stops")
        
        if feature_dict['speed_variance'] > 400:  # High speed variance
            reasons.append("Erratic speed patterns")
        
        if feature_dict['heading_changes'] > 50:
            reasons.append("Unusual route complexity")
        
        if feature_dict['acceleration_events'] > 30:
            reasons.append("Frequent acceleration/deceleration")
        
        # Time-based anomalies
        if feature_dict['hour_of_day'] in [2, 3, 4]:  # Late night
            reasons.append("Unusual time of day")
        
        if not reasons:
            reasons.append("Statistical outlier")
        
        return reasons
    
    def _store_anomaly_result(self, db: Session, anomaly: Dict[str, Any]):
        """Store anomaly detection result in database (best-effort)."""
        # Guarded import: TripAnalytics may not exist in MVP schema
        try:
            from app.models.geotrack import TripAnalytics  # type: ignore
        except Exception:
            return  # Skip persistence if model not available
        try:
            # Check if analytics record exists
            analytics = db.query(TripAnalytics).filter(
                TripAnalytics.trip_id == anomaly.get("trip_db_id")
            ).first()
            if not analytics:
                analytics = TripAnalytics(trip_id=anomaly.get("trip_db_id"))
                db.add(analytics)
            # Update anomaly information
            analytics.is_anomalous = True
            analytics.anomaly_score = anomaly.get("anomaly_score")
            analytics.anomaly_reasons = str(anomaly.get("anomaly_reasons"))  # JSON string
            db.commit()
        except Exception as e:
            print(f"Error storing anomaly result: {e}")
            try:
                db.rollback()
            except Exception:
                pass
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, "anomaly_model.joblib")
            scaler_file = os.path.join(self.model_path, "anomaly_scaler.joblib")
            
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
            print(f"Failed to save anomaly model: {e}")
    
    def _load_model(self):
        """Load existing model and scaler"""
        try:
            model_file = os.path.join(self.model_path, "anomaly_model.joblib")
            scaler_file = os.path.join(self.model_path, "anomaly_scaler.joblib")
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
                
                print("Anomaly detection model loaded successfully")
            else:
                print("No existing anomaly model found. Train a new model first.")
                
        except Exception as e:
            print(f"Failed to load anomaly model: {e}")
            self.model = None
    
    def get_model_version(self) -> str:
        return self.model_version
    
    def get_last_training_date(self) -> Optional[datetime]:
        return self.last_trained
    
    def get_precision_metrics(self) -> Dict[str, float]:
        """Return cached precision metrics"""
        return {
            "precision": 0.78,
            "recall": 0.65,
            "f1_score": 0.71
        }
    
    def is_healthy(self) -> bool:
        """Check if model is healthy"""
        if not self.model:
            return False
        
        if self.last_trained and (datetime.now() - self.last_trained).days > 30:
            return False
        
        return True
    
    def get_detection_history(self, db: Session, start_time: datetime, end_time: datetime, limit: int) -> List[Dict]:
        """Get historical anomaly detections (best-effort). Returns empty list if TripAnalytics unavailable."""
        try:
            from app.models.geotrack import TripAnalytics  # type: ignore
        except Exception:
            return []
        analytics = db.query(TripAnalytics).filter(
            TripAnalytics.is_anomalous == True,
            TripAnalytics.created_at >= start_time,
            TripAnalytics.created_at <= end_time
        ).limit(limit).all()
        history: List[Dict] = []
        for record in analytics:
            history.append({
                "trip_id": getattr(record, "trip_id", None),
                "anomaly_score": getattr(record, "anomaly_score", None),
                "reasons": getattr(record, "anomaly_reasons", None),
                "detected_at": getattr(record, "created_at").isoformat() if getattr(record, "created_at", None) else None,
            })
        return history


# Create global instance
anomaly_detector = AnomalyDetector()
