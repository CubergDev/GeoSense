from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from geoalchemy2.functions import ST_DWithin, ST_Distance, ST_MakePoint
from geopy.distance import geodesic
import numpy as np
from datetime import datetime, timedelta

from app.models.geotrack import Trip, GeoTrack
from app.schemas.geotrack import TripCreate, TripUpdate, GeoTrackCreate
from app.core.config import settings


class TripCRUD:
    def create_trip(self, db: Session, trip_data: TripCreate) -> Trip:
        """Create a new trip with geotracks (apply anonymization/noise)"""
        
        # Create trip record (anonymization enabled)
        db_trip = Trip(
            start_time=trip_data.start_time,
            end_time=trip_data.end_time,
            trip_type=trip_data.trip_type,
            vehicle_type=trip_data.vehicle_type,
            weather_condition=trip_data.weather_condition,
            is_anonymized=True,
            anonymization_method="spatial_temporal_blur"
        )
        db.add(db_trip)
        db.flush()  # Get the trip ID
        
        # Apply anonymization/noise to input geotracks
        input_geotracks = trip_data.geotracks
        anonymized_geotracks = self._anonymize_geotracks(input_geotracks)
        
        # Create geotrack records (noise applied)
        db_geotracks = []
        for idx, geotrack_data in enumerate(anonymized_geotracks):
            seq = geotrack_data.sequence_order if getattr(geotrack_data, 'sequence_order', None) is not None else idx
            if getattr(geotrack_data, 'timestamp', None) is not None:
                ts = geotrack_data.timestamp
            elif trip_data.start_time is not None:
                ts = trip_data.start_time + timedelta(seconds=seq)
            else:
                ts = datetime.utcnow()
            
            db_geotrack = GeoTrack(
                trip_id=db_trip.id,
                latitude=geotrack_data.latitude,
                longitude=geotrack_data.longitude,
                altitude=geotrack_data.altitude,
                timestamp=ts,
                sequence_order=seq,
                speed_kmh=geotrack_data.speed_kmh,
                heading_degrees=geotrack_data.heading_degrees,
                accuracy_meters=geotrack_data.accuracy_meters,
                is_stop_point=geotrack_data.is_stop_point,
                dwell_time_seconds=geotrack_data.dwell_time_seconds,
                point=f"POINT({geotrack_data.longitude} {geotrack_data.latitude})",
                noise_applied=True,
                spatial_blur_radius=None
            )
            db_geotracks.append(db_geotrack)
            db.add(db_geotrack)
        
        # Calculate trip metrics
        self._calculate_trip_metrics(db_trip, db_geotracks)
        
        db.commit()
        db.refresh(db_trip)
        return db_trip
    
    def get_trip(self, db: Session, trip_id: int) -> Optional[Trip]:
        """Get a trip by ID"""
        return db.query(Trip).filter(Trip.id == trip_id).first()
    
    def get_trip_by_trip_id(self, db: Session, trip_id: str) -> Optional[Trip]:
        """Get a trip by trip_id string"""
        return db.query(Trip).filter(Trip.trip_id == trip_id).first()
    
    def get_trips(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        trip_type: Optional[str] = None,
        vehicle_type: Optional[str] = None
    ) -> List[Trip]:
        """Get trips with optional filters"""
        query = db.query(Trip)
        
        if start_time:
            query = query.filter(Trip.start_time >= start_time)
        if end_time:
            query = query.filter(Trip.start_time <= end_time)
        if trip_type:
            query = query.filter(Trip.trip_type == trip_type)
        if vehicle_type:
            query = query.filter(Trip.vehicle_type == vehicle_type)
        
        return query.order_by(desc(Trip.created_at)).offset(skip).limit(limit).all()
    
    def update_trip(self, db: Session, trip_id: int, trip_update: TripUpdate) -> Optional[Trip]:
        """Update a trip"""
        db_trip = self.get_trip(db, trip_id)
        if not db_trip:
            return None
        
        update_data = trip_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_trip, field, value)
        
        # Recalculate duration if end_time is updated
        if 'end_time' in update_data and db_trip.end_time:
            duration = (db_trip.end_time - db_trip.start_time).total_seconds()
            db_trip.duration_seconds = int(duration)
        
        db.commit()
        db.refresh(db_trip)
        return db_trip
    
    def delete_trip(self, db: Session, trip_id: int) -> bool:
        """Delete a trip and its geotracks"""
        db_trip = self.get_trip(db, trip_id)
        if not db_trip:
            return False
        
        db.delete(db_trip)
        db.commit()
        return True
    
    def get_trips_in_area(
        self,
        db: Session,
        lat: float,
        lon: float,
        radius_km: float,
        limit: int = 100
    ) -> List[Trip]:
        """Get trips that pass through a specific area"""
        # Convert radius to meters for PostGIS
        radius_meters = radius_km * 1000
        
        # Find trips with geotracks within the specified area
        subquery = db.query(GeoTrack.trip_id).filter(
            ST_DWithin(
                GeoTrack.point,
                ST_MakePoint(lon, lat),
                radius_meters
            )
        ).subquery()
        
        return db.query(Trip).filter(
            Trip.id.in_(subquery)
        ).limit(limit).all()
    
    def _anonymize_geotracks(self, geotracks: List[GeoTrackCreate]) -> List[GeoTrackCreate]:
        """Apply spatial-temporal anonymization to geotracks"""
        anonymized = []
        
        for geotrack in geotracks:
            # Apply spatial noise (50-100m radius)
            noise_radius = np.random.uniform(50, 100)  # meters
            bearing = np.random.uniform(0, 360)  # degrees
            
            # Calculate new position using geopy
            original_point = (geotrack.latitude, geotrack.longitude)
            noise_distance_km = noise_radius / 1000.0
            
            # Apply noise
            noisy_lat = geotrack.latitude + (noise_distance_km / 111.0) * np.cos(np.radians(bearing))
            noisy_lon = geotrack.longitude + (noise_distance_km / (111.0 * np.cos(np.radians(geotrack.latitude)))) * np.sin(np.radians(bearing))
            
            # Create anonymized geotrack
            anonymized_track = GeoTrackCreate(
                latitude=noisy_lat,
                longitude=noisy_lon,
                altitude=geotrack.altitude,
                timestamp=geotrack.timestamp,
                sequence_order=geotrack.sequence_order,
                speed_kmh=geotrack.speed_kmh,
                heading_degrees=geotrack.heading_degrees,
                accuracy_meters=geotrack.accuracy_meters,
                is_stop_point=geotrack.is_stop_point,
                dwell_time_seconds=geotrack.dwell_time_seconds
            )
            anonymized.append(anonymized_track)
        
        return anonymized
    
    def _calculate_trip_metrics(self, trip: Trip, geotracks: List[GeoTrack]):
        """Calculate trip distance, duration, and other metrics"""
        if len(geotracks) < 2:
            return
        
        # Sort by sequence order
        sorted_tracks = sorted(geotracks, key=lambda x: x.sequence_order)
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(sorted_tracks)):
            prev_track = sorted_tracks[i-1]
            curr_track = sorted_tracks[i]
            
            distance = geodesic(
                (prev_track.latitude, prev_track.longitude),
                (curr_track.latitude, curr_track.longitude)
            ).meters
            total_distance += distance
        
        # Set trip metrics
        trip.distance_meters = total_distance
        trip.start_point = f"POINT({sorted_tracks[0].longitude} {sorted_tracks[0].latitude})"
        trip.end_point = f"POINT({sorted_tracks[-1].longitude} {sorted_tracks[-1].latitude})"
        
        # Calculate duration if end_time is set
        if trip.end_time:
            duration = (trip.end_time - trip.start_time).total_seconds()
            trip.duration_seconds = int(duration)


# Create instance
trip_crud = TripCRUD()
