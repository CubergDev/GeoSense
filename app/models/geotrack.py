from sqlalchemy import Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from app.core.database import Base
import uuid


class Trip(Base):
    """Trip entity representing a complete journey"""
    __tablename__ = "trips"
    
    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Anonymized trip metadata
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    distance_meters = Column(Float, nullable=True)
    
    # Geospatial data
    start_point = Column(Geometry('POINT', srid=4326), nullable=True)
    end_point = Column(Geometry('POINT', srid=4326), nullable=True)
    
    # Trip characteristics
    trip_type = Column(String, nullable=True)  # 'pickup', 'delivery', 'commute', etc.
    vehicle_type = Column(String, nullable=True)
    weather_condition = Column(String, nullable=True)
    
    # Privacy and anonymization
    is_anonymized = Column(Boolean, default=True)
    anonymization_method = Column(String, default="spatial_temporal_blur")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    geotracks = relationship("GeoTrack", back_populates="trip", cascade="all, delete-orphan")


class GeoTrack(Base):
    """Individual GPS points forming a trip track"""
    __tablename__ = "geotracks"
    
    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    
    # Geospatial data
    point = Column(Geometry('POINT', srid=4326), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float, nullable=True)
    
    # Temporal data
    timestamp = Column(DateTime(timezone=True), nullable=False)
    sequence_order = Column(Integer, nullable=False)  # Order within trip
    
    # Movement data
    speed_kmh = Column(Float, nullable=True)
    heading_degrees = Column(Float, nullable=True)  # 0-360 degrees
    accuracy_meters = Column(Float, nullable=True)
    
    # Derived features
    is_stop_point = Column(Boolean, default=False)
    dwell_time_seconds = Column(Integer, nullable=True)
    
    # Anonymization
    noise_applied = Column(Boolean, default=True)
    spatial_blur_radius = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    trip = relationship("Trip", back_populates="geotracks")


