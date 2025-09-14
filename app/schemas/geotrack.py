from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class TripTypeEnum(str, Enum):
    PICKUP = "pickup"
    DELIVERY = "delivery"
    COMMUTE = "commute"
    LEISURE = "leisure"
    BUSINESS = "business"


class VehicleTypeEnum(str, Enum):
    CAR = "car"
    TAXI = "taxi"
    BIKE = "bike"
    SCOOTER = "scooter"
    TRUCK = "truck"
    BUS = "bus"


class WeatherEnum(str, Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    CLOUDY = "cloudy"
    SNOWY = "snowy"
    FOGGY = "foggy"


# GeoTrack Schemas
class GeoTrackBase(BaseModel):
    # CSV-aligned aliases: lat, lng, alt, spd, azm, randomized_id
    latitude: float = Field(..., alias="lat", ge=-90, le=90, description="Latitude in WGS84")
    longitude: float = Field(..., alias="lng", ge=-180, le=180, description="Longitude in WGS84")
    altitude: Optional[float] = Field(None, alias="alt", description="Altitude in meters")
    timestamp: Optional[datetime] = None
    sequence_order: Optional[int] = Field(None, ge=0, description="Order of point in trip")
    speed_kmh: Optional[float] = Field(None, alias="spd", ge=0, description="Speed as provided in CSV (units TBD)")
    heading_degrees: Optional[float] = Field(None, alias="azm", ge=0, lt=360, description="Azimuth 0-360 degrees")
    accuracy_meters: Optional[float] = Field(None, ge=0, description="GPS accuracy in meters")
    is_stop_point: bool = False
    dwell_time_seconds: Optional[int] = Field(None, ge=0, description="Time spent at stop")
    randomized_id: Optional[int] = Field(None, alias="randomized_id", description="Randomized source identifier")

    class Config:
        allow_population_by_field_name = True


class GeoTrackCreate(GeoTrackBase):
    pass


class GeoTrackUpdate(BaseModel):
    speed_kmh: Optional[float] = Field(None, ge=0)
    heading_degrees: Optional[float] = Field(None, ge=0, lt=360)
    is_stop_point: Optional[bool] = None
    dwell_time_seconds: Optional[int] = Field(None, ge=0)


class GeoTrack(GeoTrackBase):
    id: int
    trip_id: int
    noise_applied: bool
    spatial_blur_radius: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


# Trip Schemas
class TripBase(BaseModel):
    start_time: datetime
    end_time: Optional[datetime] = None
    trip_type: Optional[TripTypeEnum] = None
    vehicle_type: Optional[VehicleTypeEnum] = None
    weather_condition: Optional[WeatherEnum] = None


class TripCreate(TripBase):
    geotracks: List[GeoTrackCreate] = Field(..., min_items=2, description="At least 2 GPS points required")
    
    @validator('geotracks')
    def validate_geotracks_order(cls, v):
        """Ensure geotracks contain at least 2 points and auto-assign sequence when missing"""
        if len(v) < 2:
            raise ValueError("Trip must have at least 2 GPS points")
        
        # Auto-assign sequence_order if missing and sort if provided out of order
        for i, track in enumerate(v):
            if getattr(track, 'sequence_order', None) is None:
                try:
                    track.sequence_order = i
                except Exception:
                    pass
        try:
            orders = [track.sequence_order for track in v]
            if all(o is not None for o in orders) and orders != sorted(orders):
                v = sorted(v, key=lambda t: t.sequence_order)
        except Exception:
            pass
        return v


class TripUpdate(BaseModel):
    end_time: Optional[datetime] = None
    trip_type: Optional[TripTypeEnum] = None
    vehicle_type: Optional[VehicleTypeEnum] = None
    weather_condition: Optional[WeatherEnum] = None


class Trip(TripBase):
    id: int
    trip_id: str
    duration_seconds: Optional[int]
    distance_meters: Optional[float]
    is_anonymized: bool
    anonymization_method: str
    created_at: datetime
    updated_at: Optional[datetime]
    geotracks: List[GeoTrack] = []

    class Config:
        from_attributes = True


# Analytics Schemas
class HeatmapCell(BaseModel):
    grid_id: str
    trip_count: int
    avg_duration: Optional[float] = None
    avg_speed: Optional[float] = None
    demand_intensity: Optional[float] = None
    latitude: float
    longitude: float


class HeatmapGrid(BaseModel):
    type: str = Field("h3", description="Grid type: 'h3'")
    level: Optional[int] = Field(None, ge=0, le=15, description="H3 resolution level")


class HeatmapResponse(BaseModel):
    cells: List[Dict[str, Any]]
    level: Optional[int] = None
    tiles: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    total_trips: Optional[int] = None
    analysis_period: Optional[str] = None
    grid_size_meters: Optional[int] = None


class HeatmapRequest(BaseModel):
    # P0 fields
    bbox: Optional[List[float]] = Field(None, description="[min_lng, min_lat, max_lng, max_lat]")
    grid: Optional[HeatmapGrid] = None
    metrics: Optional[List[str]] = None
    # Legacy fields
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    grid_size_meters: Optional[int] = Field(None, gt=0)
    hour_filter: Optional[int] = Field(None, ge=0, le=23)
    day_of_week_filter: Optional[int] = Field(None, ge=0, le=6)


