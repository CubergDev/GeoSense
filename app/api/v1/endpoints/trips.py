from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.crud.trip import trip_crud
from app.schemas.geotrack import Trip, TripCreate, TripUpdate
from app.utils.ingest_csv import ingest_csv_to_db

router = APIRouter()


@router.post("/", response_model=Trip, summary="Create a new trip")
async def create_trip(
    trip_data: TripCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new trip with geotracks.
    
    **Privacy Note**: All GPS coordinates are automatically anonymized using 
    spatial-temporal blurring to protect user privacy.
    
    - **trip_data**: Trip information including geotracks
    - Returns the created trip with anonymized coordinates
    """
    try:
        return trip_crud.create_trip(db=db, trip_data=trip_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create trip: {str(e)}")


@router.get("/{trip_id}", response_model=Trip, summary="Get trip by ID")
async def get_trip(
    trip_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific trip by its ID.
    
    - **trip_id**: The trip ID
    - Returns trip details with anonymized geotracks
    """
    trip = trip_crud.get_trip(db=db, trip_id=trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@router.get("/", response_model=List[Trip], summary="List trips with filters")
async def list_trips(
    skip: int = Query(0, ge=0, description="Number of trips to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trips to return"),
    start_time: Optional[str] = Query(None, description="Filter trips after this time (ISO format)"),
    end_time: Optional[str] = Query(None, description="Filter trips before this time (ISO format)"),
    trip_type: Optional[str] = Query(None, description="Filter by trip type"),
    vehicle_type: Optional[str] = Query(None, description="Filter by vehicle type"),
    db: Session = Depends(get_db)
):
    """
    List trips with optional filters.
    
    - **skip**: Number of trips to skip (pagination)
    - **limit**: Maximum trips to return
    - **start_time**: Filter trips after this timestamp
    - **end_time**: Filter trips before this timestamp  
    - **trip_type**: Filter by trip type (pickup, delivery, commute, etc.)
    - **vehicle_type**: Filter by vehicle type (car, taxi, bike, etc.)
    """
    return trip_crud.get_trips(
        db=db,
        skip=skip,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        trip_type=trip_type,
        vehicle_type=vehicle_type
    )


@router.put("/{trip_id}", response_model=Trip, summary="Update trip")
async def update_trip(
    trip_id: int,
    trip_update: TripUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a trip's metadata.
    
    **Note**: Geotracks cannot be modified after creation to maintain data integrity.
    
    - **trip_id**: The trip ID to update
    - **trip_update**: Updated trip information
    """
    trip = trip_crud.update_trip(db=db, trip_id=trip_id, trip_update=trip_update)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@router.delete("/{trip_id}", summary="Delete trip")
async def delete_trip(
    trip_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a trip and all its associated geotracks.
    
    **Warning**: This action cannot be undone.
    
    - **trip_id**: The trip ID to delete
    """
    success = trip_crud.delete_trip(db=db, trip_id=trip_id)
    if not success:
        raise HTTPException(status_code=404, detail="Trip not found")
    return {"message": "Trip deleted successfully"}


@router.get("/search/area", response_model=List[Trip], summary="Find trips in area")
async def get_trips_in_area(
    lat: float = Query(..., ge=-90, le=90, description="Center latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Center longitude"), 
    radius_km: float = Query(..., gt=0, le=50, description="Search radius in kilometers"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum trips to return"),
    db: Session = Depends(get_db)
):
    """
    Find trips that pass through a specific geographic area.
    
    Useful for:
    - Analyzing traffic in specific zones
    - Finding trips near points of interest
    - Regional demand analysis
    
    - **lat**: Center point latitude
    - **lon**: Center point longitude
    - **radius_km**: Search radius in kilometers (max 50km)
    - **limit**: Maximum number of trips to return
    """
    return trip_crud.get_trips_in_area(
        db=db,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        limit=limit
    )



@router.post("/ingest-csv", summary="Ingest trips from CSV")
async def ingest_csv(
    path: Optional[str] = Query(None, description="(Ignored) CSV path; service uses settings.DATA_CSV_PATH"),
    truncate: bool = Query(True, description="Truncate existing trips/geotracks before ingestion"),
    db: Session = Depends(get_db)
):
    """
    Load trips and geotracks from the supplied CSV into the database.
    - Columns required: randomized_id, lat, lng, alt, spd, azm
    - Auto-detects speed units by p95 and converts to km/h
    - Groups by randomized_id as one trip, orders points as they appear
    - Works in both dev and prod; always reads from settings.DATA_CSV_PATH
    """
    try:
        summary = ingest_csv_to_db(db=db, csv_path=settings.DATA_CSV_PATH, truncate=truncate)
        return {
            "status": "ok",
            "summary": summary,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV ingestion failed: {str(e)}")
