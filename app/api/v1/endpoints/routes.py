from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.crud.route import route_crud
# Note: Route/Popular schemas are not used for MVP endpoints under test; removing strict imports to avoid import-time errors.

router = APIRouter()


@router.post("/", summary="Create a new route")
async def create_route(
    route_data: dict,
    db: Session = Depends(get_db)
):
    """
    Create a new route manually.
    
    **Note**: Routes are typically auto-generated from trip analysis, 
    but can be created manually for specific use cases.
    
    - **route_data**: Route information
    """
    return route_crud.create_route(db=db, route_data=route_data)


@router.get("/{route_id}", summary="Get route by ID")
async def get_route(
    route_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific route by its ID.
    
    - **route_id**: The route ID
    """
    route = route_crud.get_route(db=db, route_id=route_id)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route


@router.get("/", summary="List routes")
async def list_routes(
    skip: int = Query(0, ge=0, description="Number of routes to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum routes to return"),
    min_usage: Optional[int] = Query(None, ge=1, description="Minimum usage count"),
    congestion_level: Optional[str] = Query(None, pattern="^(low|medium|high)$", description="Filter by congestion level"),
    db: Session = Depends(get_db)
):
    """
    List routes with optional filters.
    
    Routes are ordered by usage count (most popular first).
    
    - **skip**: Number of routes to skip (pagination)
    - **limit**: Maximum routes to return
    - **min_usage**: Only return routes with at least this many trips
    - **congestion_level**: Filter by congestion level (low/medium/high)
    """
    return route_crud.get_routes(
        db=db,
        skip=skip,
        limit=limit,
        min_usage=min_usage,
        congestion_level=congestion_level
    )


@router.put("/{route_id}", summary="Update route")
async def update_route(
    route_id: int,
    route_update: dict,
    db: Session = Depends(get_db)
):
    """
    Update a route's information.
    
    - **route_id**: The route ID to update
    - **route_update**: Updated route information
    """
    route = route_crud.update_route(db=db, route_id=route_id, route_update=route_update)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route


@router.delete("/{route_id}", summary="Delete route")
async def delete_route(
    route_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a route.
    
    **Warning**: This action cannot be undone.
    
    - **route_id**: The route ID to delete
    """
    success = route_crud.delete_route(db=db, route_id=route_id)
    if not success:
        raise HTTPException(status_code=404, detail="Route not found")
    return {"message": "Route deleted successfully"}


@router.post("/analyze/popular", summary="Identify popular routes/corridors (P0 or legacy)")
async def analyze_popular_routes(
    payload: Optional[dict] = None,
    start_time: Optional[datetime] = Query(None, description="Analysis start time (legacy)"),
    end_time: Optional[datetime] = Query(None, description="Analysis end time (legacy)"),
    min_trips: int = Query(5, ge=1, le=100, description="Minimum trips to identify a route (legacy)"),
    similarity_threshold: float = Query(0.7, ge=0.1, le=1.0, description="Route similarity threshold (legacy)"),
    db: Session = Depends(get_db)
):
    """
    Analyze and return popular corridors.

    P0 mode: provide JSON body with bbox/top_n/simplify_tolerance_m.
    Legacy mode: fall back to query params and return legacy structure.
    """
    try:
        if isinstance(payload, dict):
            bbox = payload.get("bbox")
            top_n = payload.get("top_n", 10)
            simplify = payload.get("simplify_tolerance_m", 50)
            data = route_crud.analyze_popular_corridors(
                db=db,
                bbox=bbox,
                top_n=top_n,
                simplify_tolerance_m=simplify,
            )
            return data

        if not (start_time and end_time):
            raise HTTPException(status_code=422, detail="Legacy mode requires start_time and end_time")
        routes = route_crud.identify_popular_routes(
            db=db,
            start_time=start_time,
            end_time=end_time,
            min_trips=min_trips,
            similarity_threshold=similarity_threshold
        )
        # Convert to response format (dicts to avoid schema dependency)
        popular_routes = []
        for route in routes:
            popular_routes.append({
                "route_id": getattr(route, "route_id", None),
                "name": getattr(route, "name", None),
                "usage_count": getattr(route, "usage_count", None),
                "avg_duration_minutes": ((getattr(route, "avg_duration_seconds", 0) or 0) / 60.0),
                "avg_distance_km": ((getattr(route, "avg_distance_meters", 0) or 0) / 1000.0),
                "demand_score": getattr(route, "demand_score", 0.0) or 0.0,
                "coordinates": [],
            })
        return {
            "popular_routes": popular_routes,
            "total_routes_found": len(routes),
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route analysis failed: {str(e)}")


@router.get("/analyze/bottlenecks", summary="Find traffic bottlenecks")
async def analyze_bottlenecks(
    start_time: datetime = Query(..., description="Analysis start time"),
    end_time: datetime = Query(..., description="Analysis end time"),
    area_lat: float = Query(..., ge=-90, le=90, description="Analysis area center latitude"),
    area_lon: float = Query(..., ge=-180, le=180, description="Analysis area center longitude"),
    radius_km: float = Query(1.0, gt=0, le=10, description="Analysis radius in kilometers"),
    db: Session = Depends(get_db)
):
    """
    Identify traffic bottlenecks in a specific area.
    
    Bottlenecks are identified by analyzing areas where vehicles frequently
    experience low speeds, indicating congestion or infrastructure constraints.
    
    **Use Cases**:
    - Traffic management and optimization
    - Infrastructure planning
    - Real-time congestion monitoring
    - Emergency response planning
    
    - **start_time**: Start of analysis period
    - **end_time**: End of analysis period
    - **area_lat**: Center latitude of analysis area
    - **area_lon**: Center longitude of analysis area
    - **radius_km**: Analysis radius in kilometers (max 10km)
    """
    try:
        bottlenecks = route_crud.find_bottlenecks(
            db=db,
            start_time=start_time,
            end_time=end_time,
            area_lat=area_lat,
            area_lon=area_lon,
            radius_km=radius_km
        )
        
        return {
            "bottlenecks": [
                {
                    "latitude": b["latitude"],
                    "longitude": b["longitude"],
                    "severity_score": b["severity_score"],
                    "avg_delay_seconds": max(0, (20 - b["avg_speed_kmh"]) * 60),  # Estimate delay
                    "affected_trips": b["affected_trips"],
                }
                for b in bottlenecks
            ],
            "total_bottlenecks": len(bottlenecks),
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bottleneck analysis failed: {str(e)}")


@router.get("/{route_id}/analytics", summary="Get route analytics")
async def get_route_analytics(
    route_id: int,
    start_time: datetime = Query(..., description="Analytics start time"),
    end_time: datetime = Query(..., description="Analytics end time"),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics for a specific route.
    
    - **route_id**: The route ID
    - **start_time**: Start of analytics period
    - **end_time**: End of analytics period
    """
    analytics = route_crud.get_route_analytics(
        db=db,
        route_id=route_id,
        start_time=start_time,
        end_time=end_time
    )
    
    if not analytics:
        raise HTTPException(status_code=404, detail="No analytics found for this route and time period")
    
    return analytics


@router.post("/{route_id}/calculate-demand", summary="Calculate route demand")
async def calculate_route_demand(
    route_id: int,
    analysis_date: datetime = Query(..., description="Date to analyze"),
    db: Session = Depends(get_db)
):
    """
    Calculate demand analytics for a specific route and date.
    
    This creates new analytics data for the route, analyzing:
    - Trip volume and frequency
    - Travel time patterns
    - Congestion levels
    - Demand density
    
    - **route_id**: The route ID
    - **analysis_date**: Date to analyze (will analyze the full day)
    """
    analytics = route_crud.calculate_route_demand(
        db=db,
        route_id=route_id,
        analysis_date=analysis_date
    )
    
    if not analytics:
        raise HTTPException(status_code=404, detail="Route not found or no data available for this date")
    
    return analytics
