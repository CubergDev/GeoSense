from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.core.database import get_db
from app.schemas.geotrack import (
    HeatmapRequest, HeatmapResponse, HeatmapCell
)
from app.crud.analytics import analytics_crud
from app.utils.privacy import K_ANON_DEFAULT

router = APIRouter()


@router.post("/heatmap", response_model=HeatmapResponse, summary="Generate demand heatmap")
async def generate_heatmap(
    request: HeatmapRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a demand heatmap.

    Supports P0-spec request with H3 grid and bbox as well as legacy grid_size_meters.
    """
    try:
        # P0 path with H3 grid
        if request.grid and (request.grid.type == "h3"):
            level = request.grid.level or 8
            data = analytics_crud.generate_heatmap_h3(
                db=db,
                bbox=request.bbox,
                level=level,
                metrics=request.metrics,
                k_min=K_ANON_DEFAULT,
            )
            return HeatmapResponse(**data)

        # Legacy path (kept for compatibility)
        if not (request.start_time and request.end_time):
            raise HTTPException(status_code=422, detail="Legacy heatmap requires start_time and end_time")
        heatmap_data = analytics_crud.generate_heatmap(
            db=db,
            start_time=request.start_time,
            end_time=request.end_time,
            grid_size_meters=request.grid_size_meters or 500,
            hour_filter=request.hour_filter,
            day_of_week_filter=request.day_of_week_filter
        )
        # Convert to new response shape minimally
        cells = []
        for c in heatmap_data["cells"]:
            # Map legacy HeatmapCell to P0 cell format (approximate)
            centroid = {"lat": getattr(c, "latitude", None), "lng": getattr(c, "longitude", None)}
            cells.append({
                "cell_id": getattr(c, "grid_id", ""),
                "centroid": centroid,
                "trips": getattr(c, "trip_count", 0),
                "unique_ids": None,
                "avg_speed_kmh": getattr(c, "avg_speed", None),
                "intensity": getattr(c, "demand_intensity", None) or 0.0,
            })
        return HeatmapResponse(
            cells=cells,
            level=None,
            tiles=None,
            meta={"generated_at": datetime.utcnow().isoformat() + "Z", "privacy": {"k_anon": K_ANON_DEFAULT, "suppressed": 0}},
            total_trips=heatmap_data.get("total_trips"),
            analysis_period=f"{request.start_time.isoformat()} to {request.end_time.isoformat()}",
            grid_size_meters=request.grid_size_meters or 500,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@router.get("/trip-patterns", summary="Analyze trip patterns")
async def analyze_trip_patterns(
    start_time: datetime = Query(..., description="Analysis start time"),
    end_time: datetime = Query(..., description="Analysis end time"),
    group_by: str = Query("hour", pattern="^(hour|day|week)$", description="Grouping interval"),
    db: Session = Depends(get_db)
):
    """
    Analyze trip patterns over time to identify trends and seasonality.
    
    **Insights Provided**:
    - Peak usage hours/days
    - Trip volume trends
    - Average trip characteristics
    - Seasonal patterns
    
    **Use Cases**:
    - Capacity planning
    - Dynamic pricing strategies
    - Service scheduling
    - Demand forecasting
    
    - **start_time**: Start of analysis period
    - **end_time**: End of analysis period
    - **group_by**: Time grouping (hour/day/week)
    """
    try:
        patterns = analytics_crud.analyze_trip_patterns(
            db=db,
            start_time=start_time,
            end_time=end_time,
            group_by=group_by
        )
        
        # Map to frontend-expected shape
        time_patterns = []
        for p in patterns:
            avg_duration_sec = None
            if p.get("avg_duration_minutes") is not None:
                try:
                    avg_duration_sec = float(p["avg_duration_minutes"]) * 60.0
                except Exception:
                    avg_duration_sec = None
            avg_speed = None
            try:
                d_km = p.get("avg_distance_km")
                d_min = p.get("avg_duration_minutes")
                if d_km is not None and d_min and d_min > 0:
                    avg_speed = float(d_km) / (float(d_min) / 60.0)
            except Exception:
                avg_speed = None
            time_patterns.append({
                "hour": int(p["time_group"]) if group_by == "hour" and p.get("time_group") is not None else None,
                "trip_count": p.get("trip_count", 0),
                "avg_speed": avg_speed,
                "avg_duration": avg_duration_sec,
            })

        return {
            "time_patterns": time_patterns,
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "group_by": group_by,
            "total_periods": len(time_patterns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@router.get("/demand-zones", summary="Identify high-demand zones")
async def identify_demand_zones(
    start_time: Optional[datetime] = Query(None, description="Analysis start time (legacy)"),
    end_time: Optional[datetime] = Query(None, description="Analysis end time (legacy)"),
    zone_radius_km: float = Query(0.5, gt=0, le=5, description="Zone radius in kilometers (legacy)"),
    min_trips: int = Query(10, ge=1, description="Minimum trips to qualify as a zone (legacy)"),
    eps_m: Optional[int] = Query(None, ge=10, le=5000, description="DBSCAN eps in meters (P0)"),
    min_samples: Optional[int] = Query(None, ge=1, le=1000, description="DBSCAN min_samples (P0)"),
    bbox: Optional[str] = Query(None, description="Bounding box as 'min_lng,min_lat,max_lng,max_lat' (P0)"),
    db: Session = Depends(get_db)
):
    """
    Identify geographic demand zones.

    Supports P0 DBSCAN with eps/min_samples/bbox and legacy KMeans-based parameters as fallback.
    """
    try:
        # P0 path when eps_m provided
        if eps_m is not None and min_samples is not None:
            bbox_list = None
            if bbox:
                try:
                    parts = [float(x) for x in bbox.split(',')]
                    if len(parts) == 4:
                        bbox_list = parts
                except Exception:
                    bbox_list = None
            data = analytics_crud.identify_demand_zones_dbscan(
                db=db,
                bbox=bbox_list,
                eps_m=eps_m,
                min_samples=min_samples,
                k_min=K_ANON_DEFAULT,
            )
            return data

        # Legacy path
        if not (start_time and end_time):
            raise HTTPException(status_code=422, detail="Legacy demand-zones requires start_time and end_time")
        zones = analytics_crud.identify_demand_zones(
            db=db,
            start_time=start_time,
            end_time=end_time,
            zone_radius_km=zone_radius_km,
            min_trips=min_trips
        )
        return {
            "demand_zones": zones,
            "total_zones": len(zones),
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "parameters": {
                "zone_radius_km": zone_radius_km,
                "min_trips": min_trips
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand zone analysis failed: {str(e)}")


@router.get("/safety-analysis", summary="Analyze safety patterns")
async def analyze_safety_patterns(
    start_time: datetime = Query(..., description="Analysis start time"),
    end_time: datetime = Query(..., description="Analysis end time"),
    area_lat: Optional[float] = Query(None, ge=-90, le=90, description="Focus area latitude"),
    area_lon: Optional[float] = Query(None, ge=-180, le=180, description="Focus area longitude"),
    radius_km: Optional[float] = Query(None, gt=0, le=20, description="Focus area radius"),
    db: Session = Depends(get_db)
):
    """
    Analyze safety-related patterns in trip data.
    
    Identifies potentially unsafe scenarios such as:
    - Unusual route deviations
    - Abnormal speed patterns
    - High-risk areas and times
    - Anomalous trip behaviors
    
    **Use Cases**:
    - Driver safety monitoring
    - Risk assessment
    - Emergency response optimization
    - Insurance and compliance
    
    - **start_time**: Start of analysis period
    - **end_time**: End of analysis period
    - **area_lat**: Optional focus area center latitude
    - **area_lon**: Optional focus area center longitude
    - **radius_km**: Optional focus area radius
    """
    try:
        safety_analysis = analytics_crud.analyze_safety_patterns(
            db=db,
            start_time=start_time,
            end_time=end_time,
            area_lat=area_lat,
            area_lon=area_lon,
            radius_km=radius_km
        )
        
        return {
            "safety_metrics": safety_analysis,
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "focus_area": {
                "latitude": area_lat,
                "longitude": area_lon,
                "radius_km": radius_km
            } if area_lat and area_lon else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety analysis failed: {str(e)}")


@router.get("/efficiency-metrics", summary="Calculate efficiency metrics")
async def calculate_efficiency_metrics(
    start_time: datetime = Query(..., description="Analysis start time"),
    end_time: datetime = Query(..., description="Analysis end time"),
    vehicle_type: Optional[str] = Query(None, description="Filter by vehicle type"),
    trip_type: Optional[str] = Query(None, description="Filter by trip type"),
    db: Session = Depends(get_db)
):
    """
    Calculate various efficiency metrics for trips.
    
    **Metrics Calculated**:
    - Route efficiency (actual vs direct distance)
    - Time efficiency (travel time vs optimal)
    - Speed consistency
    - Fuel/energy efficiency estimates
    
    **Use Cases**:
    - Driver performance evaluation
    - Route optimization
    - Cost analysis
    - Environmental impact assessment
    
    - **start_time**: Start of analysis period
    - **end_time**: End of analysis period
    - **vehicle_type**: Optional vehicle type filter
    - **trip_type**: Optional trip type filter
    """
    try:
        efficiency_metrics = analytics_crud.calculate_efficiency_metrics(
            db=db,
            start_time=start_time,
            end_time=end_time,
            vehicle_type=vehicle_type,
            trip_type=trip_type
        )
        
        return {
            "efficiency_metrics": efficiency_metrics,
            "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "filters": {
                "vehicle_type": vehicle_type,
                "trip_type": trip_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Efficiency analysis failed: {str(e)}")


@router.get("/comparative-analysis", summary="Compare time periods")
async def comparative_analysis(
    period1_start: datetime = Query(..., description="First period start time"),
    period1_end: datetime = Query(..., description="First period end time"),
    period2_start: datetime = Query(..., description="Second period start time"), 
    period2_end: datetime = Query(..., description="Second period end time"),
    metrics: List[str] = Query(["trip_count", "avg_distance", "avg_duration"], description="Metrics to compare"),
    db: Session = Depends(get_db)
):
    """
    Compare transportation patterns between two time periods.
    
    **Available Metrics**:
    - trip_count: Total number of trips
    - avg_distance: Average trip distance
    - avg_duration: Average trip duration
    - avg_speed: Average trip speed
    - unique_routes: Number of unique routes
    - demand_density: Trips per area unit
    
    **Use Cases**:
    - Before/after analysis of changes
    - Seasonal comparison
    - Growth tracking
    - Impact assessment
    
    - **period1_start**: First comparison period start
    - **period1_end**: First comparison period end
    - **period2_start**: Second comparison period start
    - **period2_end**: Second comparison period end
    - **metrics**: List of metrics to compare
    """
    try:
        comparison = analytics_crud.comparative_analysis(
            db=db,
            period1_start=period1_start,
            period1_end=period1_end,
            period2_start=period2_start,
            period2_end=period2_end,
            metrics=metrics
        )
        
        return {
            "comparison": comparison,
            "period1": f"{period1_start.isoformat()} to {period1_end.isoformat()}",
            "period2": f"{period2_start.isoformat()} to {period2_end.isoformat()}",
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative analysis failed: {str(e)}")
