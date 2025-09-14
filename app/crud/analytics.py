from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, text
from datetime import datetime, timedelta
import time
import numpy as np
from geopy.distance import geodesic
from collections import defaultdict

try:
    import h3
except Exception:
    h3 = None

from app.models.geotrack import Trip, GeoTrack
from app.schemas.geotrack import HeatmapCell
from app.utils.privacy import apply_k_anon, K_ANON_DEFAULT
from app.utils.meta import build_meta
from app.utils.speed import detect_speed_unit_for_period
from app.utils.cache import get_cached, set_cached


class AnalyticsCRUD:
    def generate_heatmap_h3(
        self,
        db: Session,
        bbox: Optional[List[float]] = None,
        level: int = 8,
        metrics: Optional[List[str]] = None,
        k_min: int = K_ANON_DEFAULT,
    ) -> Dict[str, Any]:
        """Generate H3 heatmap from GeoTrack points with optional bbox filter.
        Uses simple file cache keyed by bbox/level/metrics.
        """
        if h3 is None:
            raise RuntimeError("h3 library not installed")

        metrics = metrics or ["points", "unique_ids"]
        cache_params = {"bbox": bbox, "level": int(level), "metrics": sorted(metrics)}
        cached, read_ms = get_cached("heatmap_h3", cache_params)
        if cached:
            try:
                # Refresh meta timestamp and timings
                if isinstance(cached, dict) and "meta" in cached:
                    cached["meta"]["generated_at"] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
                    tm = cached["meta"].get("timings_ms") or {}
                    tm["read_cache"] = read_ms or 0
                    cached["meta"]["timings_ms"] = tm
            except Exception:
                pass
            return cached

        t0 = time.time()
        q = db.query(GeoTrack.latitude, GeoTrack.longitude, GeoTrack.trip_id, GeoTrack.speed_kmh)
        if bbox:
            min_lng, min_lat, max_lng, max_lat = bbox
            q = q.filter(
                GeoTrack.latitude >= min_lat,
                GeoTrack.latitude <= max_lat,
                GeoTrack.longitude >= min_lng,
                GeoTrack.longitude <= max_lng,
            )
        rows = q.all()

        cell_counts: Dict[str, int] = defaultdict(int)
        cell_unique: Dict[str, set] = defaultdict(set)
        cell_speed_sum: Dict[str, float] = defaultdict(float)
        cell_speed_n: Dict[str, int] = defaultdict(int)

        for lat, lng, trip_id, speed in rows:
            try:
                cell = h3.geo_to_h3(lat, lng, level)
            except Exception:
                # skip invalid coordinates
                continue
            cell_counts[cell] += 1
            if "unique_ids" in metrics and trip_id is not None:
                cell_unique[cell].add(trip_id)
            if speed is not None:
                cell_speed_sum[cell] += float(speed)
                cell_speed_n[cell] += 1

        # Build cells list
        max_count = max(cell_counts.values()) if cell_counts else 1
        items: List[Dict[str, Any]] = []
        for cell, cnt in cell_counts.items():
            lat_c, lng_c = h3.h3_to_geo(cell)
            unique_n = len(cell_unique[cell]) if "unique_ids" in metrics else None
            avg_spd = (cell_speed_sum[cell] / cell_speed_n[cell]) if cell_speed_n[cell] > 0 else None
            item = {
                "cell_id": cell,
                "centroid": {"lat": lat_c, "lng": lng_c},
                "trips": cnt,
                "unique_ids": unique_n,
                "avg_speed_kmh": avg_spd,
                "intensity": (cnt / max_count) if max_count else 0.0,
            }
            items.append(item)

        suppressed_before = len(items)
        items = apply_k_anon(items, count_key="trips", k_min=k_min)
        suppressed = suppressed_before - len(items)

        # Sort by trips desc
        items.sort(key=lambda x: x["trips"], reverse=True)
        t1 = time.time()

        res = {
            "cells": items,
            "level": level,
            "tiles": {
                "mvt_url": f"/tiles/h3/{{z}}/{{x}}/{{y}}.mvt?level={level}",
                "legend_breaks": [5, 20, 50, 100, 200],
            },
            "meta": build_meta(
                query_params={"bbox": bbox, "grid": {"type": "h3", "level": level}, "metrics": metrics},
                k_anon=k_min,
                suppressed=suppressed,
                timings_ms={"compute": int((t1 - t0) * 1000)},
            ),
        }
        # Store in cache (best-effort)
        try:
            set_cached("heatmap_h3", cache_params, res)
        except Exception:
            pass
        return res

    def generate_heatmap(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        grid_size_meters: int = 500,
        hour_filter: Optional[int] = None,
        day_of_week_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate heatmap data for demand visualization (legacy grid)."""
        
        # Build the query with filters
        query_filters = [
            "t.start_time BETWEEN :start_time AND :end_time"
        ]
        query_params = {
            'start_time': start_time,
            'end_time': end_time,
            'grid_size': grid_size_meters / 111000.0  # Convert meters to degrees (approximate)
        }
        
        if hour_filter is not None:
            query_filters.append("EXTRACT(hour FROM t.start_time) = :hour_filter")
            query_params['hour_filter'] = hour_filter
            
        if day_of_week_filter is not None:
            query_filters.append("EXTRACT(dow FROM t.start_time) = :day_filter")
            query_params['day_filter'] = day_of_week_filter
        
        # Generate heatmap grid using PostGIS
        heatmap_query = text(f"""
            WITH grid AS (
                SELECT 
                    ST_SnapToGrid(g.point, :grid_size) as grid_point,
                    COUNT(*) as trip_count,
                    AVG(t.duration_seconds) as avg_duration,
                    AVG(g.speed_kmh) as avg_speed,
                    ST_X(ST_SnapToGrid(g.point, :grid_size)) as longitude,
                    ST_Y(ST_SnapToGrid(g.point, :grid_size)) as latitude
                FROM geotracks g
                JOIN trips t ON g.trip_id = t.id
                WHERE {' AND '.join(query_filters)}
                GROUP BY ST_SnapToGrid(g.point, :grid_size)
                HAVING COUNT(*) >= 3  -- Minimum points for significance
            )
            SELECT 
                longitude,
                latitude,
                trip_count,
                avg_duration,
                avg_speed,
                CASE 
                    WHEN trip_count >= 50 THEN 1.0
                    WHEN trip_count >= 20 THEN 0.8
                    WHEN trip_count >= 10 THEN 0.6
                    WHEN trip_count >= 5 THEN 0.4
                    ELSE 0.2
                END as demand_intensity
            FROM grid
            ORDER BY trip_count DESC
            LIMIT 1000
        """)
        
        result = db.execute(heatmap_query, query_params)
        
        # Convert to heatmap cells
        cells = []
        total_trips = 0
        
        for row in result:
            cell = HeatmapCell(
                grid_id=f"{row.latitude:.6f}_{row.longitude:.6f}",
                trip_count=row.trip_count,
                avg_duration=row.avg_duration,
                avg_speed=row.avg_speed,
                demand_intensity=row.demand_intensity,
                latitude=row.latitude,
                longitude=row.longitude
            )
            cells.append(cell)
            total_trips += row.trip_count
        
        return {
            "cells": cells,
            "total_trips": total_trips
        }

    def identify_demand_zones_dbscan(
        self,
        db: Session,
        bbox: Optional[List[float]] = None,
        eps_m: int = 250,
        min_samples: int = 10,
        k_min: int = K_ANON_DEFAULT,
    ) -> Dict[str, Any]:
        """
        Identify demand zones using DBSCAN clustering with eps in meters.
        Returns clusters with center, radius_p95_m, count, density_km2, demand_score.
        """
        from sklearn.cluster import DBSCAN
        t0 = time.time()

        q = db.query(GeoTrack.latitude, GeoTrack.longitude)
        if bbox:
            min_lng, min_lat, max_lng, max_lat = bbox
            q = q.filter(
                GeoTrack.latitude >= min_lat,
                GeoTrack.latitude <= max_lat,
                GeoTrack.longitude >= min_lng,
                GeoTrack.longitude <= max_lng,
            )
        pts = q.all()
        if not pts:
            return {"clusters": [], "suppressed_below_k": k_min, "meta": build_meta({"bbox": bbox, "eps_m": eps_m, "min_samples": min_samples}, k_anon=k_min, suppressed=0, timings_ms={"compute": int((time.time()-t0)*1000)})}

        # Project to WebMercator (approx meters)
        R = 6378137.0
        xs, ys = [], []
        for lat, lng in pts:
            x = R * np.radians(lng)
            y = R * np.log(np.tan(np.pi/4 + np.radians(lat)/2))
            xs.append(x); ys.append(y)
        X = np.column_stack([xs, ys])

        dbs = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit(X)
        labels = dbs.labels_
        unique_labels = [l for l in set(labels) if l != -1]

        clusters: List[Dict[str, Any]] = []
        counts = []
        for cid, l in enumerate(unique_labels, start=1):
            mask = labels == l
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # Center in meters
            cx = float(np.mean(X[idx,0]))
            cy = float(np.mean(X[idx,1]))
            # Convert center back to lat/lng
            center_lng = np.degrees(cx / R)
            center_lat = np.degrees(2*np.arctan(np.exp(cy / R)) - np.pi/2)
            # Radius p95 in meters
            dists = np.sqrt((X[idx,0] - cx)**2 + (X[idx,1] - cy)**2)
            radius_p95_m = float(np.percentile(dists, 95))
            count = int(idx.size)
            area_km2 = float(np.pi * (max(radius_p95_m, 1.0)/1000.0)**2)
            density_km2 = count / area_km2 if area_km2 > 0 else float('inf')
            clusters.append({
                "cluster_id": cid,
                "center": {"lat": center_lat, "lng": center_lng},
                "radius_p95_m": radius_p95_m,
                "count": count,
                "density_km2": density_km2,
                "demand_score": 0.0,  # set after min-max
            })
            counts.append(count)

        # k-anon filter
        suppressed_before = len(clusters)
        clusters = [c for c in clusters if c["count"] >= k_min]
        suppressed = suppressed_before - len(clusters)

        # demand_score min-max by count
        if clusters:
            min_c = min(c["count"] for c in clusters)
            max_c = max(c["count"] for c in clusters)
            rng = max(max_c - min_c, 1)
            for c in clusters:
                c["demand_score"] = (c["count"] - min_c) / rng

        clusters.sort(key=lambda c: c["count"], reverse=True)
        t1 = time.time()
        return {
            "clusters": clusters,
            "suppressed_below_k": k_min,
            "meta": build_meta(
                query_params={"bbox": bbox, "eps_m": eps_m, "min_samples": min_samples},
                k_anon=k_min,
                suppressed=suppressed,
                timings_ms={"compute": int((t1 - t0)*1000)},
            ),
        }

    def analyze_trip_patterns(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        group_by: str = "hour"
    ) -> List[Dict[str, Any]]:
        """Analyze trip patterns over time"""
        
        # Define grouping SQL based on group_by parameter
        if group_by == "hour":
            group_sql = "EXTRACT(hour FROM start_time)"
            label_sql = "EXTRACT(hour FROM start_time) || ':00'"
        elif group_by == "day":
            group_sql = "DATE(start_time)"
            label_sql = "TO_CHAR(start_time, 'YYYY-MM-DD')"
        elif group_by == "week":
            group_sql = "DATE_TRUNC('week', start_time)"
            label_sql = "TO_CHAR(DATE_TRUNC('week', start_time), 'YYYY-MM-DD')"
        else:
            raise ValueError("group_by must be 'hour', 'day', or 'week'")
        
        pattern_query = text(f"""
            SELECT 
                {group_sql} as time_group,
                {label_sql} as time_label,
                COUNT(*) as trip_count,
                AVG(duration_seconds) as avg_duration,
                AVG(distance_meters) as avg_distance,
                STDDEV(duration_seconds) as duration_stddev,
                MIN(duration_seconds) as min_duration,
                MAX(duration_seconds) as max_duration,
                COUNT(DISTINCT vehicle_type) as vehicle_types,
                MODE() WITHIN GROUP (ORDER BY vehicle_type) as most_common_vehicle
            FROM trips
            WHERE start_time BETWEEN :start_time AND :end_time
            AND duration_seconds IS NOT NULL
            AND distance_meters IS NOT NULL
            GROUP BY {group_sql}
            ORDER BY time_group
        """)
        
        result = db.execute(pattern_query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        patterns = []
        for row in result:
            patterns.append({
                "time_group": row.time_group,
                "time_label": row.time_label,
                "trip_count": row.trip_count,
                "avg_duration_minutes": (row.avg_duration or 0) / 60.0,
                "avg_distance_km": (row.avg_distance or 0) / 1000.0,
                "duration_stddev_minutes": (row.duration_stddev or 0) / 60.0,
                "min_duration_minutes": (row.min_duration or 0) / 60.0,
                "max_duration_minutes": (row.max_duration or 0) / 60.0,
                "vehicle_types": row.vehicle_types,
                "most_common_vehicle": row.most_common_vehicle
            })
        
        return patterns
    
    def identify_demand_zones(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        zone_radius_km: float = 0.5,
        min_trips: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify high-demand geographic zones"""
        
        radius_degrees = zone_radius_km / 111.0  # Approximate conversion
        
        zones_query = text("""
            WITH trip_points AS (
                SELECT 
                    ST_X(start_point) as start_lon,
                    ST_Y(start_point) as start_lat,
                    ST_X(end_point) as end_lon,
                    ST_Y(end_point) as end_lat,
                    id as trip_id
                FROM trips
                WHERE start_time BETWEEN :start_time AND :end_time
                AND start_point IS NOT NULL
                AND end_point IS NOT NULL
            ),
            origin_clusters AS (
                SELECT 
                    ST_ClusterKMeans(ST_MakePoint(start_lon, start_lat), 20) OVER() as cluster_id,
                    start_lon, start_lat, trip_id
                FROM trip_points
            ),
            destination_clusters AS (
                SELECT 
                    ST_ClusterKMeans(ST_MakePoint(end_lon, end_lat), 20) OVER() as cluster_id,
                    end_lon, end_lat, trip_id
                FROM trip_points
            ),
            origin_zones AS (
                SELECT 
                    cluster_id,
                    AVG(start_lat) as center_lat,
                    AVG(start_lon) as center_lon,
                    COUNT(*) as trip_count,
                    'origin' as zone_type
                FROM origin_clusters
                GROUP BY cluster_id
                HAVING COUNT(*) >= :min_trips
            ),
            destination_zones AS (
                SELECT 
                    cluster_id,
                    AVG(end_lat) as center_lat,
                    AVG(end_lon) as center_lon,
                    COUNT(*) as trip_count,
                    'destination' as zone_type
                FROM destination_clusters
                GROUP BY cluster_id
                HAVING COUNT(*) >= :min_trips
            )
            SELECT * FROM origin_zones
            UNION ALL
            SELECT * FROM destination_zones
            ORDER BY trip_count DESC
            LIMIT 50
        """)
        
        result = db.execute(zones_query, {
            'start_time': start_time,
            'end_time': end_time,
            'min_trips': min_trips
        })
        
        zones = []
        for row in result:
            zones.append({
                "zone_id": f"{row.zone_type}_{row.cluster_id}",
                "center_latitude": row.center_lat,
                "center_longitude": row.center_lon,
                "trip_count": row.trip_count,
                "zone_type": row.zone_type,
                "demand_score": min(row.trip_count / 100.0, 1.0),  # Normalize to 0-1
                "radius_km": zone_radius_km
            })
        
        return zones
    
    def analyze_safety_patterns(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        area_lat: Optional[float] = None,
        area_lon: Optional[float] = None,
        radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze safety-related patterns"""
        
        # Base query filters
        query_filters = ["t.start_time BETWEEN :start_time AND :end_time"]
        query_params = {
            'start_time': start_time,
            'end_time': end_time
        }
        
        # Add area filter if specified
        if area_lat and area_lon and radius_km:
            query_filters.append("""
                EXISTS (
                    SELECT 1 FROM geotracks g 
                    WHERE g.trip_id = t.id 
                    AND ST_DWithin(g.point::geography, ST_MakePoint(:area_lon, :area_lat)::geography, :radius_meters)
                )
            """)
            query_params.update({
                'area_lat': area_lat,
                'area_lon': area_lon,
                'radius_meters': radius_km * 1000
            })
        
        # Analyze various safety metrics
        safety_query = text(f"""
            WITH trip_metrics AS (
                SELECT 
                    t.id,
                    t.duration_seconds,
                    t.distance_meters,
                    CASE WHEN t.distance_meters > 0 THEN t.duration_seconds::float / t.distance_meters * 3.6 ELSE NULL END as avg_speed_kmh,
                    (
                        SELECT MAX(g.speed_kmh) 
                        FROM geotracks g 
                        WHERE g.trip_id = t.id AND g.speed_kmh IS NOT NULL
                    ) as max_speed,
                    (
                        SELECT COUNT(*) 
                        FROM geotracks g 
                        WHERE g.trip_id = t.id AND g.speed_kmh > 80  -- High speed threshold
                    ) as high_speed_points,
                    (
                        SELECT COUNT(*) 
                        FROM geotracks g 
                        WHERE g.trip_id = t.id AND g.is_stop_point = true
                    ) as stop_count
                FROM trips t
                WHERE {' AND '.join(query_filters)}
                AND t.duration_seconds IS NOT NULL
                AND t.distance_meters IS NOT NULL
            )
            SELECT 
                COUNT(*) as total_trips,
                AVG(avg_speed_kmh) as overall_avg_speed,
                STDDEV(avg_speed_kmh) as speed_variance,
                COUNT(CASE WHEN max_speed > 100 THEN 1 END) as excessive_speed_trips,
                COUNT(CASE WHEN avg_speed_kmh < 5 THEN 1 END) as very_slow_trips,
                AVG(high_speed_points) as avg_high_speed_points,
                AVG(stop_count) as avg_stops_per_trip,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY max_speed) as speed_95th_percentile,
                COUNT(CASE WHEN duration_seconds > 3600 THEN 1 END) as long_duration_trips
            FROM trip_metrics
        """)
        
        result = db.execute(safety_query, query_params).first()
        
        if not result:
            return {"error": "No data found for the specified criteria"}
        
        # Calculate safety scores and risk indicators
        safety_metrics = {
            "total_trips_analyzed": result.total_trips,
            "average_speed_kmh": round(result.overall_avg_speed or 0, 2),
            "speed_variance": round(result.speed_variance or 0, 2),
            "excessive_speed_incidents": {
                "count": result.excessive_speed_trips,
                "percentage": round((result.excessive_speed_trips / result.total_trips) * 100, 2) if result.total_trips > 0 else 0
            },
            "very_slow_trips": {
                "count": result.very_slow_trips,
                "percentage": round((result.very_slow_trips / result.total_trips) * 100, 2) if result.total_trips > 0 else 0
            },
            "avg_high_speed_points_per_trip": round(result.avg_high_speed_points or 0, 2),
            "avg_stops_per_trip": round(result.avg_stops_per_trip or 0, 2),
            "speed_95th_percentile": round(result.speed_95th_percentile or 0, 2),
            "long_duration_trips": {
                "count": result.long_duration_trips,
                "percentage": round((result.long_duration_trips / result.total_trips) * 100, 2) if result.total_trips > 0 else 0
            }
        }
        
        # Calculate overall safety score (0-1, higher is safer)
        safety_score = 1.0
        if result.excessive_speed_trips > 0:
            safety_score -= (result.excessive_speed_trips / result.total_trips) * 0.3
        if result.very_slow_trips > result.total_trips * 0.1:  # More than 10% very slow trips
            safety_score -= 0.2
        if result.speed_variance and result.speed_variance > 20:  # High speed variance
            safety_score -= 0.2
        
        safety_metrics["overall_safety_score"] = max(0.0, round(safety_score, 3))
        
        return safety_metrics
    
    def calculate_efficiency_metrics(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        vehicle_type: Optional[str] = None,
        trip_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate various efficiency metrics"""
        
        # Build query filters
        query_filters = ["start_time BETWEEN :start_time AND :end_time"]
        query_params = {
            'start_time': start_time,
            'end_time': end_time
        }
        
        if vehicle_type:
            query_filters.append("vehicle_type = :vehicle_type")
            query_params['vehicle_type'] = vehicle_type
            
        if trip_type:
            query_filters.append("trip_type = :trip_type")
            query_params['trip_type'] = trip_type
        
        efficiency_query = text(f"""
            WITH trip_efficiency AS (
                SELECT 
                    id,
                    distance_meters,
                    duration_seconds,
                    ST_Distance(start_point::geography, end_point::geography) as direct_distance,
                    CASE 
                        WHEN duration_seconds > 0 AND distance_meters > 0 
                        THEN distance_meters::float / duration_seconds * 3.6 
                        ELSE NULL 
                    END as avg_speed_kmh,
                    CASE 
                        WHEN ST_Distance(start_point::geography, end_point::geography) > 0 
                        THEN distance_meters::float / ST_Distance(start_point::geography, end_point::geography)
                        ELSE NULL 
                    END as route_efficiency_ratio
                FROM trips
                WHERE {' AND '.join(query_filters)}
                AND start_point IS NOT NULL
                AND end_point IS NOT NULL
                AND distance_meters IS NOT NULL
                AND duration_seconds IS NOT NULL
                AND distance_meters > 100  -- Minimum trip distance
            )
            SELECT 
                COUNT(*) as total_trips,
                AVG(distance_meters) / 1000.0 as avg_distance_km,
                AVG(duration_seconds) / 60.0 as avg_duration_minutes,
                AVG(avg_speed_kmh) as avg_speed_kmh,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_speed_kmh) as p95_speed_kmh,
                AVG(route_efficiency_ratio) as avg_route_efficiency,
                STDDEV(route_efficiency_ratio) as route_efficiency_stddev,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY route_efficiency_ratio) as efficiency_25th,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY route_efficiency_ratio) as efficiency_75th,
                AVG(direct_distance) / 1000.0 as avg_straight_len_km,
                COUNT(CASE WHEN route_efficiency_ratio > 1.5 THEN 1 END) as inefficient_routes,
                COUNT(CASE WHEN avg_speed_kmh < 10 THEN 1 END) as very_slow_trips
            FROM trip_efficiency
        """)
        
        result = db.execute(efficiency_query, query_params).first()
        
        if not result or result.total_trips == 0:
            return {"error": "No trips found matching the criteria"}
        
        # Detect speed unit and compute p95 from raw speeds
        unit_info = detect_speed_unit_for_period(db, start_time, end_time)
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            "total_trips": int(result.total_trips or 0),
            "average_distance_km": round(result.avg_distance_km or 0, 2),
            "average_duration_minutes": round(result.avg_duration_minutes or 0, 2),
            "average_speed_kmh": round(result.avg_speed_kmh or 0, 2),
            "p95_speed_kmh": round((result.p95_speed_kmh or unit_info.get("p95_spd_kmh") or 0), 2),
            "avg_straight_len_km": round(result.avg_straight_len_km or 0, 3),
            "avg_detour_ratio": round(result.avg_route_efficiency or 0, 3),
            "route_efficiency": {
                "average_ratio": round(result.avg_route_efficiency or 0, 3),
                "standard_deviation": round(result.route_efficiency_stddev or 0, 3),
                "25th_percentile": round(result.efficiency_25th or 0, 3),
                "75th_percentile": round(result.efficiency_75th or 0, 3)
            },
            "inefficient_routes": {
                "count": int(result.inefficient_routes or 0),
                "percentage": round(((result.inefficient_routes or 0) / (result.total_trips or 1)) * 100, 2)
            },
            "very_slow_trips": {
                "count": int(result.very_slow_trips or 0),
                "percentage": round(((result.very_slow_trips or 0) / (result.total_trips or 1)) * 100, 2)
            },
            "speed_unit": "km/h",
            "unit_detection_method": unit_info.get("unit_detection_method", "auto-p95"),
        }
        
        # Overall efficiency score (0-1, higher is more efficient)
        base_efficiency = min(1.0, max(0.0, 2.0 - (result.avg_route_efficiency or 2.0)))
        speed_penalty = max(0.0, (20 - (result.avg_speed_kmh or 0)) / 20 * 0.3)
        efficiency_score = max(0.0, base_efficiency - speed_penalty)
        
        efficiency_metrics["overall_efficiency_score"] = round(efficiency_score, 3)
        
        return efficiency_metrics
    
    def comparative_analysis(
        self,
        db: Session,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare metrics between two time periods"""
        
        def get_period_metrics(start_time: datetime, end_time: datetime) -> Dict[str, float]:
            query = text("""
                SELECT 
                    COUNT(*) as trip_count,
                    AVG(distance_meters) / 1000.0 as avg_distance_km,
                    AVG(duration_seconds) / 60.0 as avg_duration_minutes,
                    AVG(CASE WHEN duration_seconds > 0 AND distance_meters > 0 
                        THEN distance_meters::float / duration_seconds * 3.6 
                        ELSE NULL END) as avg_speed_kmh,
                    COUNT(DISTINCT 
                        CASE WHEN start_point IS NOT NULL AND end_point IS NOT NULL
                        THEN ST_SnapToGrid(start_point, 0.01) || '|' || ST_SnapToGrid(end_point, 0.01)
                        END
                    ) as unique_routes
                FROM trips
                WHERE start_time BETWEEN :start_time AND :end_time
                AND distance_meters IS NOT NULL
                AND duration_seconds IS NOT NULL
            """)
            
            result = db.execute(query, {
                'start_time': start_time,
                'end_time': end_time
            }).first()
            
            return {
                'trip_count': result.trip_count or 0,
                'avg_distance': result.avg_distance_km or 0,
                'avg_duration': result.avg_duration_minutes or 0,
                'avg_speed': result.avg_speed_kmh or 0,
                'unique_routes': result.unique_routes or 0
            }
        
        # Get metrics for both periods
        period1_metrics = get_period_metrics(period1_start, period1_end)
        period2_metrics = get_period_metrics(period2_start, period2_end)
        
        # Calculate comparisons
        comparison = {}
        for metric in metrics:
            if metric in period1_metrics and metric in period2_metrics:
                p1_value = period1_metrics[metric]
                p2_value = period2_metrics[metric]
                
                if p1_value > 0:
                    change_percent = ((p2_value - p1_value) / p1_value) * 100
                else:
                    change_percent = 0 if p2_value == 0 else float('inf')
                
                comparison[metric] = {
                    "period1_value": round(p1_value, 2),
                    "period2_value": round(p2_value, 2),
                    "absolute_change": round(p2_value - p1_value, 2),
                    "percent_change": round(change_percent, 2),
                    "trend": "increase" if p2_value > p1_value else "decrease" if p2_value < p1_value else "stable"
                }
        
        return comparison


# Create instance
analytics_crud = AnalyticsCRUD()
