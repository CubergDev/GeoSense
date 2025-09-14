from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, text
from geoalchemy2.functions import ST_MakeLine, ST_DWithin, ST_Distance, ST_Length
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from shapely.geometry import LineString, Point

try:
    import h3
except Exception:
    h3 = None

try:
    import polyline as polyline_encoder
except Exception:
    polyline_encoder = None

from app.models.geotrack import Trip, GeoTrack
# Note: RouteCreate/RouteUpdate schemas are not part of MVP; using forward refs in annotations only
from app.utils.privacy import K_ANON_DEFAULT
from app.utils.meta import build_meta
from app.utils.cache import get_cached, set_cached


def _encode_polyline(coords_latlng: List[Tuple[float, float]]) -> str:
    """Encode list of (lat, lng) to polyline; fallback to simple string if lib missing."""
    if polyline_encoder:
        try:
            return polyline_encoder.encode(coords_latlng, precision=5)
        except Exception:
            pass
    # Fallback: semicolon-separated lat,lng pairs
    return ";".join(f"{lat:.5f},{lng:.5f}" for lat, lng in coords_latlng)


class RouteCRUD:
    def create_route(self, db: Session, route_data: RouteCreate) -> Route:
        """Create a new route"""
        db_route = Route(
            name=route_data.name,
            description=route_data.description,
            usage_count=0,
            incident_count=0
        )
        db.add(db_route)
        db.commit()
        db.refresh(db_route)
        return db_route
    
    def get_route(self, db: Session, route_id: int) -> Optional[Route]:
        """Get a route by ID"""
        return db.query(Route).filter(Route.id == route_id).first()
    
    def get_route_by_route_id(self, db: Session, route_id: str) -> Optional[Route]:
        """Get a route by route_id string"""
        return db.query(Route).filter(Route.route_id == route_id).first()
    
    def get_routes(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        min_usage: Optional[int] = None,
        congestion_level: Optional[str] = None
    ) -> List[Route]:
        """Get routes with optional filters"""
        query = db.query(Route)
        
        if min_usage is not None:
            query = query.filter(Route.usage_count >= min_usage)
        if congestion_level:
            query = query.filter(Route.congestion_level == congestion_level)
        
        return query.order_by(desc(Route.usage_count)).offset(skip).limit(limit).all()
    
    def update_route(self, db: Session, route_id: int, route_update: RouteUpdate) -> Optional[Route]:
        """Update a route"""
        db_route = self.get_route(db, route_id)
        if not db_route:
            return None
        
        update_data = route_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_route, field, value)
        
        db.commit()
        db.refresh(db_route)
        return db_route
    
    def delete_route(self, db: Session, route_id: int) -> bool:
        """Delete a route"""
        db_route = self.get_route(db, route_id)
        if not db_route:
            return False
        
        db.delete(db_route)
        db.commit()
        return True
    
    def identify_popular_routes(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        min_trips: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Route]:
        """Identify popular routes from trip data using clustering"""
        
        # Get all trips in the time period
        trips = db.query(Trip).filter(
            and_(
                Trip.start_time >= start_time,
                Trip.start_time <= end_time,
                Trip.distance_meters.isnot(None),
                Trip.distance_meters > 500  # Minimum 500m trips
            )
        ).all()
        
        if len(trips) < min_trips:
            return []
        
        # Group trips by similarity (simplified clustering)
        route_clusters = self._cluster_trips_by_similarity(db, trips, similarity_threshold)
        
        # Create routes for clusters with enough trips
        popular_routes = []
        for cluster in route_clusters:
            if len(cluster) >= min_trips:
                route = self._create_route_from_cluster(db, cluster)
                if route:
                    popular_routes.append(route)
        
        return popular_routes

    def analyze_popular_corridors(
        self,
        db: Session,
        bbox: List[float],
        top_n: int = 10,
        simplify_tolerance_m: int = 50,
        k_min: int = K_ANON_DEFAULT,
    ) -> Dict[str, Any]:
        """
        Build popular corridors by clustering simplified trajectories using start/end H3 cells,
        and angle/length buckets. Returns P0 structure. Uses simple file cache keyed by bbox/top_n/tolerance.
        """
        if h3 is None:
            raise RuntimeError("h3 library not installed")

        cache_params = {
            "bbox": bbox,
            "top_n": int(top_n),
            "simplify_tolerance_m": int(simplify_tolerance_m),
            "k_min": int(k_min),
        }
        cached, read_ms = get_cached("popular_corridors", cache_params)
        if cached:
            try:
                if isinstance(cached, dict) and "meta" in cached:
                    cached["meta"]["generated_at"] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
                    tm = cached["meta"].get("timings_ms") or {}
                    tm["read_cache"] = read_ms or 0
                    cached["meta"]["timings_ms"] = tm
            except Exception:
                pass
            return cached

        # Fetch geotracks in bbox ordered by trip and sequence
        min_lng, min_lat, max_lng, max_lat = bbox
        rows = (
            db.query(GeoTrack.trip_id, GeoTrack.latitude, GeoTrack.longitude, GeoTrack.sequence_order, GeoTrack.speed_kmh)
            .filter(
                GeoTrack.latitude >= min_lat,
                GeoTrack.latitude <= max_lat,
                GeoTrack.longitude >= min_lng,
                GeoTrack.longitude <= max_lng,
            )
            .order_by(GeoTrack.trip_id.asc(), GeoTrack.sequence_order.asc())
            .all()
        )
        if not rows:
            res_empty = {"corridors": [], "suppressed_below_k": k_min, "meta": build_meta({"bbox": bbox, "top_n": top_n, "simplify_tolerance_m": simplify_tolerance_m}, k_anon=k_min, suppressed=0)}
            try:
                set_cached("popular_corridors", cache_params, res_empty)
            except Exception:
                pass
            return res_empty

        # Group points per trip
        trips: Dict[int, List[Tuple[float, float, Optional[float]]]] = defaultdict(list)
        for trip_id, lat, lng, seq, spd in rows:
            trips[trip_id].append((lat, lng, spd))

        # Helper: WebMercator projection
        R = 6378137.0
        def to_m(lat: float, lng: float) -> Tuple[float, float]:
            x = R * np.radians(lng)
            y = R * np.log(np.tan(np.pi/4 + np.radians(lat)/2))
            return float(x), float(y)
        def to_ll(x: float, y: float) -> Tuple[float, float]:
            lng = np.degrees(x / R)
            lat = np.degrees(2*np.arctan(np.exp(y / R)) - np.pi/2)
            return float(lat), float(lng)

        clusters: Dict[str, Dict[str, Any]] = {}
        for trip_id, pts in trips.items():
            if len(pts) < 2:
                continue
            latlng = [(lat, lng) for lat, lng, _ in pts]
            # Simplify in meters
            coords_m = [to_m(lat, lng) for lat, lng in latlng]
            line = LineString(coords_m)
            simp = line.simplify(simplify_tolerance_m, preserve_topology=False) if simplify_tolerance_m > 0 else line
            if simp.length < 100:  # skip tiny
                continue
            # Cluster key: start/end H3 at level 9 + angle and length buckets
            start_lat, start_lng = to_ll(*list(simp.coords)[0])
            end_lat, end_lng = to_ll(*list(simp.coords)[-1])
            try:
                s_cell = h3.geo_to_h3(start_lat, start_lng, 9)
                e_cell = h3.geo_to_h3(end_lat, end_lng, 9)
            except Exception:
                continue
            dx = list(simp.coords)[-1][0] - list(simp.coords)[0][0]
            dy = list(simp.coords)[-1][1] - list(simp.coords)[0][1]
            ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
            ang_bucket = int(ang // 10) * 10
            length_m = float(simp.length)
            len_bucket = int(length_m // 100) * 100
            key = f"{s_cell}->{e_cell}|a{ang_bucket}|l{len_bucket}"
            if key not in clusters:
                clusters[key] = {"trips": 0, "lines": [], "speeds": [], "detours": []}
            clusters[key]["trips"] += 1
            clusters[key]["lines"].append(simp)
            # Gather speeds
            speeds = [s for _, _, s in pts if s is not None]
            if speeds:
                clusters[key]["speeds"].extend(speeds)
            # Detour ratio per trip (approx meters)
            path_len = float(line.length)
            straight_len = float(LineString([coords_m[0], coords_m[-1]]).length)
            detour = (path_len / straight_len) if straight_len > 50 else 1.0
            clusters[key]["detours"].append(detour)

        # Build corridors from clusters with k-anon
        items: List[Dict[str, Any]] = []
        for key, data in clusters.items():
            if data["trips"] < k_min:
                continue
            # representative line: longest
            rep = max(data["lines"], key=lambda l: l.length)
            # width p90 as distance of original line points to rep
            # Approximate with buffer width; alternatively compute sample distances
            # For simplicity, take p90 of distances between rep and all vertices of all lines
            dists = []
            for ln in data["lines"]:
                for x, y in ln.coords:
                    dists.append(LineString(rep).distance(Point(x, y)))
            width_p90_m = float(np.percentile(dists, 90)) if dists else 0.0
            # polyline encode representative line in lat/lng
            coords_latlng = [to_ll(x, y) for x, y in rep.coords]
            poly_str = _encode_polyline(coords_latlng)
            median_speed = float(np.median(data["speeds"])) if data["speeds"] else None
            median_detour = float(np.median(data["detours"])) if data["detours"] else None
            # Simple confidence proportional to trips vs max
            items.append({
                "key": key,
                "corridor_id": key,
                "polyline": poly_str,
                "trips": int(data["trips"]),
                "width_p90_m": width_p90_m,
                "median_speed_kmh": median_speed,
                "median_detour_ratio": median_detour,
            })

        if not items:
            res_empty2 = {"corridors": [], "suppressed_below_k": k_min, "meta": build_meta({"bbox": bbox, "top_n": top_n, "simplify_tolerance_m": simplify_tolerance_m}, k_anon=k_min, suppressed=0)}
            try:
                set_cached("popular_corridors", cache_params, res_empty2)
            except Exception:
                pass
            return res_empty2

        max_trips = max(it["trips"] for it in items)
        for it in items:
            it["confidence"] = (it["trips"] / max_trips) if max_trips else 0.0
        # Sort and take top_n
        items.sort(key=lambda x: x["trips"], reverse=True)
        items = items[:top_n]
        res = {
            "corridors": [
                {
                    "corridor_id": it["corridor_id"],
                    "polyline": it["polyline"],
                    "trips": it["trips"],
                    "width_p90_m": it["width_p90_m"],
                    "median_speed_kmh": it.get("median_speed_kmh"),
                    "median_detour_ratio": it.get("median_detour_ratio"),
                    "confidence": it["confidence"],
                }
                for it in items
            ],
            "suppressed_below_k": k_min,
            "meta": build_meta({"bbox": bbox, "top_n": top_n, "simplify_tolerance_m": simplify_tolerance_m}, k_anon=k_min, suppressed=0),
        }
        try:
            set_cached("popular_corridors", cache_params, res)
        except Exception:
            pass
        return res
    
    def find_bottlenecks(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        area_lat: float,
        area_lon: float,
        radius_km: float = 1.0
    ) -> List[dict]:
        """Find traffic bottlenecks in a specific area"""
        
        # Query for geotracks in the area with low speeds
        radius_meters = radius_km * 1000
        
        bottleneck_query = text("""
            SELECT 
                ST_X(ST_Centroid(ST_Collect(point))) as longitude,
                ST_Y(ST_Centroid(ST_Collect(point))) as latitude,
                AVG(speed_kmh) as avg_speed,
                COUNT(*) as point_count,
                COUNT(DISTINCT trip_id) as trip_count
            FROM geotracks g
            JOIN trips t ON g.trip_id = t.id
            WHERE ST_DWithin(
                point, 
                ST_MakePoint(:lon, :lat)::geography, 
                :radius
            )
            AND t.start_time BETWEEN :start_time AND :end_time
            AND g.speed_kmh IS NOT NULL
            AND g.speed_kmh < 15  -- Low speed threshold
            GROUP BY ST_SnapToGrid(point, 0.001)  -- Group by ~100m grid
            HAVING COUNT(*) >= 10  -- Minimum points for significance
            ORDER BY AVG(speed_kmh) ASC, COUNT(*) DESC
            LIMIT 20
        """)
        
        result = db.execute(bottleneck_query, {
            'lat': area_lat,
            'lon': area_lon,
            'radius': radius_meters,
            'start_time': start_time,
            'end_time': end_time
        })
        
        bottlenecks = []
        for row in result:
            severity_score = max(0, (20 - row.avg_speed) / 20.0)  # Higher score for lower speeds
            bottlenecks.append({
                'latitude': row.latitude,
                'longitude': row.longitude,
                'severity_score': severity_score,
                'avg_speed_kmh': row.avg_speed,
                'affected_trips': row.trip_count,
                'data_points': row.point_count
            })
        
        return bottlenecks
    
    def get_route_analytics(
        self,
        db: Session,
        route_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> List[RouteAnalytics]:
        """Get analytics for a specific route"""
        return db.query(RouteAnalytics).filter(
            and_(
                RouteAnalytics.route_id == route_id,
                RouteAnalytics.analysis_date >= start_time,
                RouteAnalytics.analysis_date <= end_time
            )
        ).order_by(RouteAnalytics.analysis_date).all()
    
    def calculate_route_demand(
        self,
        db: Session,
        route_id: int,
        analysis_date: datetime
    ) -> Optional[RouteAnalytics]:
        """Calculate demand analytics for a route on a specific date"""
        
        route = self.get_route(db, route_id)
        if not route:
            return None
        
        # Calculate analytics for the day
        day_start = analysis_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # Get trips for this route (simplified - would need proper route matching)
        trips_query = text("""
            SELECT 
                COUNT(*) as trip_count,
                AVG(duration_seconds) as avg_duration,
                STDDEV(duration_seconds) as duration_variance,
                AVG(distance_meters) as avg_distance
            FROM trips t
            WHERE t.start_time BETWEEN :start_time AND :end_time
            AND t.distance_meters IS NOT NULL
        """)
        
        result = db.execute(trips_query, {
            'start_time': day_start,
            'end_time': day_end
        }).first()
        
        if not result or result.trip_count == 0:
            return None
        
        # Create analytics record
        analytics = RouteAnalytics(
            route_id=route_id,
            analysis_date=analysis_date,
            trip_count=result.trip_count,
            avg_travel_time=result.avg_duration,
            travel_time_variance=result.duration_variance or 0,
            demand_density=result.trip_count / max(route.avg_distance_meters or 1000, 1000) * 1000,
            congestion_index=1.0  # Simplified - would calculate based on speed data
        )
        
        db.add(analytics)
        db.commit()
        db.refresh(analytics)
        return analytics
    
    def _cluster_trips_by_similarity(
        self,
        db: Session,
        trips: List[Trip],
        similarity_threshold: float
    ) -> List[List[Trip]]:
        """Simple clustering of trips by start/end point similarity"""
        
        clusters = []
        
        for trip in trips:
            if not trip.start_point or not trip.end_point:
                continue
            
            # Find matching cluster
            matched_cluster = None
            for cluster in clusters:
                if self._trips_are_similar(db, trip, cluster[0], similarity_threshold):
                    matched_cluster = cluster
                    break
            
            if matched_cluster:
                matched_cluster.append(trip)
            else:
                clusters.append([trip])
        
        return clusters
    
    def _trips_are_similar(
        self,
        db: Session,
        trip1: Trip,
        trip2: Trip,
        threshold: float
    ) -> bool:
        """Check if two trips are similar based on start/end points"""
        
        if not all([trip1.start_point, trip1.end_point, trip2.start_point, trip2.end_point]):
            return False
        
        # Calculate distances between start and end points
        start_distance_query = db.execute(text("""
            SELECT ST_Distance(:point1::geography, :point2::geography) as distance
        """), {
            'point1': str(trip1.start_point),
            'point2': str(trip2.start_point)
        }).scalar()
        
        end_distance_query = db.execute(text("""
            SELECT ST_Distance(:point1::geography, :point2::geography) as distance
        """), {
            'point1': str(trip1.end_point),
            'point2': str(trip2.end_point)
        }).scalar()
        
        # Consider trips similar if start and end points are within 500m
        max_distance = 500  # meters
        return (start_distance_query < max_distance and end_distance_query < max_distance)
    
    def _create_route_from_cluster(self, db: Session, trip_cluster: List[Trip]) -> Optional[Route]:
        """Create a route from a cluster of similar trips"""
        
        if not trip_cluster:
            return None
        
        # Calculate average metrics
        total_distance = sum(t.distance_meters or 0 for t in trip_cluster)
        total_duration = sum(t.duration_seconds or 0 for t in trip_cluster)
        avg_distance = total_distance / len(trip_cluster)
        avg_duration = total_duration / len(trip_cluster)
        avg_speed = (avg_distance / max(avg_duration, 1)) * 3.6  # m/s to km/h
        
        # Create route
        route = Route(
            name=f"Route_{len(trip_cluster)}_trips",
            description=f"Popular route identified from {len(trip_cluster)} similar trips",
            usage_count=len(trip_cluster),
            avg_duration_seconds=avg_duration,
            avg_distance_meters=avg_distance,
            avg_speed_kmh=avg_speed,
            demand_score=min(len(trip_cluster) / 100.0, 1.0),  # Normalize to 0-1
            congestion_level="low" if avg_speed > 30 else "medium" if avg_speed > 15 else "high"
        )
        
        db.add(route)
        db.commit()
        db.refresh(route)
        return route


# Create instance
route_crud = RouteCRUD()
