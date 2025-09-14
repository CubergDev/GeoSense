"""
Geospatial utility functions for GeoAI backend
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from geopy.distance import geodesic
import math


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing between two geographic points
    
    Args:
        lat1, lon1: Start point coordinates
        lat2, lon2: End point coordinates
    
    Returns:
        Bearing in degrees (0-360)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def create_bounding_box(center_lat: float, center_lon: float, radius_km: float) -> Dict[str, float]:
    """
    Create a bounding box around a center point
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius in kilometers
    
    Returns:
        Dictionary with min_lat, max_lat, min_lon, max_lon
    """
    # Rough conversion: 1 degree ≈ 111 km
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    
    return {
        'min_lat': center_lat - lat_offset,
        'max_lat': center_lat + lat_offset,
        'min_lon': center_lon - lon_offset,
        'max_lon': center_lon + lon_offset
    }


def point_in_polygon(point_lat: float, point_lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point_lat: Point latitude
        point_lon: Point longitude
        polygon: List of (lat, lon) tuples defining the polygon
    
    Returns:
        True if point is inside polygon
    """
    x, y = point_lon, point_lat
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def simplify_track(coordinates: List[Tuple[float, float]], tolerance_meters: float = 10.0) -> List[Tuple[float, float]]:
    """
    Simplify a GPS track using Douglas-Peucker algorithm
    
    Args:
        coordinates: List of (lat, lon) coordinate pairs
        tolerance_meters: Simplification tolerance in meters
    
    Returns:
        Simplified list of coordinates
    """
    if len(coordinates) <= 2:
        return coordinates
    
    def perpendicular_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line"""
        # Simplified calculation using geodesic distances
        line_length = geodesic(line_start, line_end).meters
        if line_length == 0:
            return geodesic(point, line_start).meters
        
        # Project point onto line and calculate distance
        # This is an approximation for geographic coordinates
        return min(
            geodesic(point, line_start).meters,
            geodesic(point, line_end).meters,
            line_length * 0.1  # Rough perpendicular distance estimate
        )
    
    def douglas_peucker(coords: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
        """Recursive Douglas-Peucker implementation"""
        if len(coords) <= 2:
            return coords
        
        # Find the point with maximum distance from the line
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(coords) - 1):
            distance = perpendicular_distance(coords[i], coords[0], coords[-1])
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursive call
            left_part = douglas_peucker(coords[:max_index + 1], tolerance)
            right_part = douglas_peucker(coords[max_index:], tolerance)
            
            # Combine results (remove duplicate point)
            return left_part[:-1] + right_part
        else:
            # All points are within tolerance, return endpoints
            return [coords[0], coords[-1]]
    
    return douglas_peucker(coordinates, tolerance_meters)


def calculate_route_similarity(route1: List[Tuple[float, float]], route2: List[Tuple[float, float]], 
                             sample_points: int = 10) -> float:
    """
    Calculate similarity between two routes using Hausdorff distance
    
    Args:
        route1: First route as list of (lat, lon) pairs
        route2: Second route as list of (lat, lon) pairs
        sample_points: Number of points to sample from each route
    
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if len(route1) < 2 or len(route2) < 2:
        return 0.0
    
    # Sample points from both routes
    def sample_route(route: List[Tuple[float, float]], n_points: int) -> List[Tuple[float, float]]:
        if len(route) <= n_points:
            return route
        
        indices = np.linspace(0, len(route) - 1, n_points, dtype=int)
        return [route[i] for i in indices]
    
    sampled_route1 = sample_route(route1, sample_points)
    sampled_route2 = sample_route(route2, sample_points)
    
    # Calculate directed Hausdorff distances
    def directed_hausdorff(set1: List[Tuple[float, float]], set2: List[Tuple[float, float]]) -> float:
        max_min_distance = 0
        for point1 in set1:
            min_distance = min(geodesic(point1, point2).meters for point2 in set2)
            max_min_distance = max(max_min_distance, min_distance)
        return max_min_distance
    
    hausdorff_1to2 = directed_hausdorff(sampled_route1, sampled_route2)
    hausdorff_2to1 = directed_hausdorff(sampled_route2, sampled_route1)
    hausdorff_distance = max(hausdorff_1to2, hausdorff_2to1)
    
    # Convert distance to similarity score (0-1)
    # Routes within 500m are considered very similar
    max_distance = 500  # meters
    similarity = max(0.0, 1.0 - (hausdorff_distance / max_distance))
    
    return min(1.0, similarity)


def create_grid_cells(bounds: Dict[str, float], cell_size_meters: int) -> List[Dict[str, Any]]:
    """
    Create a grid of cells within given bounds
    
    Args:
        bounds: Dictionary with min_lat, max_lat, min_lon, max_lon
        cell_size_meters: Size of each grid cell in meters
    
    Returns:
        List of grid cell dictionaries with center coordinates and bounds
    """
    # Convert cell size to degrees (approximate)
    lat_step = cell_size_meters / 111000.0  # meters to degrees latitude
    
    cells = []
    current_lat = bounds['min_lat']
    
    while current_lat < bounds['max_lat']:
        # Longitude step varies with latitude
        lon_step = cell_size_meters / (111000.0 * math.cos(math.radians(current_lat)))
        current_lon = bounds['min_lon']
        
        while current_lon < bounds['max_lon']:
            cell = {
                'center_lat': current_lat + lat_step / 2,
                'center_lon': current_lon + lon_step / 2,
                'min_lat': current_lat,
                'max_lat': current_lat + lat_step,
                'min_lon': current_lon,
                'max_lon': current_lon + lon_step,
                'cell_id': f"{current_lat:.6f}_{current_lon:.6f}"
            }
            cells.append(cell)
            current_lon += lon_step
        
        current_lat += lat_step
    
    return cells


def anonymize_coordinates(lat: float, lon: float, noise_radius_meters: float = 100.0) -> Tuple[float, float]:
    """
    Apply spatial noise to coordinates for anonymization
    
    Args:
        lat: Original latitude
        lon: Original longitude
        noise_radius_meters: Maximum noise radius in meters
    
    Returns:
        Anonymized (lat, lon) coordinates
    """
    # Generate random noise
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, noise_radius_meters)
    
    # Convert to coordinate offsets
    lat_offset = (radius * np.cos(angle)) / 111000.0  # meters to degrees
    lon_offset = (radius * np.sin(angle)) / (111000.0 * np.cos(np.radians(lat)))
    
    return lat + lat_offset, lon + lon_offset


def calculate_trip_metrics(coordinates: List[Tuple[float, float]], timestamps: Optional[List] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a trip
    
    Args:
        coordinates: List of (lat, lon) coordinate pairs
        timestamps: Optional list of timestamps
    
    Returns:
        Dictionary with trip metrics
    """
    if len(coordinates) < 2:
        return {}
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(1, len(coordinates)):
        distance = geodesic(coordinates[i-1], coordinates[i]).meters
        total_distance += distance
    
    # Calculate duration if timestamps provided
    duration_seconds = None
    if timestamps and len(timestamps) == len(coordinates):
        duration_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    
    # Calculate bearing changes (route complexity)
    bearing_changes = 0
    total_bearing_change = 0.0
    
    if len(coordinates) >= 3:
        for i in range(1, len(coordinates) - 1):
            bearing1 = calculate_bearing(coordinates[i-1][0], coordinates[i-1][1], 
                                       coordinates[i][0], coordinates[i][1])
            bearing2 = calculate_bearing(coordinates[i][0], coordinates[i][1],
                                       coordinates[i+1][0], coordinates[i+1][1])
            
            bearing_diff = abs(bearing2 - bearing1)
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff
            
            if bearing_diff > 30:  # Significant direction change
                bearing_changes += 1
                total_bearing_change += bearing_diff
    
    # Calculate route efficiency (straight line vs actual distance)
    straight_distance = geodesic(coordinates[0], coordinates[-1]).meters
    route_efficiency = total_distance / max(straight_distance, 1.0)
    
    metrics = {
        'total_distance_meters': total_distance,
        'straight_line_distance_meters': straight_distance,
        'route_efficiency_ratio': route_efficiency,
        'bearing_changes': bearing_changes,
        'avg_bearing_change_degrees': total_bearing_change / max(bearing_changes, 1),
        'start_coordinates': coordinates[0],
        'end_coordinates': coordinates[-1],
        'total_points': len(coordinates)
    }
    
    if duration_seconds:
        metrics.update({
            'duration_seconds': duration_seconds,
            'avg_speed_mps': total_distance / max(duration_seconds, 1),
            'avg_speed_kmh': (total_distance / max(duration_seconds, 1)) * 3.6
        })
    
    return metrics


def detect_stops(coordinates: List[Tuple[float, float]], timestamps: List, 
                min_duration_seconds: int = 60, max_radius_meters: float = 50.0) -> List[Dict[str, Any]]:
    """
    Detect stops in a GPS track
    
    Args:
        coordinates: List of (lat, lon) coordinate pairs
        timestamps: List of corresponding timestamps
        min_duration_seconds: Minimum duration to consider a stop
        max_radius_meters: Maximum movement radius to consider stationary
    
    Returns:
        List of detected stops with location and duration
    """
    if len(coordinates) != len(timestamps) or len(coordinates) < 3:
        return []
    
    stops = []
    i = 0
    
    while i < len(coordinates) - 1:
        # Look for potential stop starting at point i
        stop_start = i
        stop_center_lat = coordinates[i][0]
        stop_center_lon = coordinates[i][1]
        
        # Find end of potential stop
        j = i + 1
        while j < len(coordinates):
            distance = geodesic((stop_center_lat, stop_center_lon), coordinates[j]).meters
            if distance > max_radius_meters:
                break
            j += 1
        
        stop_end = j - 1
        
        # Check if this qualifies as a stop
        if stop_end > stop_start:
            duration = (timestamps[stop_end] - timestamps[stop_start]).total_seconds()
            
            if duration >= min_duration_seconds:
                # Calculate stop center (average of coordinates)
                avg_lat = np.mean([coord[0] for coord in coordinates[stop_start:stop_end+1]])
                avg_lon = np.mean([coord[1] for coord in coordinates[stop_start:stop_end+1]])
                
                stops.append({
                    'start_time': timestamps[stop_start],
                    'end_time': timestamps[stop_end],
                    'duration_seconds': duration,
                    'center_lat': avg_lat,
                    'center_lon': avg_lon,
                    'start_index': stop_start,
                    'end_index': stop_end
                })
        
        i = max(i + 1, stop_end + 1)  # Move past this stop
    
    return stops
