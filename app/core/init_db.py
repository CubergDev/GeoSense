"""
Database initialization and setup utilities
"""
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import engine, Base, SessionLocal
from app.models.geotrack import Trip, GeoTrack


def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def create_spatial_indexes():
    """Create spatial indexes for better geospatial query performance"""
    print("Creating spatial indexes...")
    
    db = SessionLocal()
    try:
        # Create spatial indexes for geotracks
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_geotracks_point 
            ON geotracks USING GIST (point);
        """))
        
        # Create spatial indexes for trips
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trips_start_point 
            ON trips USING GIST (start_point);
        """))
        
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trips_end_point 
            ON trips USING GIST (end_point);
        """))
        
        
        # Create time-based indexes
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trips_start_time 
            ON trips (start_time);
        """))
        
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_geotracks_timestamp 
            ON geotracks (timestamp);
        """))
        
        # Create composite indexes for common queries
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trips_time_vehicle 
            ON trips (start_time, vehicle_type);
        """))
        
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_geotracks_trip_sequence 
            ON geotracks (trip_id, sequence_order);
        """))
        
        db.commit()
        print("✓ Spatial indexes created successfully")
        
    except Exception as e:
        print(f"Error creating spatial indexes: {e}")
        db.rollback()
    finally:
        db.close()


def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    db = SessionLocal()
    try:
        # Check if sample data already exists
        existing_trips = db.query(Trip).count()
        if existing_trips > 0:
            print(f"Sample data already exists ({existing_trips} trips found)")
            return
        
        from datetime import datetime, timedelta
        import random
        
        # Create sample trips with geotracks
        sample_locations = [
            # New York City area
            (40.7128, -74.0060),  # Manhattan
            (40.6892, -74.0445),  # Statue of Liberty
            (40.7589, -73.9851),  # Times Square
            (40.7505, -73.9934),  # Empire State Building
            (40.7614, -73.9776),  # Central Park
            # San Francisco area
            (37.7749, -122.4194),  # Downtown SF
            (37.8199, -122.4783),  # Golden Gate Bridge
            (37.8024, -122.4058),  # Alcatraz
            (37.7849, -122.4094),  # Union Square
        ]
        
        vehicle_types = ['car', 'taxi', 'bike', 'scooter']
        trip_types = ['pickup', 'delivery', 'commute', 'leisure']
        
        for i in range(50):  # Create 50 sample trips
            # Random start and end locations
            start_loc = random.choice(sample_locations)
            end_loc = random.choice(sample_locations)
            
            while end_loc == start_loc:
                end_loc = random.choice(sample_locations)
            
            # Random trip timing
            start_time = datetime.now() - timedelta(
                days=random.randint(1, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Create trip
            trip = Trip(
                start_time=start_time,
                end_time=start_time + timedelta(minutes=random.randint(10, 120)),
                trip_type=random.choice(trip_types),
                vehicle_type=random.choice(vehicle_types),
                start_point=f"POINT({start_loc[1]} {start_loc[0]})",
                end_point=f"POINT({end_loc[1]} {end_loc[0]})",
                is_anonymized=True,
                anonymization_method="spatial_temporal_blur"
            )
            
            db.add(trip)
            db.flush()  # Get trip ID
            
            # Create sample geotracks for this trip
            num_points = random.randint(5, 20)
            lat_step = (end_loc[0] - start_loc[0]) / num_points
            lon_step = (end_loc[1] - start_loc[1]) / num_points
            
            for j in range(num_points):
                # Linear interpolation with some noise
                lat = start_loc[0] + (lat_step * j) + random.uniform(-0.001, 0.001)
                lon = start_loc[1] + (lon_step * j) + random.uniform(-0.001, 0.001)
                
                geotrack = GeoTrack(
                    trip_id=trip.id,
                    latitude=lat,
                    longitude=lon,
                    timestamp=start_time + timedelta(minutes=j * 2),
                    sequence_order=j,
                    speed_kmh=random.uniform(10, 60),
                    heading_degrees=random.uniform(0, 360),
                    point=f"POINT({lon} {lat})",
                    noise_applied=True,
                    spatial_blur_radius=50.0
                )
                
                db.add(geotrack)
            
            # Calculate trip metrics
            from geopy.distance import geodesic
            distance = geodesic(start_loc, end_loc).meters
            duration = (trip.end_time - trip.start_time).total_seconds()
            
            trip.distance_meters = distance
            trip.duration_seconds = int(duration)
        
        db.commit()
        print(f"✓ Sample data created successfully (50 trips with geotracks)")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.rollback()
    finally:
        db.close()


def setup_postgis():
    """Enable PostGIS extension if using PostgreSQL"""
    print("Setting up PostGIS...")
    
    db = SessionLocal()
    try:
        # Enable PostGIS extension
        db.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        db.commit()
        print("✓ PostGIS extension enabled")
        
    except Exception as e:
        print(f"PostGIS setup info: {e}")
        # This might fail on non-PostgreSQL databases, which is okay
    finally:
        db.close()


def init_database():
    """Initialize the complete database"""
    print("🚀 Initializing GeoAI database...")
    
    try:
        # Setup PostGIS if available
        setup_postgis()
        
        # Create tables
        create_tables()
        
        # Create spatial indexes
        create_spatial_indexes()
        
        print("✅ Database initialization completed successfully!")
        print("\nNext steps:")
        print("1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Ingest your CSV data via /api/v1/trips/ingest-csv or configure DATA_CSV_PATH")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    init_database()
