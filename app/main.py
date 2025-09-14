from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
import uvicorn
import os
from pathlib import Path

from app.core.config import settings
from app.core.database import get_db, engine, Base
from app.api.v1.api import api_router

# create tables; don't crash if db is down
try:
    Base.metadata.create_all(bind=engine)
except Exception as _e:
    # try later when db is up (health endpoint reports)
    pass

# create app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="""
    GeoAI Backend API for anonymized geospatial trip analytics.
    
    ## Features
    
    * **Trip Management**: Full CRUD operations for trips and geotracks
    * **Route Analytics**: Identify popular routes and bottlenecks
    * **Heatmap Generation**: Visualize demand patterns across areas
    * **ML Predictions**: Demand forecasting and anomaly detection
    * **Privacy-First**: All data is anonymized to protect user privacy
    
    ## Use Cases
    
    * Identify popular routes and traffic bottlenecks
    * Build heat maps of transportation demand
    * Optimize driver/vehicle distribution
    * Detect unusual routes and safety scenarios
    
    ## Data Privacy
    
    All geospatial data is anonymized using spatial-temporal blurring techniques.
    No personal information is stored or processed.
    """,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static frontend at /ui and /frontend if present
try:
    BASE_DIR = Path(__file__).resolve().parent.parent  # .../app
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
    if FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
        app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
except Exception:
    pass


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GeoAI Backend API",
        "version": settings.PROJECT_VERSION,
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_STR,
        "features": [
            "Trip and GeoTrack management",
            "Route analytics and identification",
            "Demand heatmap generation",
            "ML-powered predictions",
            "Privacy-preserving anonymization"
        ]
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with data source details"""
    csv_path = settings.DATA_CSV_PATH
    csv_present = os.path.exists(csv_path) if csv_path else False
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "data_source": "database_via_ingest",
            "csv_ingest_enabled": settings.DEV_ENABLE_INGEST,
            "csv_path": csv_path,
            "csv_present": csv_present,
            "ml_pipeline": "enabled" if settings.ENABLE_ML_PIPELINE else "disabled"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "data_source": "database_via_ingest",
                "csv_ingest_enabled": settings.DEV_ENABLE_INGEST,
                "csv_path": csv_path,
                "csv_present": csv_present,
                "error": str(e)
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
