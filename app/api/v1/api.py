from fastapi import APIRouter
from app.api.v1.endpoints import trips, analytics, routes, ml

api_router = APIRouter()

# Include trip-related endpoints
api_router.include_router(
    trips.router,
    prefix="/trips",
    tags=["trips"]
)

# Include analytics endpoints
api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)

# Include routes analytics/endpoints
api_router.include_router(
    routes.router,
    prefix="/routes",
    tags=["routes"]
)

# Include ML endpoints
api_router.include_router(
    ml.router,
    prefix="/ml",
    tags=["ml"]
)
