from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # General
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://geo:geo123!@localhost:5432/geoai"
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GeoAI Backend"
    PROJECT_VERSION: str = "1.0.0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Privacy / Ingestion
    HASH_SALT: str = "dev-salt-change"
    DEV_ENABLE_INGEST: bool = True
    DATA_CSV_PATH: str = "./trips.csv"
    CACHE_DIR: str = "./data/cache"
    CACHE_TTL_SECONDS: int = 300
    
    # Redis for caching and ML pipeline
    REDIS_URL: str = "redis://localhost:6379"
    
    # ML Configuration
    ML_MODEL_PATH: str = "./models"
    ENABLE_ML_PIPELINE: bool = True
    
    # Geospatial
    DEFAULT_SRID: int = 4326  # WGS84
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
