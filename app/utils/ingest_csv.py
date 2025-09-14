from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
from datetime import datetime, timedelta, timezone
import hashlib

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.geotrack import TripCreate, GeoTrackCreate
from app.crud.trip import trip_crud


VALID_COLS = ["randomized_id", "lat", "lng", "alt", "spd", "azm"]


def _hash_id(raw_id: int, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + str(int(raw_id))).encode("utf-8"))
    return h.hexdigest()


def _unit_from_p95(p95: float) -> str:
    if p95 is None:
        return "km/h"
    if 10 <= p95 <= 40:
        return "m/s"
    if 40 <= p95 <= 140:
        return "km/h"
    return "km/h"


def ingest_csv_to_db(db: Session, csv_path: Optional[str] = None, truncate: bool = True) -> Dict[str, Any]:
    """
    Ingest trips.csv into DB Trip/GeoTrack tables according to P0 spec (simplified).
    - Validates ranges, drops NaN/duplicates
    - Auto-detects speed unit by p95 and converts to km/h
    - Groups by randomized_id as one trip, orders by file order
    - Synthesizes timestamps (1s step) and computes basic trip metrics via TripCRUD
    Returns summary with counts and unit info.
    """
    csv_path = csv_path or settings.DATA_CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV (assume header present)
    df = pd.read_csv(csv_path)
    # Normalize columns
    cols_map = {c: c.strip() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)

    missing = [c for c in VALID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Basic validation and cleanup
    df = df.dropna(subset=["randomized_id", "lat", "lng"]).copy()
    # Range filters
    df = df[(df["lat"].between(-90, 90)) & (df["lng"].between(-180, 180))]
    if "spd" in df:
        # Keep all rows: coerce to numeric, fill NaN with 0, clamp negatives to 0
        df["spd"] = pd.to_numeric(df["spd"], errors="coerce").fillna(0)
        df.loc[df["spd"] < 0, "spd"] = 0
    if "azm" in df:
        # Keep rows: set invalid azimuths to None instead of dropping
        df["azm"] = pd.to_numeric(df["azm"], errors="coerce")
        mask_valid_azm = (df["azm"] >= 0) & (df["azm"] < 360)
        df.loc[~mask_valid_azm, "azm"] = None
    if "alt" in df:
        # Keep rows: coerce to numeric, null out extreme altitudes instead of dropping rows
        df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
        df.loc[df["alt"].abs() >= 10000, "alt"] = None

    # Speed unit detection
    p95 = df["spd"].quantile(0.95) if "spd" in df and not df["spd"].isna().all() else None
    unit = _unit_from_p95(p95 if p95 is not None else None)
    if unit == "m/s":
        df["speed_kmh"] = df["spd"].astype(float) * 3.6
    else:
        df["speed_kmh"] = df["spd"].astype(float)

    # Heading
    hd = pd.to_numeric(df["azm"], errors="coerce")
    df["heading_degrees"] = hd.where(~hd.isna(), None)

    # Sort by randomized_id and preserve original order
    if "__row__" not in df.columns:
        df["__row__"] = range(len(df))
    df.sort_values(["randomized_id", "__row__"], inplace=True)

    # Optionally truncate existing data
    if truncate:
        # Raw SQL for speed; SQLAlchemy ORM delete cascades might require configured FK
        db.execute("DELETE FROM geotracks")
        db.execute("DELETE FROM trips")
        db.commit()

    # Iterate groups and create trips
    base_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    trips_created = 0
    points_created = 0

    for rid, g in df.groupby("randomized_id", sort=False):
        start_ts = base_time
        # Build geotracks
        geotracks: List[GeoTrackCreate] = []
        for idx, row in enumerate(g.itertuples(index=False)):
            lat = float(getattr(row, "lat"))
            lng = float(getattr(row, "lng"))
            alt = float(getattr(row, "alt")) if hasattr(row, "alt") and getattr(row, "alt") is not None else None
            spd_kmh = float(getattr(row, "speed_kmh")) if hasattr(row, "speed_kmh") else None
            azm = float(getattr(row, "heading_degrees")) if hasattr(row, "heading_degrees") else None
            ts = start_ts + timedelta(seconds=idx)
            geotracks.append(
                GeoTrackCreate(
                    latitude=lat,
                    longitude=lng,
                    altitude=alt,
                    timestamp=ts,
                    sequence_order=idx,
                    speed_kmh=spd_kmh,
                    heading_degrees=azm,
                    is_stop_point=False,
                    dwell_time_seconds=None,
                )
            )
        trip_data = TripCreate(
            start_time=start_ts,
            end_time=start_ts + timedelta(seconds=len(geotracks) - 1),
            geotracks=geotracks,
        )
        trip_crud.create_trip(db=db, trip_data=trip_data)
        trips_created += 1
        points_created += len(geotracks)

    # Commit done by create_trip; ensure flushed
    db.commit()

    return {
        "csv_path": csv_path,
        "rows_in_csv": int(len(df)),
        "trips_created": int(trips_created),
        "points_created": int(points_created),
        "speed_unit_detected": unit,
        "p95_spd_raw": float(p95) if p95 is not None else None,
    }
