from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime


def detect_speed_unit_for_period(db: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """
    Auto-detect speed unit based on p95 of recorded speeds in GeoTrack.speed_kmh column.
    Returns dict with: {"speed_unit": str, "unit_detection_method": "auto-p95", "p95_spd_raw": float|None, "p95_spd_kmh": float|None}
    Rule per spec:
      - If 10<=p95<=40 -> assume m/s and convert to km/h for reports
      - If 40<=p95<=140 -> assume km/h
      - Else -> unknown (fallback to km/h)
    """
    q = text(
        """
        SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY g.speed_kmh) AS p95
        FROM geotracks g
        JOIN trips t ON g.trip_id = t.id
        WHERE t.start_time BETWEEN :start_time AND :end_time
          AND g.speed_kmh IS NOT NULL
        """
    )
    res = db.execute(q, {"start_time": start_time, "end_time": end_time}).first()
    p95 = float(res.p95) if res and res.p95 is not None else None

    speed_unit = "km/h"
    if p95 is None:
        return {"speed_unit": speed_unit, "unit_detection_method": "auto-p95", "p95_spd_raw": None, "p95_spd_kmh": None}

    if 10 <= p95 <= 40:
        speed_unit = "m/s"
        p95_kmh = p95 * 3.6
    elif 40 <= p95 <= 140:
        speed_unit = "km/h"
        p95_kmh = p95
    else:
        speed_unit = "km/h"
        p95_kmh = p95

    return {
        "speed_unit": speed_unit,
        "unit_detection_method": "auto-p95",
        "p95_spd_raw": p95,
        "p95_spd_kmh": float(p95_kmh) if p95_kmh is not None else None,
    }
