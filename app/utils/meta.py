from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional


def build_meta(query_params: Dict[str, Any], k_anon: int = 5, epsilon: Optional[float] = None,
               suppressed: int = 0, speed_unit: str = "km/h", unit_detection_method: str = "auto-p95",
               timings_ms: Optional[Dict[str, float]] = None, data_version: str = "trips_csv_v1") -> Dict[str, Any]:
    now_iso = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    return {
        "generated_at": now_iso,
        "data_version": data_version,
        "query": query_params,
        "privacy": {"k_anon": k_anon, "epsilon": epsilon, "suppressed": suppressed},
        "speed_unit": speed_unit,
        "unit_detection_method": unit_detection_method,
        "timings_ms": timings_ms or {},
    }
