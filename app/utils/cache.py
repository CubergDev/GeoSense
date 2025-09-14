from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os
import json
import hashlib
import time
from datetime import datetime

from app.core.config import settings


def _ensure_cache_dir() -> str:
    cache_dir = settings.CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    return cache_dir


def _key_hash(params: Dict[str, Any]) -> str:
    try:
        # Stable JSON dump
        payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(params)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _path_for(kind: str, params: Dict[str, Any]) -> str:
    cache_dir = _ensure_cache_dir()
    key = _key_hash(params)
    fname = f"{kind}__{key}.json"
    return os.path.join(cache_dir, fname)


def get_cached(kind: str, params: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Try to read a fresh cached JSON payload for the given kind+params.
    Returns (data, read_time_ms). If cache is absent or stale, returns (None, read_time_ms).
    """
    t0 = time.time()
    ttl = int(ttl_seconds if ttl_seconds is not None else getattr(settings, "CACHE_TTL_SECONDS", 300))
    path = _path_for(kind, params)
    try:
        if not os.path.exists(path):
            return None, int((time.time() - t0) * 1000)
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        if age > ttl:
            return None, int((time.time() - t0) * 1000)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, int((time.time() - t0) * 1000)
    except Exception:
        return None, int((time.time() - t0) * 1000)


def set_cached(kind: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
    """
    Store JSON payload to cache.
    """
    path = _path_for(kind, params)
    try:
        # Ensure parent dir
        _ensure_cache_dir()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        # Best-effort: ignore caching errors
        pass
