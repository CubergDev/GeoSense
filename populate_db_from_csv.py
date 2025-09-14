"""
Populate the database with trips and geotracks from a CSV file.

This script is a thin CLI wrapper around app.utils.ingest_csv.ingest_csv_to_db.
It will:
- Ensure database tables exist
- Read the CSV in the format: randomized_id, lat, lng, alt, spd, azm
- Auto-detect speed units (km/h vs m/s) and convert to km/h
- Group rows by randomized_id into trips, preserving file order
- Synthesize timestamps at 1-second intervals (unless provided)
- Insert data into Trip and GeoTrack tables

Usage examples:
  python populate_db_from_csv.py
  python populate_db_from_csv.py --csv ./trips.csv --no-truncate
  python populate_db_from_csv.py --csv C:\\path\\to\\trips.csv --truncate
  python populate_db_from_csv.py --db-url postgresql://geo:geo123!@localhost:5432/geoai

Notes:
- By default, uses trips.csv located in the project root (next to this script) and truncates existing data.
- DATABASE_URL is read from environment or app.core.config.settings.
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Optional

from app.core.database import Base, engine, SessionLocal
from app.core.config import settings
from app.utils.ingest_csv import ingest_csv_to_db


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the database from a CSV file (trips and geotracks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Path to CSV with columns: randomized_id, lat, lng, alt, spd, azm (default: project_root/trips.csv)",
    )
    trunc_group = parser.add_mutually_exclusive_group()
    trunc_group.add_argument(
        "--truncate",
        dest="truncate",
        action="store_true",
        help="Truncate existing trips/geotracks before ingestion",
    )
    trunc_group.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="Do not truncate existing data before ingestion",
    )
    parser.set_defaults(truncate=True)

    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="Override DATABASE_URL for this run (also sets ENV for child imports)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Optionally override DB URL via env
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url
        # Re-import settings if needed; settings is a BaseSettings instance reading env at init.
        # In practice, engine is already created from app.core.database at import time.
        # For a one-off script, we proceed with the already-created engine.

    # Ensure tables exist (safe to call repeatedly)
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[ERROR] Failed to ensure tables exist: {e}", file=sys.stderr)
        return 2

    # Resolve CSV path: default to project root trips.csv if not provided
    csv_path = args.csv_path or os.path.join(os.path.dirname(__file__), "trips.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 3

    db = SessionLocal()
    try:
        print("🚀 Ingesting CSV into database...")
        print(f" - Database URL: {os.getenv('DATABASE_URL', settings.DATABASE_URL)}")
        print(f" - CSV path: {csv_path}")
        print(f" - Truncate existing data: {args.truncate}")

        summary = ingest_csv_to_db(db=db, csv_path=csv_path, truncate=args.truncate)

        print("\n✅ Ingestion completed successfully!")
        print("Summary:")
        print(f" - Rows in CSV:        {summary.get('rows_in_csv')}")
        print(f" - Trips created:      {summary.get('trips_created')}")
        print(f" - Points created:     {summary.get('points_created')}")
        print(f" - Speed unit detected:{' '}{summary.get('speed_unit_detected')} (p95={summary.get('p95_spd_raw')})")
        print(f" - Source file:        {summary.get('csv_path')}")
        return 0
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
