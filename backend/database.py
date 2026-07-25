"""SQLite storage for hazard reports."""
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Optional

# NOTE: default DB location is the OS temp dir, NOT the project folder.
# If the project lives inside a cloud-synced folder (OneDrive/Dropbox/Google
# Drive), SQLite's file locking can throw "disk I/O error" there. Override
# with the DB_PATH env var to persist reports somewhere specific, e.g.:
#   DB_PATH=C:\path\to\reports.db uvicorn main:app
DB_PATH = Path(os.environ.get("DB_PATH", Path(tempfile.gettempdir()) / "urban_hazard_reports.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    defect_area_ratio REAL NOT NULL,
    nearby_report_count INTEGER NOT NULL,
    pothole_count INTEGER NOT NULL DEFAULT 1,
    severity_score REAL NOT NULL,
    is_hotspot INTEGER NOT NULL DEFAULT 0,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    timestamp REAL NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    # timeout: how long to wait on a locked db before raising "database is
    # locked" — the default (5s) was too tight once Live Scan mode + the 4s
    # polling loop started overlapping requests, which is the likely cause
    # of intermittent 500s. WAL mode lets readers (GET /reports, GET
    # /priority-list) run concurrently with a writer (POST /report) instead
    # of blocking each other.
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False):
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
    conn = get_conn()
    conn.execute(SCHEMA)
    # Migration: earlier runs (before multi-pothole-per-photo support) created
    # this table without pothole_count. CREATE TABLE IF NOT EXISTS won't add
    # it to an already-existing file, so add it defensively here.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(reports)").fetchall()}
    if "pothole_count" not in existing_cols:
        conn.execute("ALTER TABLE reports ADD COLUMN pothole_count INTEGER NOT NULL DEFAULT 1")
    conn.commit()
    conn.close()


def insert_report(
    type_: str,
    confidence: float,
    defect_area_ratio: float,
    nearby_report_count: int,
    severity_score: float,
    is_hotspot: bool,
    lat: float,
    lng: float,
    pothole_count: int = 1,
    timestamp: Optional[float] = None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO reports
           (type, confidence, defect_area_ratio, nearby_report_count,
            pothole_count, severity_score, is_hotspot, lat, lng, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            type_,
            confidence,
            defect_area_ratio,
            nearby_report_count,
            pothole_count,
            severity_score,
            int(is_hotspot),
            lat,
            lng,
            timestamp or time.time(),
        ),
    )
    conn.commit()
    report_id = cur.lastrowid
    conn.close()
    return report_id


def get_all_reports(type_: Optional[str] = None) -> list[dict]:
    conn = get_conn()
    if type_:
        rows = conn.execute(
            "SELECT * FROM reports WHERE type = ? ORDER BY timestamp DESC", (type_,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM reports ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_reports(ids: list[int]):
    """Bulk-delete reports by id — used by the auto-expiry sweep."""
    if not ids:
        return
    conn = get_conn()
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM reports WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()
