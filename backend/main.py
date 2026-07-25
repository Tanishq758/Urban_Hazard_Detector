"""
Urban Hazard Detector - FastAPI backend.

Endpoints
  POST /report        multipart image + lat/lng -> runs detection, scores
                       severity, checks clustering, stores + returns the record
  GET  /reports        all stored reports (for map rendering)
  GET  /priority-list   top-N hotspot clusters ranked by combined severity
  GET  /health          liveness check (also reports model load status)

Serves the single-page frontend from /frontend at "/".
"""
import hmac
import logging
import os
import secrets
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from database import init_db, insert_report, get_all_reports, delete_reports
from inference import detector
from severity import (
    compute_severity,
    count_nearby,
    is_hotspot,
    build_priority_list,
    severity_tier,
    find_expired_report_ids,
    EXPIRY_DAYS,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Urban Hazard Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Default FastAPI behavior on an unhandled exception is a bare 500 with
    # no body detail, which is why the frontend could only ever show
    # "Request failed (HTTP 500)" — no way to tell what actually broke. This
    # logs the full traceback to the server console (check that terminal if
    # this fires) AND returns the message to the client so it's visible
    # without needing server access.
    logger.error("Unhandled error on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {exc.__class__.__name__}: {exc}"},
    )

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Demo fallback coordinates (used when the browser has no / denies geolocation,
# e.g. indoors at a hackathon venue). Jittered slightly per-request in the
# frontend so repeated indoor demo photos don't all land on the exact same pin.
# India-focused deployment: defaults to Connaught Place, New Delhi.
DEFAULT_DEMO_LAT = 28.6315
DEFAULT_DEMO_LNG = 77.2167


@app.on_event("startup")
def startup():
    init_db()


# --- admin auth ---
# Deliberately minimal for a hackathon build: one shared password (override it
# via the ADMIN_PASSWORD env var before a real demo — don't ship the default),
# and login tokens live only in memory, so every admin session is cleared on
# server restart. No rate-limiting on login attempts, no per-admin accounts.
# Good enough to gate a "wipe the map" button behind; not a real auth system.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
active_admin_tokens: set[str] = set()


def require_admin(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing admin token — log in first.")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in active_admin_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired admin session — log in again.")
    return token


def cleanup_expired_reports():
    """Lazy sweep: delete any report whose precise spot (EXPIRY_RADIUS_M,
    much tighter than the 50m hotspot-clustering radius) has had no new
    report in EXPIRY_DAYS. Runs at the top of every read/write endpoint
    below instead of on a background scheduler — simplest thing that works
    given the app is already being polled every few seconds by the frontend,
    so expired pins disappear within one polling cycle of crossing the
    threshold without needing extra infrastructure."""
    try:
        all_reports = get_all_reports()
        expired_ids = find_expired_report_ids(all_reports)
        if expired_ids:
            delete_reports(expired_ids)
            logger.info(
                "Auto-removed %d report(s) with no activity at their spot in %d day(s): ids=%s",
                len(expired_ids), EXPIRY_DAYS, expired_ids,
            )
    except Exception:
        # Never let the cleanup sweep itself break a request.
        logger.error("Expiry cleanup sweep failed:\n%s", traceback.format_exc())


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.session is not None}


MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous for a single camera frame / photo


@app.post("/report")
async def create_report(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lng: float = Form(...),
):
    # Reject non-image uploads (e.g. accidentally selected videos) before
    # reading the body — a multi-hundred-MB video read into memory here is
    # what previously made the server appear to hang / "go unreachable".
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected a photo, got '{image.content_type or 'an unknown file type'}'. "
                "Videos aren't supported — please upload a JPG/PNG or use the camera capture."
            ),
        )

    raw_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_UPLOAD_BYTES // (1024*1024)}MB). Try a smaller photo.",
        )

    try:
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image — file may be corrupted or in an unsupported format (HEIC isn't supported; use JPG/PNG).",
        )

    try:
        result = detector.detect(frame)
    except Exception as e:
        logger.error("Model inference failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    if not result["detected"]:
        return {"detected": False, "inference_ms": result["inference_ms"]}

    try:
        cleanup_expired_reports()
        existing = get_all_reports(type_=result["type"])
        nearby_count = count_nearby(existing, result["type"], lat, lng)
        # One report per photo (one map pin), but its severity is boosted
        # when the photo itself contains multiple potholes — see
        # compute_severity()'s in_frame_boost in severity.py.
        severity_score = compute_severity(
            result["defect_area_ratio"], result["confidence"], nearby_count, result["count"]
        )
        hotspot = is_hotspot(nearby_count)

        report_id = insert_report(
            type_=result["type"],
            confidence=result["confidence"],
            defect_area_ratio=result["defect_area_ratio"],
            nearby_report_count=nearby_count,
            pothole_count=result["count"],
            severity_score=severity_score,
            is_hotspot=hotspot,
            lat=lat,
            lng=lng,
        )
    except Exception as e:
        logger.error("Storing report failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Could not save report (database busy?): {e}")

    return {
        "detected": True,
        "id": report_id,
        "type": result["type"],
        "confidence": result["confidence"],
        "defect_area_ratio": result["defect_area_ratio"],
        "bbox_frac": result["bbox_frac"],
        "pothole_count": result["count"],
        "detections": result["detections"],
        "nearby_report_count": nearby_count,
        "severity_score": severity_score,
        "severity_tier": severity_tier(severity_score),
        "is_hotspot": hotspot,
        "lat": lat,
        "lng": lng,
        "inference_ms": result["inference_ms"],
    }


@app.get("/reports")
def list_reports(type: str | None = None):
    cleanup_expired_reports()
    reports = get_all_reports(type_=type)
    for r in reports:
        r["severity_tier"] = severity_tier(r["severity_score"])
    return {"count": len(reports), "reports": reports}


@app.get("/priority-list")
def priority_list(top_n: int = 5):
    cleanup_expired_reports()
    reports = get_all_reports()
    return {"priority_list": build_priority_list(reports, top_n=top_n)}


@app.get("/demo-location")
def demo_location():
    """Fallback coordinates for indoor demo mode."""
    return {"lat": DEFAULT_DEMO_LAT, "lng": DEFAULT_DEMO_LNG}


@app.post("/admin/login")
def admin_login(password: str = Form(...)):
    if not hmac.compare_digest(password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect admin password.")
    token = secrets.token_urlsafe(24)
    active_admin_tokens.add(token)
    return {"token": token}


@app.post("/admin/logout")
def admin_logout(token: str = Depends(require_admin)):
    active_admin_tokens.discard(token)
    return {"status": "logged out"}


@app.delete("/admin/reports")
def admin_clear_all(_: str = Depends(require_admin)):
    ids = [r["id"] for r in get_all_reports()]
    delete_reports(ids)
    logger.info("Admin cleared all %d report(s).", len(ids))
    return {"deleted": len(ids)}


@app.delete("/admin/reports/{report_id}")
def admin_delete_one(report_id: int, _: str = Depends(require_admin)):
    delete_reports([report_id])
    logger.info("Admin deleted report id=%s", report_id)
    return {"deleted": 1}


# --- static frontend (mounted last so /report, /reports etc. take priority) ---
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
