---
title: Urban Hazard Detector
emoji: 🚧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Urban Hazard Detector

"We don't just detect potholes — we tell you which one to fix first when you can only fix five this week."

Point a phone/webcam camera at a pothole → your `best.onnx` model (YOLOv8, single class `pothole`) detects it in the browser-submitted frame → a transparent severity formula scores it → it's geotagged and mapped live → when reports cluster, the app flags a hotspot and surfaces a ranked Priority Fix List.

## What's here

```
urban-hazard-detector/
├── backend/
│   ├── main.py          FastAPI app: POST /report, GET /reports, GET /priority-list
│   ├── inference.py     ONNX Runtime wrapper (preprocess, NMS, postprocess)
│   ├── severity.py      severity formula + haversine clustering + hotspot + priority list
│   ├── database.py      SQLite storage
│   ├── model/best.onnx  your trained pothole detector
│   └── requirements.txt
└── frontend/
    └── index.html       single-page app: camera capture, Leaflet map, priority table, hotspot toast
```

Your model (`best.onnx`) is a YOLOv8 export: input `images` is `[1,3,640,640]` float32, output `output0` is `[1,5,8400]` (box xywh + pothole confidence, already decoded/sigmoid'd — confirmed by inspecting the model directly). Only one trained class: `pothole`.

## Architecture choice (heads-up, differs slightly from the walkthrough doc)

The walkthrough suggested running ONNX Runtime **Web** (client-side, in-browser). This build instead runs inference **server-side** in FastAPI (`onnxruntime` Python), because decoding a raw YOLO output (NMS, box math) reliably in browser JS eats hackathon hours you don't have, and doing it once in Python is both simpler and easier to debug live. The frontend just captures a frame and POSTs it — same end-to-end behavior (capture → classify → severity → geotag → store → map), same team-plan roles, just less risk in the browser layer. If you want true on-device inference later, `inference.py` is the only file that needs a JS port.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` (or `http://<your-laptop-ip>:8000` from a phone on the same wifi — needed for phone camera access, which requires either `localhost` or HTTPS).

**Important — SQLite + OneDrive:** if this folder is synced (OneDrive/Dropbox/Drive), SQLite can throw `disk I/O error` writing the DB file. The backend defaults to storing `reports.db` in the OS temp folder for this reason. Override with `DB_PATH=/some/local/path/reports.db` if you want it to persist next to the code.

## API

- `POST /report` — multipart form: `image` (file), `lat`, `lng`. Runs detection; if a pothole clears the confidence threshold (0.35), computes severity, checks nearby reports (50m radius, 6h window) for clustering, stores it, and returns the full record (including `bbox_frac` for drawing the overlay and `is_hotspot`). Returns `{"detected": false}` if nothing found — no record is stored.
- `GET /reports` — all stored reports, for map pins.
- `GET /priority-list?top_n=5` — reports greedily clustered by proximity + type, ranked by combined severity (`avg_severity × cluster-size boost + recency boost`).
- `GET /demo-location` — fallback lat/lng for indoor demos.
- `POST /admin/login` — form field `password`. Returns a bearer token on success.
- `POST /admin/logout`, `DELETE /admin/reports`, `DELETE /admin/reports/{id}` — require `Authorization: Bearer <token>`.

## Severity formula (matches the walkthrough spec)

```
severity_score = defect_area_ratio * 0.4
               + confidence * 0.2
               + nearby_factor * 0.4
```

`defect_area_ratio` comes straight from the YOLO bounding box (fraction of frame). `nearby_factor` is `min(nearby_report_count, 5) / 5` — added this normalization so the raw formula (which is unbounded otherwise) stays in `[0, 1]`, matching the green/yellow/red tiers (`<0.34` / `<0.67` / `≥0.67`).

Hotspot rule: **≥3** reports of the same type within **50m** and the last **6 hours** (tunable constants at the top of `severity.py`: `CLUSTER_RADIUS_M`, `TIME_WINDOW_S`, `HOTSPOT_N`).

Multi-pothole photos boost severity (`in_frame_boost`, +15% per extra pothole in the same frame), and reports auto-expire after 3 days of no new activity within a tight 10m radius of their exact spot (`find_expired_report_ids` in `severity.py`).

## Frontend features

- Live camera (`getUserMedia`, rear camera preferred) with a "Capture & Report" button, plus an "Upload Photo" fallback for printed demo photos.
- "Live Scan" toggle — auto-captures every 3s (inference is ~0.5–1s/frame on CPU, so this stays responsive without hammering the backend).
- All detected potholes in a frame drawn as boxes on the snapshot, with label + confidence, in a popup modal.
- Geolocation with graceful fallback: real GPS → "Use Demo Location" button → click-the-map manual pin (all three needed for an indoor hackathon demo).
- Leaflet map, teardrop pins colored by severity tier, sized up for multi-pothole/nearby-cluster reports, auto-refreshes every 4s.
- Priority Fix List table (top 5 clusters), clickable rows that fly the map to that cluster.
- Toast banner when a new hotspot forms.
- Route planner (India-focused): destination autocomplete via Nominatim, route drawn via OSRM, on-route hazards flagged, optional live "Start Navigation" mode with a moving position marker and proximity alerts.
- Admin login (top-right of header) gates a "Clear All Potholes" button and a per-pin delete option.

## Team plan mapping (from the walkthrough)

- **Member 1 (Model & CV):** `best.onnx` — already done, dropped into `backend/model/`.
- **Member 2 (Backend & Logic):** `main.py`, `severity.py`, `database.py`.
- **Member 3 (Frontend & Map):** `frontend/index.html`.
- **Member 4 (Data/Integration/Demo):** use `/demo-location` + the manual map-pin button to pre-assign coordinates to your printed demo photos, then rehearse the sequence that triggers a hotspot on cue (3 same-type reports within 50m).

## Deployment

Docker-ready — see `Dockerfile` (Hugging Face Spaces, port 7860) and `render.yaml` (Render Blueprint, free tier). Both need `ADMIN_PASSWORD` set as a platform secret before going live — the code falls back to a default password otherwise. Free tiers on both platforms have ephemeral storage, so reports reset on sleep/restart unless you add persistent storage separately.

## Tested

Verified in a sandbox: model loads and runs inference (~970ms/frame on CPU), `/report` correctly returns `detected: false` on a hazard-free image, clustering/severity/hotspot logic produces expected results across a simulated 3-report cluster (hotspot triggers on the 3rd report, priority list correctly aggregates them into one ranked cluster), admin login/logout/delete-all/delete-one all round-trip correctly (including rejecting bad passwords and expired tokens), and the frontend is served correctly at `/`. Not yet tested: real pothole photos through the live camera (do this first before your demo — confidence threshold in `inference.py` (`CONF_THRES = 0.35`) may need tuning up/down based on your model's real-world confidence spread).
