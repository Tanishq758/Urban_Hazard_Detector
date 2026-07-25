"""
Severity scoring + haversine clustering / hotspot / priority-list logic.

severity_score = [(defect_area_ratio * 0.4)
                 + (confidence * 0.2)
                 + (nearby_factor * 0.4)] * in_frame_boost

nearby_factor is nearby_report_count capped and normalized to [0,1] via
min(count, NEARBY_CAP) / NEARBY_CAP, so the base score stays in [0,1] and the
green/yellow/red tiers below are meaningful. NEARBY_CAP and the raw formula
weights match the hackathon walkthrough's spec.

in_frame_boost scales the score up when a *single photo* contains multiple
potholes (pothole_count > 1) — same "more of them = worse" multiplier used
for multi-report clusters in build_priority_list() below, just applied
within one frame instead of across separate reports. defect_area_ratio
itself is already the *total* damaged fraction across every box found in
the frame (see inference.py), so this boost stacks on top of that rather
than duplicating it — a photo with 5 small potholes scores higher than one
with a single pothole of the same total area, matching how a person would
actually judge "this stretch of road is worse."
"""
import time
from math import radians, sin, cos, sqrt, atan2

# --- tunables (documented here so they're easy to demo-tune live) ---
CLUSTER_RADIUS_M = 50          # R: "nearby" reports within this many meters
TIME_WINDOW_S = 6 * 3600       # T: only count reports from the last 6 hours
HOTSPOT_N = 3                  # N: >= N reports of same type in R/T => hotspot
NEARBY_CAP = 5                 # normalization cap for nearby_report_count

# Auto-expiry: a report disappears once its *precise* spot has gone quiet.
# EXPIRY_RADIUS_M is deliberately much tighter than CLUSTER_RADIUS_M — the
# 50m cluster radius is meant to lump distinct-but-nearby potholes into one
# "problem area" for prioritization, which is the opposite of what expiry
# needs: expiry should only look at reports of essentially the same pothole,
# not accidentally keep a report alive forever because some other pothole
# 40m away keeps getting photographed.
EXPIRY_RADIUS_M = 10           # "precise" area for expiry grouping
EXPIRY_DAYS = 3
EXPIRY_WINDOW_S = EXPIRY_DAYS * 24 * 3600

SEVERITY_TIERS = {
    "low": (0.0, 0.34),
    "medium": (0.34, 0.67),
    "high": (0.67, 1.01),
}


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lng2 - lng1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def severity_tier(score: float) -> str:
    for tier, (lo, hi) in SEVERITY_TIERS.items():
        if lo <= score < hi:
            return tier
    return "high"


def count_nearby(
    existing_reports: list[dict],
    type_: str,
    lat: float,
    lng: float,
    now: float | None = None,
) -> int:
    """Count existing reports of the same type within CLUSTER_RADIUS_M and
    TIME_WINDOW_S of the given point."""
    now = now if now is not None else time.time()
    count = 0
    for r in existing_reports:
        if r["type"] != type_:
            continue
        if now - r["timestamp"] > TIME_WINDOW_S:
            continue
        if haversine_m(lat, lng, r["lat"], r["lng"]) <= CLUSTER_RADIUS_M:
            count += 1
    return count


IN_FRAME_BOOST_RATE = 0.15  # +15% per extra pothole found in the same photo


def compute_severity(
    defect_area_ratio: float,
    confidence: float,
    nearby_report_count: int,
    pothole_count: int = 1,
) -> float:
    nearby_factor = min(nearby_report_count, NEARBY_CAP) / NEARBY_CAP
    base = (defect_area_ratio * 0.4) + (confidence * 0.2) + (nearby_factor * 0.4)
    in_frame_boost = 1 + IN_FRAME_BOOST_RATE * max(0, pothole_count - 1)
    return round(min(base * in_frame_boost, 1.0), 4)


def is_hotspot(nearby_report_count: int) -> bool:
    # +1 to include the new report itself in the cluster size check
    return (nearby_report_count + 1) >= HOTSPOT_N


def find_expired_report_ids(all_reports: list[dict], now: float | None = None) -> list[int]:
    """Group reports by *precise* location (EXPIRY_RADIUS_M, same type —
    much tighter than the clustering radius used for hotspots/priority).
    If a group's most recent report is older than EXPIRY_WINDOW_S (no new
    photo of that exact spot in EXPIRY_DAYS), every report in the group is
    considered stale and its id is returned for deletion.

    A group with recent activity keeps ALL its reports, even old ones —
    only a group that's gone fully quiet expires, so a pothole that keeps
    getting re-reported never disappears just because its first sighting
    was days ago.
    """
    now = now if now is not None else time.time()
    remaining = list(all_reports)
    expired_ids = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        rest = []
        for r in remaining:
            if r["type"] == seed["type"] and haversine_m(
                seed["lat"], seed["lng"], r["lat"], r["lng"]
            ) <= EXPIRY_RADIUS_M:
                group.append(r)
            else:
                rest.append(r)
        remaining = rest

        most_recent = max(m["timestamp"] for m in group)
        if now - most_recent > EXPIRY_WINDOW_S:
            expired_ids.extend(m["id"] for m in group)

    return expired_ids


def build_priority_list(all_reports: list[dict], top_n: int = 5) -> list[dict]:
    """Greedy spatial clustering (by type) over all reports, ranked by a
    combined severity score = avg(severity) * cluster_size, with a mild
    recency boost so freshly-reported clusters float up."""
    now = time.time()
    remaining = list(all_reports)
    clusters = []

    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        rest = []
        for r in remaining:
            if r["type"] == seed["type"] and haversine_m(
                seed["lat"], seed["lng"], r["lat"], r["lng"]
            ) <= CLUSTER_RADIUS_M:
                members.append(r)
            else:
                rest.append(r)
        remaining = rest

        avg_severity = sum(m["severity_score"] for m in members) / len(members)
        cluster_size = len(members)
        most_recent = max(m["timestamp"] for m in members)
        recency_boost = max(0.0, 1.0 - (now - most_recent) / TIME_WINDOW_S) * 0.2
        combined_severity = round(
            avg_severity * (1 + 0.15 * (cluster_size - 1)) + recency_boost, 4
        )

        centroid_lat = sum(m["lat"] for m in members) / cluster_size
        centroid_lng = sum(m["lng"] for m in members) / cluster_size

        clusters.append(
            {
                "type": seed["type"],
                "lat": round(centroid_lat, 6),
                "lng": round(centroid_lng, 6),
                "cluster_size": cluster_size,
                "avg_severity": round(avg_severity, 4),
                "combined_severity": combined_severity,
                "tier": severity_tier(min(combined_severity, 1.0)),
                "is_hotspot": cluster_size >= HOTSPOT_N,
                "most_recent": most_recent,
            }
        )

    clusters.sort(key=lambda c: c["combined_severity"], reverse=True)
    return clusters[:top_n]
