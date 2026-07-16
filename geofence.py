"""
geofence.py
-----------
Location-gated attendance: verifies a check-in's GPS coordinates fall within
a configured radius of campus before (or alongside) face recognition.

Honesty check, up front: browser-based GPS is not tamper-proof. A determined
person can override their reported coordinates via browser dev tools, an
extension, or a spoofing app paired with certain browsers -- there is no
"mock location" detection API exposed to web pages the way native Android/iOS
apps have. What this DOES meaningfully raise the bar on: casual "I'll just
mark myself present from home" behavior, and -- combined with the face
recognition this sits alongside -- an attacker now needs to fake BOTH their
GPS location AND pass as the registered student's face at the same moment.
That combination is the actual security value here, not GPS alone. This is
documented in the README too; don't oversell this feature to end users.

Settings (center point, radius, mode) are loaded from GEOFENCE_SETTINGS_PATH
if present (managed via the /settings page), falling back to environment
variable defaults from config.GEOFENCE_DEFAULTS otherwise.
"""

import os
import json
import math

import config


def load_settings():
    """Returns the active geofence settings dict, file override > env defaults."""
    if os.path.exists(config.GEOFENCE_SETTINGS_PATH):
        try:
            with open(config.GEOFENCE_SETTINGS_PATH) as f:
                saved = json.load(f)
            merged = dict(config.GEOFENCE_DEFAULTS)
            merged.update(saved)
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return dict(config.GEOFENCE_DEFAULTS)


def save_settings(settings):
    """
    Validates before writing -- an admin fat-fingering radius=0 or a
    latitude outside [-90, 90] would otherwise silently lock every student
    out (or corrupt the distance math) with no error until someone notices
    nobody can check in. Raises ValueError with a human-readable message on
    invalid input; app.py turns that into a flash message.
    """
    lat = settings.get("latitude")
    lng = settings.get("longitude")
    radius = settings.get("radius_meters")
    max_accuracy = settings.get("max_accuracy_meters")

    if settings.get("enabled"):
        if lat is None or lng is None or not math.isfinite(lat) or not math.isfinite(lng):
            raise ValueError("Campus location must be a valid point on the map.")
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError("Campus coordinates are out of range.")
        if radius is None or not math.isfinite(radius) or radius < 10:
            raise ValueError("Radius must be at least 10 meters.")
        if max_accuracy is None or not math.isfinite(max_accuracy) or max_accuracy <= 0:
            raise ValueError("Minimum GPS accuracy must be a positive number.")
    if settings.get("mode") not in ("strict", "flag"):
        settings["mode"] = "strict"

    config.ensure_directories()
    with open(config.GEOFENCE_SETTINGS_PATH, "w") as f:
        json.dump(settings, f)


def haversine_distance_meters(lat1, lng1, lat2, lng2):
    """
    Great-circle distance between two lat/lng points, in meters.
    Standard formula for "how far apart are two points on Earth" -- accurate
    enough for anything short of surveying, which is all a campus geofence needs.
    """
    R = 6371000  # Earth's mean radius, meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)

    a = (math.sin(d_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def check_location(latitude, longitude, accuracy=None):
    """
    Validates a submitted GPS reading against the configured campus geofence.

    Returns a dict:
        {
            "enabled": bool,            # was geofencing even active for this check?
            "allowed": bool,            # should this check-in be permitted?
            "verified": bool,           # was location successfully confirmed on-campus?
            "distance_meters": float | None,
            "mode": "strict" | "flag",
            "reason": str | None        # human-readable explanation when not verified
        }

    If geofencing is disabled entirely, `enabled` is False and `allowed` is
    always True (the feature is simply off).
    """
    settings = load_settings()

    if not settings.get("enabled") or settings.get("latitude") is None or settings.get("longitude") is None:
        return {"enabled": False, "allowed": True, "verified": True, "distance_meters": None,
                "mode": settings.get("mode", "strict"), "reason": None}

    mode = settings.get("mode", "strict")

    if latitude is None or longitude is None:
        reason = "Location wasn't provided — enable location access in your browser and try again."
        return {"enabled": True, "allowed": (mode == "flag"), "verified": False,
                "distance_meters": None, "mode": mode, "reason": reason}

    # A client can submit anything in a form field, including "nan"/"inf",
    # which Python's float() parses without error -- catch that here rather
    # than letting it silently produce a nonsensical distance later.
    valid_numbers = math.isfinite(latitude) and math.isfinite(longitude)
    in_range = valid_numbers and -90 <= latitude <= 90 and -180 <= longitude <= 180
    if not in_range:
        reason = "Location data looked invalid — please try again."
        return {"enabled": True, "allowed": (mode == "flag"), "verified": False,
                "distance_meters": None, "mode": mode, "reason": reason}

    max_accuracy = settings.get("max_accuracy_meters", 150)
    if accuracy is not None and math.isfinite(accuracy) and accuracy > max_accuracy:
        reason = f"GPS signal too imprecise (±{accuracy:.0f}m) to verify your location — try moving outdoors or near a window."
        return {"enabled": True, "allowed": (mode == "flag"), "verified": False,
                "distance_meters": None, "mode": mode, "reason": reason}

    distance = haversine_distance_meters(
        latitude, longitude, settings["latitude"], settings["longitude"]
    )
    radius = settings.get("radius_meters", 300)

    if distance <= radius:
        return {"enabled": True, "allowed": True, "verified": True,
                "distance_meters": distance, "mode": mode, "reason": None}

    reason = f"You appear to be {distance:.0f}m from campus — attendance can only be marked within {radius:.0f}m."
    return {"enabled": True, "allowed": (mode == "flag"), "verified": False,
            "distance_meters": distance, "mode": mode, "reason": reason}