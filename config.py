"""
config.py
---------
Centralized, environment-driven configuration.

Why this exists: on Hugging Face Spaces (Docker SDK), the container's local
filesystem is WIPED every time the Space restarts or rebuilds — unless you
attach a persistent Storage volume, which HF mounts at /data. Rather than
hardcoding paths everywhere, every path in this app is built from DATA_DIR,
which you point at /data in production and leave as-is for local development.

    Local dev:      DATA_DIR unset  -> everything lives under ./data
    HF Spaces demo: DATA_DIR unset  -> works, but registered students/photos/
                                        attendance history disappear on restart
    HF Spaces prod: DATA_DIR=/data  -> persists across restarts (requires a
                                        Storage volume attached in Space settings)

All other tunables (recognition threshold, admin password, secret key) are
also environment-driven so they can be changed from the HF Spaces "Variables
and secrets" panel without touching code or rebuilding the image.
"""

import os
import secrets
import json

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

DATASET_DIR = os.path.join(DATA_DIR, "dataset")        # raw registered face photos, per student
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")   # gallery vectors (embeddings.json)
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.json")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Face recognition tuning
# ---------------------------------------------------------------------------
IMG_SIZE = 96                     # input resolution fed to MobileNetV2
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.50"))   # min cosine similarity to accept a match
MATCH_MARGIN = float(os.environ.get("MATCH_MARGIN", "0.05"))         # best match must beat 2nd-best by this much
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "60"))       # Laplacian variance floor (lower = more blurry)
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "80"))           # px, reject faces smaller than this
MIN_BRIGHTNESS = float(os.environ.get("MIN_BRIGHTNESS", "40"))       # mean grayscale intensity floor
MAX_BRIGHTNESS = float(os.environ.get("MAX_BRIGHTNESS", "225"))      # mean grayscale intensity ceiling

# ---------------------------------------------------------------------------
# Performance optimizations
# ---------------------------------------------------------------------------
# Reduce the number of face photos required for registration to speed up the process
MIN_REQUIRED_PHOTOS_FOR_REGISTRATION = int(os.environ.get("MIN_REQUIRED_PHOTOS_FOR_REGISTRATION", "1"))

# Performance tuning for MTCNN (using only supported parameters)
MTCNN_MIN_FACE_SIZE = int(os.environ.get("MTCNN_MIN_FACE_SIZE", "40"))
MTCNN_SCALE_FACTOR = float(os.environ.get("MTCNN_SCALE_FACTOR", "0.8"))

# ---------------------------------------------------------------------------
# App / security
# ---------------------------------------------------------------------------
# Generate a secure random key
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# If ADMIN_PASSWORD is unset, the app runs fully open (matches the original
# "no auth needed" brief -- convenient for local demos). Set it as a Space
# secret before you register real students on a public URL.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
AUTH_ENABLED = bool(ADMIN_PASSWORD)

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
PORT = int(os.environ.get("PORT", "7860"))  # HF Spaces Docker SDK default

# Hugging Face Spaces renders the app inside an iframe on huggingface.co, so
# the app must list those origins as allowed frame ancestors -- otherwise the
# browser refuses to load the frame and the Space shows "refused to connect".
# Override with FRAME_ANCESTORS='none' (or a custom allowlist) for a
# standalone deployment that should never be embedded.
FRAME_ANCESTORS = os.environ.get(
    "FRAME_ANCESTORS",
    "'self' https://huggingface.co https://*.hf.space",
).strip()

# ---------------------------------------------------------------------------
# Additional configuration options
# ---------------------------------------------------------------------------
MAX_PHOTO_UPLOADS = int(os.environ.get("MAX_PHOTO_UPLOADS", "10"))  # Max photos per registration
MAX_PHOTO_SIZE = int(os.environ.get("MAX_PHOTO_SIZE", "5")) * 1024 * 1024  # Max photo size in bytes (5MB default)
DEFAULT_CONFIDENCE_DISPLAY = float(os.environ.get("DEFAULT_CONFIDENCE_DISPLAY", "0.7"))  # Default confidence threshold for UI

# Server-side caps -- MAX_PHOTO_UPLOADS/MAX_PHOTO_SIZE above only meant
# something if enforced; a client can send any request it likes regardless
# of what the browser UI restricts, so these are checked in app.py itself,
# and MAX_CONTENT_LENGTH below is a blanket cap enforced by Flask before a
# request body is even fully read into memory.
MAX_CAPTURED_IMAGES = int(os.environ.get("MAX_CAPTURED_IMAGES", "10"))  # webcam captures per registration
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "25")) * 1024 * 1024  # whole-request cap

# ---------------------------------------------------------------------------
# Geofencing (location-gated attendance)
# ---------------------------------------------------------------------------
# These environment variables are only the FALLBACK defaults, used the very
# first time the app runs. Once set, an admin manages the real campus
# location, radius, and mode from the in-app /settings page (a Leaflet map),
# which writes to GEOFENCE_SETTINGS_PATH below -- that file, if present,
# always wins over these env vars. This means redeploying the container never
# wipes out an admin's saved campus location (as long as DATA_DIR persists).
#
# Geofencing is OFF by default. It only turns on once a center point has been
# saved (via env vars or the settings page) -- a bogus (0, 0) default could
# otherwise silently lock every student out.
GEOFENCE_SETTINGS_PATH = os.path.join(DATA_DIR, "geofence.json")

_env_lat = os.environ.get("UNIVERSITY_LAT")
_env_lng = os.environ.get("UNIVERSITY_LNG")

GEOFENCE_DEFAULTS = {
    "enabled": bool(_env_lat and _env_lng),
    "latitude": float(_env_lat) if _env_lat else None,
    "longitude": float(_env_lng) if _env_lng else None,
    "radius_meters": float(os.environ.get("GEOFENCE_RADIUS_METERS", "300")),
    # "strict" blocks check-ins outside the fence outright.
    # "flag" allows them through but marks the record unverified for review.
    "mode": os.environ.get("GEOFENCE_MODE", "strict"),
    # Reject/flag GPS readings this imprecise, in meters -- a phone reporting
    # +/-1000m accuracy isn't a trustworthy signal either way.
    "max_accuracy_meters": float(os.environ.get("GEOFENCE_MAX_ACCURACY_METERS", "150")),
}


def ensure_directories():
    """Creates every directory this app writes to, if they don't already exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def startup_warnings():
    """Plain-text warnings printed once at boot -- not exceptions, just nudges."""
    warnings = []
    if AUTH_ENABLED and len(ADMIN_PASSWORD) < 8:
        warnings.append("ADMIN_PASSWORD is shorter than 8 characters -- consider a longer one for a public deployment.")
    if SECRET_KEY == "dev-secret-key-change-me-in-production":
        warnings.append("SECRET_KEY is still the default value -- set a random SECRET_KEY before deploying publicly.")
    return warnings