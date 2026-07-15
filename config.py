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
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.55"))   # min cosine similarity to accept a match
MATCH_MARGIN = float(os.environ.get("MATCH_MARGIN", "0.05"))         # best match must beat 2nd-best by this much
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "60"))       # Laplacian variance floor (lower = more blurry)
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "80"))           # px, reject faces smaller than this
MIN_BRIGHTNESS = float(os.environ.get("MIN_BRIGHTNESS", "40"))       # mean grayscale intensity floor
MAX_BRIGHTNESS = float(os.environ.get("MAX_BRIGHTNESS", "225"))      # mean grayscale intensity ceiling

# ---------------------------------------------------------------------------
# App / security
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me-in-production")

# If ADMIN_PASSWORD is unset, the app runs fully open (matches the original
# "no auth needed" brief -- convenient for local demos). Set it as a Space
# secret before you register real students on a public URL.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
AUTH_ENABLED = bool(ADMIN_PASSWORD)

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
PORT = int(os.environ.get("PORT", "7860"))  # HF Spaces Docker SDK default

# ---------------------------------------------------------------------------
# Additional configuration options
# ---------------------------------------------------------------------------
MAX_PHOTO_UPLOADS = int(os.environ.get("MAX_PHOTO_UPLOADS", "10"))  # Max photos per registration
MAX_PHOTO_SIZE = int(os.environ.get("MAX_PHOTO_SIZE", "5")) * 1024 * 1024  # Max photo size in bytes (5MB default)
DEFAULT_CONFIDENCE_DISPLAY = float(os.environ.get("DEFAULT_CONFIDENCE_DISPLAY", "0.7"))  # Default confidence threshold for UI


def ensure_directories():
    """Creates every directory this app writes to, if they don't already exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)