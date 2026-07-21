"""
app.py
------
Flask application entry point.

    GET  /                        - overview / home
    GET  /login, GET/POST /login  - admin password gate (no-op if ADMIN_PASSWORD unset)
    GET  /logout
    GET  /register, POST /register - register a student (instant, no training step)
    GET  /students                 - roster grid (photos, delete)
    POST /students/<id>/delete
    POST /reindex                  - rebuild the embedding gallery from stored photos
    GET  /attendance, POST /attendance - webcam/upload check-in (open, kiosk-style)
    GET  /dashboard                - records, filters, 7-day chart      (admin-gated)
    GET  /dashboard/export.csv     - CSV export                        (admin-gated)
    GET  /settings, POST /settings - campus geofence location/radius/mode (admin-gated)
    GET  /media/<student_id>/<filename> - serves a student's private photo
    GET  /healthz                  - uptime check

Run locally with:  python app.py
Run in production (Docker/HF Spaces) with: gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import io
import csv
import json
import base64
import binascii
import shutil
from datetime import datetime, date, timedelta

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, send_file, abort, Response
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy.exc import IntegrityError

import config
from database import db, Student, AttendanceRecord, run_lightweight_migrations
import face_engine
import geofence
from auth import (
    login_required, is_logged_in, check_password, login_limiter,
    attendance_limiter, client_ip, get_csrf_token, csrf_protect
)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH  # blanket cap on request body size
app.secret_key = config.SECRET_KEY

# Session cookie hardening. SECURE is skipped under FLASK_DEBUG (local http dev) 
# since browsers won't set a Secure cookie over plain HTTP anyway.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = not config.DEBUG

# Hugging Face Spaces (and most PaaS) sit behind a reverse proxy -- without
# this, request.remote_addr is always the proxy's IP, which would make the
# rate limiters below either useless or wrongly lock out every visitor at once.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

db.init_app(app)

# Cache for face recognition gallery to avoid reloading from disk on each request
_gallery_cache = None
_gallery_last_modified = 0

def get_cached_gallery():
    """Get the face recognition gallery with caching to avoid frequent disk reads."""
    global _gallery_cache, _gallery_last_modified
    try:
        current_mtime = os.path.getmtime(config.EMBEDDINGS_PATH) if os.path.exists(config.EMBEDDINGS_PATH) else 0
        if _gallery_cache is None or current_mtime > _gallery_last_modified:
            _gallery_cache = face_engine.load_gallery()
            _gallery_last_modified = current_mtime
    except Exception:
        # If there's an error loading the gallery, reset the cache
        _gallery_cache = None
    return _gallery_cache

# Run at import time (not just under __main__) so this also works when
# gunicorn imports `app` directly in production.
config.ensure_directories()
with app.app_context():
    db.create_all()
    run_lightweight_migrations(db.engine)

# Warm up face recognition models after all modules are loaded
try:
    face_engine.warm_up_models()
except Exception as e:
    print(f"[app] Warning: Could not warm up models: {e}")

for warning in config.startup_warnings():
    print(f"[app] WARNING: {warning}")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


@app.before_request
def _csrf_protect():
    csrf_protect()


@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    # X-Frame-Options can't express a cross-origin allowlist (ALLOW-FROM is
    # deprecated and ignored), so only emit the legacy DENY header when
    # framing is fully disallowed; otherwise rely on CSP frame-ancestors,
    # which lets Hugging Face embed the Space in its iframe.
    if config.FRAME_ANCESTORS == "'none'":
        response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Explicit allowlist for the CDNs this app actually loads (fonts, icons,
    # the map, and the chart library) rather than a blanket allow. Inline
    # scripts are used throughout the templates for simplicity, hence
    # 'unsafe-inline' on script-src -- a nonce-based policy would close that
    # gap for anyone hardening this further.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        f"frame-ancestors {config.FRAME_ANCESTORS}"
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_data_url(data_url):
    """Converts a webcam canvas data URL into a BGR numpy image."""
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_student_id(raw_id):
    """Keeps only filesystem/URL-safe characters, since this doubles as a folder name."""
    return "".join(c for c in raw_id if c.isalnum() or c in "-_")


def primary_photo_url(student):
    """URL to a student's first saved reference photo, or None if unavailable."""
    try:
        files = sorted(f for f in os.listdir(student.image_folder)
                        if os.path.isfile(os.path.join(student.image_folder, f)))
    except OSError:
        files = []
    if not files:
        return None
    return url_for("media", student_id=student.student_id, filename=files[0])


@app.context_processor
def inject_globals():
    geofence_settings = geofence.load_settings()
    return {
        "auth_enabled": config.AUTH_ENABLED,
        "logged_in": is_logged_in(),
        "max_photo_uploads": config.MAX_PHOTO_UPLOADS,
        "geofence_enabled": bool(geofence_settings.get("enabled") and geofence_settings.get("latitude") is not None),
        "csrf_token": get_csrf_token
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if not config.AUTH_ENABLED:
        return redirect(url_for("index"))

    ip = client_ip()

    if request.method == "POST":
        if login_limiter.is_limited(ip):
            flash("Too many failed attempts — please wait a few minutes and try again.", "error")
        elif check_password(request.form.get("password", "")):
            login_limiter.reset(ip)
            session["logged_in"] = True
            flash("Welcome back.", "success")
            return redirect(request.args.get("next") or url_for("index"))
        else:
            login_limiter.record(ip)
            flash("Incorrect password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    flash("Logged out.", "success")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Core pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    student_count = Student.query.count()
    present_today = AttendanceRecord.query.filter_by(date=date.today()).count()
    indexed_count = len(face_engine.load_gallery())
    return render_template(
        "index.html",
        student_count=student_count,
        present_today=present_today,
        indexed_count=indexed_count
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "GET":
        return render_template("register.html")

    raw_id = request.form.get("student_id", "").strip()
    student_id = sanitize_student_id(raw_id)
    name = request.form.get("name", "").strip()

    if not student_id or not name:
        flash("Student ID and name are required.", "error")
        return redirect(url_for("register"))

    if Student.query.filter_by(student_id=student_id).first():
        flash(f"Student ID '{student_id}' is already registered.", "error")
        return redirect(url_for("register"))

    student_folder = os.path.join(config.DATASET_DIR, student_id)
    os.makedirs(student_folder, exist_ok=True)

    saved_faces = []
    warnings = []
    saved_count = 0

    def handle_image(image_bgr):
        nonlocal saved_count
        face = face_engine.detect_face(image_bgr)
        if face is None:
            warnings.append("A photo was skipped — no face detected.")
            return
        for w in face_engine.assess_quality(face):
            warnings.append(w)
        filepath = os.path.join(student_folder, f"photo_{saved_count}.jpg")
        if not cv2.imwrite(filepath, image_bgr):
            warnings.append("A photo was skipped — it could not be saved.")
            return
        saved_faces.append(face)
        saved_count += 1

    # Process uploaded files
    uploaded_files = request.files.getlist("face_images")
    if len(uploaded_files) > config.MAX_PHOTO_UPLOADS:
        flash(f"You can upload a maximum of {config.MAX_PHOTO_UPLOADS} photos.", "error")
        return redirect(url_for("register"))

    for f in uploaded_files:
        if f and f.filename and is_allowed_file(f.filename):
            file_bytes = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                handle_image(image_bgr)

    # Process captured images from webcam. The 5-photo cap in register.html's
    # JS is a UI nicety only -- a request built by hand could send any number
    # of data URLs, so the real limit is enforced here.
    captured_json = request.form.get("captured_images")
    if captured_json:
        try:
            data_urls = json.loads(captured_json)
            if isinstance(data_urls, list) and len(data_urls) > config.MAX_CAPTURED_IMAGES:
                flash(f"You can capture a maximum of {config.MAX_CAPTURED_IMAGES} photos.", "error")
                return redirect(url_for("register"))
            for data_url in data_urls:
                if not (isinstance(data_url, str) and data_url.startswith("data:image")):
                    continue
                try:
                    handle_image(decode_data_url(data_url))
                except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
                    warnings.append("A captured photo was skipped — it could not be read.")
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    if saved_count == 0:
        flash("No usable face photos were saved — make sure a face is clearly visible and try again.", "error")
        try:
            os.rmdir(student_folder)
        except OSError:
            pass
        return redirect(url_for("register"))

    new_student = Student(student_id=student_id, name=name, image_folder=student_folder)
    db.session.add(new_student)
    try:
        db.session.commit()
    except IntegrityError:
        # Two requests registered the same student_id at almost the same
        # moment -- the DB's unique constraint is the real guard here, the
        # earlier query-based check is just a fast path for the common case.
        db.session.rollback()
        shutil.rmtree(student_folder, ignore_errors=True)
        flash(f"Student ID '{student_id}' was just registered by someone else — please use a different ID.", "error")
        return redirect(url_for("register"))

    for face in saved_faces:
        face_engine.add_photo_to_gallery(student_id, face)

    if student_id != raw_id:
        flash(f"Student ID simplified to '{student_id}' (letters, numbers, - and _ only).", "warning")
    flash(f"Registered {name} ({student_id}) with {saved_count} photo(s) — ready for attendance immediately.", "success")
    if warnings:
        shown = warnings[:3]
        flash(" · ".join(shown) + (f" (+{len(warnings) - 3} more)" if len(warnings) > 3 else ""), "warning")

    return redirect(url_for("students"))


@app.route("/reindex", methods=["POST"])
@login_required
def reindex():
    result = face_engine.reindex_gallery()
    flash(
        f"Reindexed {result['students_indexed']} student(s) from "
        f"{result['images_processed']} photo(s) ({result['images_skipped']} skipped).",
        "success"
    )
    return redirect(url_for("students"))


@app.route("/students")
@login_required
def students():
    all_students = Student.query.order_by(Student.name).all()
    photos = {}
    counts = {}
    
    for s in all_students:
        photos[s.student_id] = primary_photo_url(s)
        try:
            if s.image_folder and os.path.isdir(s.image_folder):
                counts[s.student_id] = len(
                    [f for f in os.listdir(s.image_folder) if os.path.isfile(os.path.join(s.image_folder, f))]
                )
            else:
                counts[s.student_id] = 0
        except (OSError, PermissionError):
            counts[s.student_id] = 0
    
    return render_template("students.html", students=all_students, photos=photos, counts=counts)


@app.route("/students/<student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    student = Student.query.filter_by(student_id=student_id).first()
    if student is None:
        abort(404)

    folder = student.image_folder
    name = student.name
    db.session.delete(student)
    db.session.commit()
    face_engine.remove_student_from_gallery(student_id)
    if folder and os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)

    flash(f"Removed {name} ({student_id}) and their photos.", "success")
    return redirect(url_for("students"))


@app.route("/media/<student_id>/<filename>")
def media(student_id, filename):
    # Photos are intentionally NOT under /static (see face_engine.py docstring).
    # When an admin password is set, only logged-in admins can fetch photos directly.
    if config.AUTH_ENABLED and not is_logged_in():
        abort(404)

    # Validate student_id and filename to prevent directory traversal attacks
    sanitized_student_id = sanitize_student_id(student_id)
    if sanitized_student_id != student_id:
        abort(404)
        
    secure_filename_val = secure_filename(filename)
    if secure_filename_val != filename:
        abort(404)

    filepath = os.path.join(config.DATASET_DIR, sanitized_student_id, secure_filename_val)
    if not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath)


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "GET":
        return render_template("attendance.html", geofence_settings=geofence.load_settings())

    ip = client_ip()
    if attendance_limiter.is_limited(ip):
        return render_template("attendance.html", result={
            "success": False,
            "message": "Too many attempts from this device — please wait a minute and try again."
        })
    attendance_limiter.record(ip)

    # --- Location check FIRST: cheap, and no point running face recognition
    # if the check-in is going to be rejected on location anyway. -----------
    def parse_float(key):
        raw = request.form.get(key)
        try:
            return float(raw) if raw not in (None, "") else None
        except ValueError:
            return None

    latitude = parse_float("latitude")
    longitude = parse_float("longitude")
    accuracy = parse_float("accuracy")

    location = geofence.check_location(latitude, longitude, accuracy)

    if location["enabled"] and not location["allowed"]:
        return render_template("attendance.html", result={
            "success": False,
            "message": location["reason"] or "Location verification failed.",
            "location": location
        })

    # --- Face recognition ----------------------------------------------------
    image_bgr = None
    try:
        if request.form.get("snapshot"):
            image_bgr = decode_data_url(request.form["snapshot"])
        else:
            # No snapshot captured
            image_bgr = None
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as e:
        print(f"[app] Error processing image: {e}")
        image_bgr = None

    if image_bgr is None:
        return render_template("attendance.html", result={
            "success": False, "message": "No usable image received — please capture a photo using the webcam."
        })

    gallery = get_cached_gallery()
    match = face_engine.match_face(image_bgr)

    if match["student_id"] is None:
        reason = match.get("reason", "Face not recognized.")
        confidence = match.get("confidence", 0.0)
        return render_template("attendance.html", result={
            "success": False,
            "message": reason,
            "confidence": confidence
        })

    student = Student.query.filter_by(student_id=match["student_id"]).first()
    if student is None:
        return render_template("attendance.html", result={"success": False, "message": "Matched student record not found."})

    today = date.today()
    already_marked = AttendanceRecord.query.filter_by(student_pk=student.id, date=today).first()

    if already_marked:
        message = f"{student.name} was already marked present today at {already_marked.time.strftime('%H:%M:%S')}."
    else:
        record = AttendanceRecord(
            student_pk=student.id, date=today, time=datetime.now().time(), confidence=match["confidence"],
            latitude=latitude, longitude=longitude,
            distance_meters=location["distance_meters"],
            location_verified=location["verified"] if location["enabled"] else None
        )
        db.session.add(record)
        try:
            db.session.commit()
            message = f"Welcome, {student.name}! You're marked present."
            if location["enabled"] and location["mode"] == "flag" and not location["verified"]:
                message += " (Flagged: location could not be verified.)"
        except IntegrityError:
            # Two near-simultaneous check-ins for the same student today --
            # the uq_student_date index caught it, so treat it the same as
            # the "already marked" branch above rather than erroring out.
            db.session.rollback()
            message = f"{student.name} was already marked present today."

    return render_template("attendance.html", result={
        "success": True,
        "message": message,
        "student_name": student.name,
        "confidence": match["confidence"],
        "location": location,
        # Only surface the reference photo when the app is running open (no admin
        # password set) -- otherwise an anonymous kiosk visitor could pull photos
        # of registered students just by triggering recognition repeatedly.
        "photo_url": primary_photo_url(student) if not config.AUTH_ENABLED else None
    })


@app.route("/dashboard")
@login_required
def dashboard():
    try:
        query = AttendanceRecord.query.join(Student)

        filter_date = request.args.get("date", "")
        filter_student = request.args.get("student_id", "")

        if filter_date:
            try:
                query = query.filter(AttendanceRecord.date == datetime.strptime(filter_date, "%Y-%m-%d").date())
            except ValueError:
                pass
        if filter_student:
            query = query.filter(Student.student_id == filter_student)

        records = query.order_by(AttendanceRecord.date.desc(), AttendanceRecord.time.desc()).all()
        all_students = Student.query.order_by(Student.name).all()

        chart_labels, chart_values = [], []
        for i in range(6, -1, -1):
            day = date.today() - timedelta(days=i)
            chart_labels.append(day.strftime("%a %d"))
            chart_values.append(AttendanceRecord.query.filter_by(date=day).count())

        today_present = AttendanceRecord.query.filter_by(date=date.today()).count()
        flagged_today = AttendanceRecord.query.filter_by(date=date.today(), location_verified=False).count()

        return render_template(
            "dashboard.html",
            records=records,
            students=all_students,
            filter_date=filter_date,
            filter_student=filter_student,
            chart_labels=chart_labels,
            chart_values=chart_values,
            today_present=today_present,
            total_students=len(all_students),
            flagged_today=flagged_today
        )
    except Exception as e:
        flash(f"Error loading dashboard: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/dashboard/export.csv")
@login_required
def export_csv():
    rows = AttendanceRecord.query.join(Student).order_by(
        AttendanceRecord.date.desc(), AttendanceRecord.time.desc()
    ).all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Date", "Time", "Student ID", "Name", "Confidence", "Distance (m)", "Location Verified"])
    for r in rows:
        writer.writerow([
            r.date, r.time.strftime("%H:%M:%S"), r.student.student_id, r.student.name, f"{r.confidence:.3f}",
            f"{r.distance_meters:.0f}" if r.distance_meters is not None else "",
            ("Yes" if r.location_verified else "No") if r.location_verified is not None else "N/A"
        ])

    response = Response(buffer.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_export.csv"
    return response


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        try:
            new_settings = {
                "enabled": request.form.get("enabled") == "on",
                "latitude": float(request.form["latitude"]),
                "longitude": float(request.form["longitude"]),
                "radius_meters": float(request.form.get("radius_meters", 300)),
                "mode": request.form.get("mode", "strict"),
                "max_accuracy_meters": float(request.form.get("max_accuracy_meters", 150)),
            }
            geofence.save_settings(new_settings)
        except (KeyError, ValueError) as e:
            flash(str(e) if isinstance(e, ValueError) and str(e) else "Please set a valid location on the map before saving.", "error")
            return redirect(url_for("settings"))

        flash("Campus location settings saved.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", geofence_settings=geofence.load_settings())


@app.route("/favicon.ico")
def favicon():
    return "", 204  # No Content response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)