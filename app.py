"""
app.py
------
Flask application entry point. Defines all routes:

    GET  /            - home page (student count, model status, quick-start guide)
    GET  /register     - registration form (upload photos and/or capture via webcam)
    POST /register     - saves face images for a new student
    GET  /train          - (re)trains the CNN on all currently registered students
    GET  /attendance      - take-attendance page (webcam or upload)
    POST /attendance      - runs face recognition on the submitted image, marks attendance
    GET  /dashboard        - attendance records, filterable by date and student

Run with:  python app.py   (then open http://127.0.0.1:5000)
"""

import os
import json
import base64
import io
from datetime import datetime, date

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image

from database import db, Student, AttendanceRecord
import model as face_model

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///attendance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "dev-secret-key-change-me"  # only used to sign flash messages; fine for a school project

db.init_app(app)

DATASET_DIR = os.path.join("static", "dataset")
CONFIDENCE_THRESHOLD = 0.75  # minimum CNN confidence required to mark someone present


def sanitize_student_id(student_id):
    """
    Sanitize student ID to be safe for filesystem paths.
    Replace problematic characters like '/' with safe alternatives.
    """
    # Replace problematic characters with underscores
    sanitized = student_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('|', '_')
    return sanitized


def cleanup_orphaned_students():
    """
    Remove student records from the database that don't have corresponding image folders.
    """
    students = Student.query.all()
    for student in students:
        if not os.path.exists(student.image_folder):
            # Remove attendance records first due to foreign key constraint
            AttendanceRecord.query.filter_by(student_pk=student.id).delete()
            # Remove the student
            db.session.delete(student)
            print(f"Removed orphaned student: {student.student_id}")
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error during cleanup: {e}")


def decode_data_url(data_url):
    """
    Converts a webcam canvas data URL ('data:image/jpeg;base64,...') into a BGR
    numpy image, the format OpenCV and our model.py functions expect.
    """
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


@app.route("/")
def index():
    student_count = Student.query.count()
    trained = os.path.exists(face_model.MODEL_PATH)
    return render_template("index.html", student_count=student_count, trained=trained)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()

    # Validate inputs
    if not student_id or not name:
        flash("Student ID and name are required.", "error")
        return redirect(url_for("register"))

    # Check for potentially dangerous characters in student_id
    if '..' in student_id or student_id.startswith('/') or '../' in student_id:
        flash("Invalid characters in student ID.", "error")
        return redirect(url_for("register"))

    if Student.query.filter_by(student_id=student_id).first():
        flash(f"Student ID '{student_id}' is already registered.", "error")
        return redirect(url_for("register"))

    # Sanitize the student ID for filesystem safety
    sanitized_student_id = sanitize_student_id(student_id)
    student_folder = os.path.join(DATASET_DIR, sanitized_student_id)
    os.makedirs(student_folder, exist_ok=True)

    saved_count = 0

    # --- Option A: regular file uploads (<input type="file" name="face_images" multiple>) ---
    uploaded_files = request.files.getlist("face_images")
    for f in uploaded_files:
        if f and f.filename:
            # Verify it's actually an image file
            if not f.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            
            filepath = os.path.join(student_folder, f"upload_{saved_count}.jpg")
            f.save(filepath)
            saved_count += 1

    # --- Option B: webcam captures, sent as a JSON array of data URLs ---
    captured_json = request.form.get("captured_images")
    if captured_json:
        try:
            data_urls = json.loads(captured_json)
            for data_url in data_urls:
                if not data_url.startswith('data:image'):
                    continue  # Skip invalid data URLs
                image_bgr = decode_data_url(data_url)
                filepath = os.path.join(student_folder, f"capture_{saved_count}.jpg")
                cv2.imwrite(filepath, image_bgr)
                saved_count += 1
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            pass  # malformed capture data - simply ignore it

    if saved_count == 0:
        flash("Please upload or capture at least one face image.", "error")
        # Clean up the empty folder if it was created
        try:
            os.rmdir(student_folder)
        except OSError:
            pass  # Folder might not be empty or might not exist
        return redirect(url_for("register"))

    new_student = Student(student_id=student_id, name=name, image_folder=student_folder)
    db.session.add(new_student)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Error saving student: {str(e)}", "error")
        # Clean up the folder
        try:
            import shutil
            shutil.rmtree(student_folder)
        except:
            pass
        return redirect(url_for("register"))

    flash(
        f"Registered {name} ({student_id}) with {saved_count} image(s). "
        f"Don't forget to (re)train the model before taking attendance.",
        "success"
    )
    return redirect(url_for("register"))


@app.route("/train")
def train():
    result = face_model.train_model(epochs=10)
    if result["success"]:
        flash(f"{result['message']} (training accuracy: {result['final_accuracy'] * 100:.1f}%)", "success")
    else:
        flash(result["message"], "error")
    return redirect(url_for("index"))


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "GET":
        return render_template("attendance.html")

    # Accept either a webcam snapshot (data URL) or a regular file upload
    image_bgr = None
    if request.form.get("snapshot"):
        image_bgr = decode_data_url(request.form["snapshot"])
    elif "face_image" in request.files and request.files["face_image"].filename:
        file_bytes = np.frombuffer(request.files["face_image"].read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return render_template("attendance.html", result={"success": False, "message": "No image received. Please capture or upload a photo."})

    prediction = face_model.predict_face(image_bgr, confidence_threshold=CONFIDENCE_THRESHOLD)

    if prediction["student_id"] is None:
        return render_template("attendance.html", result={
            "success": False,
            "message": prediction.get("reason", "Face not recognized."),
            "confidence": prediction["confidence"]
        })

    student = Student.query.filter_by(student_id=prediction["student_id"]).first()
    if student is None:
        return render_template("attendance.html", result={"success": False, "message": "Matched student not found in database."})

    today = date.today()
    already_marked = AttendanceRecord.query.filter_by(student_pk=student.id, date=today).first()

    if already_marked:
        message = f"{student.name} was already marked present today at {already_marked.time.strftime('%H:%M:%S')}."
    else:
        record = AttendanceRecord(
            student_pk=student.id,
            date=today,
            time=datetime.now().time(),
            confidence=prediction["confidence"]
        )
        db.session.add(record)
        db.session.commit()
        message = f"{student.name} marked present (confidence: {prediction['confidence'] * 100:.1f}%)."

    return render_template("attendance.html", result={
        "success": True,
        "message": message,
        "student_name": student.name,
        "confidence": prediction["confidence"]
    })


@app.route("/dashboard")
def dashboard():
    query = AttendanceRecord.query.join(Student)

    filter_date = request.args.get("date", "")
    filter_student = request.args.get("student_id", "")

    if filter_date:
        try:
            parsed_date = datetime.strptime(filter_date, "%Y-%m-%d").date()
            query = query.filter(AttendanceRecord.date == parsed_date)
        except ValueError:
            pass  # ignore malformed date input, fall through to unfiltered results

    if filter_student:
        query = query.filter(Student.student_id == filter_student)

    records = query.order_by(AttendanceRecord.date.desc(), AttendanceRecord.time.desc()).all()
    all_students = Student.query.order_by(Student.name).all()

    return render_template(
        "dashboard.html",
        records=records,
        students=all_students,
        filter_date=filter_date,
        filter_student=filter_student
    )


@app.route("/diagnose")
def diagnose():
    """Diagnostic endpoint to help troubleshoot face detection issues"""
    import model as face_model
    face_model.diagnose_face_detection()
    flash("Face detection diagnosis printed to console. Check terminal output.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with app.app_context():
        db.create_all()  # creates attendance.db and its tables on first run
        cleanup_orphaned_students()  # Clean up any orphaned records
    app.run(debug=True)
