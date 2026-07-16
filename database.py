"""
database.py
-----------
SQLAlchemy ORM models and connection setup.

This file defines the database schema as three tables:

    students (Student)              -- registered people who can check in
    attendance_records (AttendanceRecord) -- individual check-ins, tied to students
    alembic_version                 -- migration tracking (managed by Alembic)

Each student has:
    - A unique student_id (used as a folder name under DATASET_DIR)
    - A display name
    - A reference to their image folder (filled in by app.register_student)
    - Created at timestamp
    - Last seen timestamp (last attendance date)
    - Account status (active/inactive)

Each attendance record has:
    - A foreign key to a student (student_pk)
    - Date and time of check-in
    - Confidence score from face recognition
    - Optional location data (lat/lng, distance from campus center, verification status)
    - Device information (browser/device type)
    - IP address of the check-in
    - Session identifier

This file also contains the lightweight migration runner, which handles schema
changes that are too small to justify pulling in Alembic (a full migration
framework) -- currently just the location columns added in a later revision.

The models themselves are intentionally minimal; business logic lives in app.py
and face_engine.py, not here.
"""

from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

db = SQLAlchemy()


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    image_folder = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime)  # Last attendance date
    active = db.Column(db.Boolean, default=True)  # Account status

    # cascade="all, delete-orphan" so deleting a student also deletes their
    # attendance history in one db.session.delete(student) call.
    attendance_records = db.relationship(
        "AttendanceRecord", backref="student", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Student {self.student_id} - {self.name}>"


class AttendanceRecord(db.Model):
    __tablename__ = "attendance_records"

    id = db.Column(db.Integer, primary_key=True)
    student_pk = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.utcnow().date())
    time = db.Column(db.Time, nullable=False, default=lambda: datetime.utcnow().time())
    confidence = db.Column(db.Float, nullable=False)  # cosine similarity score at recognition time

    # Populated only when geofencing is enabled (see geofence.py).
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    distance_meters = db.Column(db.Float, nullable=True)
    location_verified = db.Column(db.Boolean, nullable=True)  # None = geofencing was off for this check-in
    
    # Additional fields for enhanced functionality
    device_info = db.Column(db.String(200))  # Browser/device info
    ip_address = db.Column(db.String(45))    # IP address of check-in
    session_id = db.Column(db.String(100))   # Session identifier

    # Composite unique constraint: one check-in per student per day
    __table_args__ = (
        db.UniqueConstraint("student_pk", "date", name="uq_student_date"),
    )

    def __repr__(self):
        return f"<AttendanceRecord student_pk={self.student_pk} date={self.date}>"


def run_lightweight_migrations(engine):
    """
    Run schema updates too small to justify a full migration framework.
    
    Checks the current schema against expected state and applies deltas as
    needed. This is run automatically at boot time by app.py.
    """
    with engine.connect() as conn:
        # Check for location columns (added after initial release)
        result = conn.execute(text("PRAGMA table_info(attendance_records)")).fetchall()
        existing_columns = {row[1] for row in result}
        
        # Add location-related columns if missing
        location_columns = {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "distance_meters": "FLOAT",
            "location_verified": "BOOLEAN"
        }
        
        for col_name, col_type in location_columns.items():
            if col_name not in existing_columns:
                conn.execute(text(f"ALTER TABLE attendance_records ADD COLUMN {col_name} {col_type}"))
        
        # Add enhanced functionality columns to attendance_records if missing
        enhanced_columns = {
            "device_info": "VARCHAR(200)",
            "ip_address": "VARCHAR(45)",
            "session_id": "VARCHAR(100)"
        }
        
        for col_name, col_type in enhanced_columns.items():
            if col_name not in existing_columns:
                conn.execute(text(f"ALTER TABLE attendance_records ADD COLUMN {col_name} {col_type}"))
        
        # Add new student columns if missing
        result = conn.execute(text("PRAGMA table_info(students)")).fetchall()
        existing_student_columns = {row[1] for row in result}
        
        student_columns = {
            "created_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
            "last_seen": "DATETIME",
            "active": "BOOLEAN DEFAULT 1"
        }
        
        for col_name, col_def in student_columns.items():
            if col_name not in existing_student_columns:
                conn.execute(text(f"ALTER TABLE students ADD COLUMN {col_name} {col_def}"))
        
        # Create unique index for student/date constraint if needed
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_student_date "
                "ON attendance_records(student_pk, date)"
            ))
        except Exception as e:
            print(f"[database] Could not create uq_student_date index (likely pre-existing duplicate "
                  f"rows) -- duplicate same-day attendance rows are not blocked at the DB level "
                  f"until this is resolved manually. Error: {e}")


def update_last_seen(student_id):
    """Update the last seen timestamp for a student."""
    student = Student.query.filter_by(student_id=student_id).first()
    if student:
        student.last_seen = db.func.current_timestamp()
        db.session.commit()


def get_active_students_count():
    """Get count of students who have been active recently."""
    # Count students who have attended in the last 30 days
    thirty_days_ago = datetime.now() - timedelta(days=30)
    
    active_count = db.session.query(Student).filter(
        Student.active == True,
        Student.last_seen >= thirty_days_ago
    ).count()
    
    return active_count