"""
database.py
------------
SQLAlchemy models.

    Student           - a registered student. image_folder points to their
                         private photo directory (NOT under /static -- see
                         face_engine.py and the /media route in app.py for why).
    AttendanceRecord   - one row per "present" event.

Note on `confidence`: this is now a cosine-similarity score (roughly 0-1,
occasionally slightly negative for a very bad match) rather than a softmax
probability, since recognition switched from a trained classifier to
embedding similarity matching. See face_engine.py for details.

Note on location fields: latitude/longitude/distance_meters/location_verified
are nullable and only populated when geofencing is enabled (see geofence.py).
Existing databases from before this feature was added won't have these
columns -- run_lightweight_migrations() below adds them on startup via plain
ALTER TABLE statements, so there's no need for a full migration framework
for a handful of nullable columns on SQLite.
"""

from datetime import datetime
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

    # cascade="all, delete-orphan" so deleting a student also deletes their
    # attendance history in one db.session.delete(student) call.
    attendance_records = db.relationship(
        "AttendanceRecord", backref="student", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Student {self.student_id} - {self.name}>"


class AttendanceRecord(db.Model):
    __tablename__ = "attendance_records"
    __table_args__ = (
        # Belt-and-suspenders against the check-then-insert race in
        # app.py's attendance route: two near-simultaneous requests for the
        # same student on the same day can't both slip through as separate
        # rows even if the app-level "already marked?" check races.
        db.UniqueConstraint("student_pk", "date", name="uq_student_date"),
    )

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

    def __repr__(self):
        return f"<AttendanceRecord student_pk={self.student_pk} date={self.date}>"


def run_lightweight_migrations(engine):
    """
    Adds any columns/indexes that exist on the current models but not yet in
    an existing SQLite database file -- covers upgrading a deployment that
    predates the geofencing columns or the uq_student_date constraint,
    without needing Alembic for a handful of nullable columns and one index.
    Safe to call every startup; it's a no-op once caught up.
    """
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(attendance_records)"))}
        new_columns = {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "distance_meters": "FLOAT",
            "location_verified": "BOOLEAN",
        }
        for name, col_type in new_columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE attendance_records ADD COLUMN {name} {col_type}"))
        conn.commit()

        # SQLite can't ALTER TABLE to add a constraint after the fact, but a
        # unique index enforces the same guarantee. If a pre-existing database
        # somehow already has duplicate (student_pk, date) rows, this will
        # fail -- caught and logged rather than blocking startup, since that
        # would turn a data-quality issue into a launch-blocking outage.
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_student_date "
                "ON attendance_records(student_pk, date)"
            ))
            conn.commit()
        except Exception as e:
            print(f"[database] Could not create uq_student_date index (likely pre-existing duplicate "
                  f"rows) -- duplicate same-day attendance rows are not blocked at the DB level "
                  f"until this is resolved manually. Error: {e}")