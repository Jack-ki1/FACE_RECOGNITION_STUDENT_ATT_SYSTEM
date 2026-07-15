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
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

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

    id = db.Column(db.Integer, primary_key=True)
    student_pk = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.utcnow().date())
    time = db.Column(db.Time, nullable=False, default=lambda: datetime.utcnow().time())
    confidence = db.Column(db.Float, nullable=False)  # cosine similarity score at recognition time

    def __repr__(self):
        return f"<AttendanceRecord student_pk={self.student_pk} date={self.date}>"
