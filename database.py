"""
database.py
------------
Defines the SQLAlchemy database models for the attendance system.

Two tables:
1. Student           - stores registered student info (student_id, name, and the folder
                        where their face images live on disk).
2. AttendanceRecord  - one row per "present" event: which student, on what date/time,
                        and how confident the CNN was about the match.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)   # school-issued ID, e.g. "STU001"
    name = db.Column(db.String(100), nullable=False)
    image_folder = db.Column(db.String(200), nullable=False)             # e.g. static/dataset/STU001
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # One student -> many attendance records. backref lets us do record.student
    attendance_records = db.relationship("AttendanceRecord", backref="student", lazy=True)

    def __repr__(self):
        return f"<Student {self.student_id} - {self.name}>"


class AttendanceRecord(db.Model):
    __tablename__ = "attendance_records"

    id = db.Column(db.Integer, primary_key=True)
    student_pk = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.utcnow().date())
    time = db.Column(db.Time, nullable=False, default=lambda: datetime.utcnow().time())
    confidence = db.Column(db.Float, nullable=False)  # CNN softmax confidence (0.0 - 1.0) at recognition time

    def __repr__(self):
        return f"<AttendanceRecord student_pk={self.student_pk} date={self.date}>"
