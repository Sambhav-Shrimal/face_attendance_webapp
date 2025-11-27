"""
Face Recognition Attendance Web App - SQLite Version
Run this file to start the web server
Access from phone: http://YOUR_PC_IP:5000
"""
import sys
import traceback

def log_error(e):
    with open("error_log.txt", "w") as f:
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())

import os
from flask import Flask

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)



from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
from datetime import datetime, date, time, timedelta
from typing import Optional
import pytz
import pickle
import cv2
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import csv
import os

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
DB_FILE = "attendance_system.db"
IST = pytz.timezone("Asia/Kolkata")


# ---------------- DB Connection ----------------
def get_db():
    """Get database connection"""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except Exception as err:
        print(f"Database error: {err}")
        return None


def now_ist():
    """Return current time in IST"""
    return datetime.now(IST).replace(tzinfo=None)


# ---------------- Initialize Database ----------------
def init_db():
    """Create database and tables if not exist"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Create employee_faces table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS employee_faces
                       (
                           employee_id
                           TEXT
                           PRIMARY
                           KEY,
                           emp_name
                           TEXT
                           NOT
                           NULL,
                           face_encoding
                           BLOB
                           NOT
                           NULL,
                           pay_type
                           TEXT
                           NOT
                           NULL,
                           rate
                           REAL
                           NOT
                           NULL,
                           shift_hours
                           REAL
                           NOT
                           NULL,
                           registered_date
                           TEXT
                           NOT
                           NULL
                       )
                       """)

        # Create attendance_records table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS attendance_records
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           employee_id
                           TEXT
                           NOT
                           NULL,
                           emp_name
                           TEXT
                           NOT
                           NULL,
                           date
                           TEXT
                           NOT
                           NULL,
                           start_time
                           TEXT,
                           lunch_start
                           TEXT,
                           lunch_end
                           TEXT,
                           end_time
                           TEXT,
                           total_hours
                           REAL,
                           break_hours
                           REAL,
                           overtime_hours
                           REAL,
                           pay_type
                           TEXT,
                           rate
                           REAL,
                           shift_hours
                           REAL,
                           total_pay
                           REAL,
                           last_updated
                           TEXT,
                           UNIQUE
                       (
                           employee_id,
                           date
                       )
                           )
                       """)

        # Create advance_payments table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS advance_payments
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           employee_id
                           TEXT
                           NOT
                           NULL,
                           emp_name
                           TEXT
                           NOT
                           NULL,
                           amount
                           REAL
                           NOT
                           NULL,
                           date
                           TEXT
                           NOT
                           NULL,
                           payment_mode
                           TEXT
                           NOT
                           NULL,
                           reference_no
                           TEXT,
                           reason
                           TEXT,
                           status
                           TEXT
                           DEFAULT
                           'pending',
                           remaining_balance
                           REAL
                           NOT
                           NULL,
                           created_at
                           TEXT
                           NOT
                           NULL,
                           FOREIGN
                           KEY
                       (
                           employee_id
                       ) REFERENCES employee_faces
                       (
                           employee_id
                       )
                           )
                       """)

        # Create company_settings table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS company_settings
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           DEFAULT
                           1,
                           office_start_time
                           TEXT
                           DEFAULT
                           '09:00:00',
                           office_end_time
                           TEXT
                           DEFAULT
                           '17:00:00',
                           ot_rate_multiplier
                           REAL
                           DEFAULT
                           1.5,
                           sunday_auto_ot
                           INTEGER
                           DEFAULT
                           1
                       )
                       """)

        # Insert default settings
        cursor.execute("INSERT OR IGNORE INTO company_settings (id) VALUES (1)")

        # Create holidays table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS holidays
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           holiday_date
                           TEXT
                           NOT
                           NULL
                           UNIQUE,
                           holiday_name
                           TEXT
                           NOT
                           NULL,
                           is_ot_applicable
                           INTEGER
                           DEFAULT
                           1
                       )
                       """)

        # Create break_types table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS break_types
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           break_name
                           TEXT
                           NOT
                           NULL,
                           duration_minutes
                           INTEGER
                           NOT
                           NULL,
                           is_paid
                           INTEGER
                           DEFAULT
                           1,
                           deduct_rate_per_minute
                           REAL
                           DEFAULT
                           0
                       )
                       """)

        # Insert default breaks
        cursor.execute("SELECT COUNT(*) FROM break_types")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("""
                           INSERT INTO break_types (break_name, duration_minutes, is_paid, deduct_rate_per_minute)
                           VALUES ('Lunch Break', 30, 1, 0),
                                  ('Tea Break', 15, 1, 0)
                           """)

        conn.commit()
        cursor.close()
        conn.close()
        print("✓ SQLite Database initialized")

    except Exception as e:
        print(f"Database init error: {e}")


# ---------------- Face Recognition Functions ----------------
def load_known_faces():
    """Load all employee faces from database"""
    conn = get_db()
    if not conn:
        return [], [], []

    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, emp_name, face_encoding FROM employee_faces")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    known_encodings = []
    known_ids = []
    known_names = []

    for row in rows:
        encoding = pickle.loads(row['face_encoding'])
        known_encodings.append(encoding)
        known_ids.append(row['employee_id'])
        known_names.append(row['emp_name'])

    return known_encodings, known_ids, known_names


def base64_to_image(base64_string):
    """Convert base64 string to numpy image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None


def check_duplicate_face(face_encoding):
    """Check if face already exists in database (anti-scam)"""
    known_encodings, known_ids, known_names = load_known_faces()

    if not known_encodings:
        return None

    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    # Only flag as duplicate if distance is very small (< 0.35 is extremely similar)
    if len(face_distances) > 0:
        best_match_distance = np.min(face_distances)
        best_match_idx = np.argmin(face_distances)

        # Threshold of 0.35 means it's the SAME person (not just similar)
        if best_match_distance < 0.35:
            return {
                'employee_id': known_ids[best_match_idx],
                'emp_name': known_names[best_match_idx]
            }

    return None


def recognize_face_from_image(image, known_encodings, known_ids, known_names):
    """Recognize face in image"""
    if not known_encodings:
        return None

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(image, face_locations)
    if not face_encodings:
        return None

    face_encoding = face_encodings[0]
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    # Use stricter tolerance for attendance (0.45)
    best_match_idx = np.argmin(face_distances)
    best_match_distance = face_distances[best_match_idx]

    # Only match if distance is less than 0.45 (stricter than 0.6)
    if best_match_distance < 0.45:
        return {
            'employee_id': known_ids[best_match_idx],
            'emp_name': known_names[best_match_idx]
        }

    return None

# ---------------- Attendance Logic ----------------
def hours_diff(start: Optional[datetime], end: Optional[datetime]) -> float:
    """Return difference in hours."""
    if not start or end is None:
        return 0.0
    return round((end - start).total_seconds() / 3600, 2)


def get_company_settings(conn):
    """Get company settings"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company_settings WHERE id = 1")
    settings = cursor.fetchone()
    cursor.close()
    return dict(settings) if settings else None


def is_holiday(check_date, conn):
    """Check if date is a holiday"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM holidays WHERE holiday_date = ?", (str(check_date),))
    holiday = cursor.fetchone()
    cursor.close()
    return dict(holiday) if holiday else None


def is_sunday(check_date):
    """Check if date is Sunday"""
    return check_date.weekday() == 6


def parse_time(time_str):
    """Parse time string to time object"""
    if isinstance(time_str, str):
        parts = time_str.split(':')
        return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    return time_str


def calculate_smart_payroll(start_time, end_time, lunch_start, lunch_end, rate, shift_hours, check_date, conn):
    """Calculate payroll with smart OT system"""
    settings = get_company_settings(conn)

    # Parse office times
    office_start_time = parse_time(settings['office_start_time'])
    office_end_time = parse_time(settings['office_end_time'])

    office_start = datetime.combine(check_date, office_start_time)
    office_end = datetime.combine(check_date, office_end_time)
    ot_multiplier = settings['ot_rate_multiplier']

    # Get break configuration
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM break_types")
    breaks = [dict(row) for row in cursor.fetchall()]
    cursor.close()

    # Calculate total paid break time
    total_paid_break_minutes = sum(b['duration_minutes'] for b in breaks if b['is_paid'])
    total_paid_break_hours = total_paid_break_minutes / 60.0

    # Check if holiday or Sunday
    is_holiday_today = is_holiday(check_date, conn) is not None
    is_sunday_today = is_sunday(check_date)
    sunday_ot_enabled = bool(settings['sunday_auto_ot'])

    # Calculate break taken
    break_taken_hours = hours_diff(lunch_start, lunch_end)
    break_taken_minutes = break_taken_hours * 60

    # Calculate excess break
    excess_break_minutes = max(0, break_taken_minutes - total_paid_break_minutes)
    excess_break_hours = excess_break_minutes / 60.0

    # Break deduction
    break_deduction = 0.0
    if excess_break_minutes > 0:
        deduct_rate = breaks[0]['deduct_rate_per_minute'] if breaks else (rate / 60.0)
        break_deduction = round(excess_break_minutes * deduct_rate, 2)

    # Total work time
    total_work = hours_diff(start_time, end_time)

    # Net hours
    if break_taken_hours <= total_paid_break_hours:
        net_hours = round(total_work, 2)
    else:
        net_hours = round(total_work - excess_break_hours, 2)

    # If Sunday or Holiday - all OT
    if is_holiday_today or (is_sunday_today and sunday_ot_enabled):
        total_pay = round(net_hours * rate * ot_multiplier - break_deduction, 2)
        return {
            'total_hours': net_hours,
            'break_hours': break_taken_hours,
            'paid_break_hours': total_paid_break_hours,
            'excess_break_hours': round(excess_break_hours, 2),
            'overtime_hours': net_hours,
            'regular_hours': 0.0,
            'before_office_ot': 0.0,
            'after_office_ot': 0.0,
            'total_pay': max(0, total_pay),
            'break_deduction': break_deduction,
            'is_holiday': is_holiday_today,
            'is_sunday': is_sunday_today
        }

    # Normal day - calculate OT
    before_office_ot = 0.0
    if start_time < office_start:
        before_office_ot = hours_diff(start_time, min(office_start, end_time))

    after_office_ot = 0.0
    if end_time > office_end:
        after_office_ot = hours_diff(max(office_end, start_time), end_time)

    total_ot = round(before_office_ot + after_office_ot, 2)
    regular_hours = round(net_hours - total_ot, 2)

    regular_pay = regular_hours * rate
    ot_pay = total_ot * rate * ot_multiplier
    total_pay = round(regular_pay + ot_pay - break_deduction, 2)

    return {
        'total_hours': net_hours,
        'break_hours': break_taken_hours,
        'paid_break_hours': total_paid_break_hours,
        'excess_break_hours': round(excess_break_hours, 2),
        'overtime_hours': total_ot,
        'regular_hours': regular_hours,
        'before_office_ot': round(before_office_ot, 2),
        'after_office_ot': round(after_office_ot, 2),
        'total_pay': max(0, total_pay),
        'break_deduction': break_deduction,
        'is_holiday': False,
        'is_sunday': False
    }


def get_today_record(employee_id):
    """Get today's attendance record"""
    conn = get_db()
    if not conn:
        return None

    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM attendance_records WHERE employee_id = ? AND date = ?",
        (employee_id, str(date.today()))
    )
    rec = cursor.fetchone()
    cursor.close()
    conn.close()
    return dict(rec) if rec else None


def determine_action(rec):
    """Determine what action to take"""
    if not rec:
        return "CHECK_IN"
    if rec.get("start_time") and not rec.get("lunch_start"):
        return "LUNCH_START"
    if rec.get("lunch_start") and not rec.get("lunch_end"):
        return "RESUME"
    if rec.get("lunch_end") and not rec.get("end_time"):
        return "CHECK_OUT"
    if rec.get("end_time"):
        return "COMPLETED"
    return "CHECK_IN"


def perform_action(employee_id, emp_name, action, emp_info):
    """Perform attendance action"""
    conn = get_db()
    if not conn:
        return {"success": False, "message": "Database connection failed"}

    cursor = conn.cursor()
    now = now_ist()

    try:
        if action == "CHECK_IN":
            cursor.execute("""
                INSERT OR REPLACE INTO attendance_records
                (employee_id, emp_name, date, start_time, pay_type, rate, shift_hours, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (employee_id, emp_name, str(date.today()), str(now),
                  emp_info['pay_type'], emp_info['rate'], emp_info['shift_hours'], str(now)))
            conn.commit()
            return {
                "success": True,
                "action": "CHECK_IN",
                "message": f"Good morning {emp_name}! Checked in at {now.strftime('%H:%M:%S')}",
                "time": now.strftime('%H:%M:%S')
            }

        elif action == "LUNCH_START":
            cursor.execute("""
                           UPDATE attendance_records
                           SET lunch_start  = ?,
                               last_updated = ?
                           WHERE employee_id = ? AND date = ?
                           """, (str(now), str(now), employee_id, str(date.today())))
            conn.commit()
            return {
                "success": True,
                "action": "LUNCH_START",
                "message": f"{emp_name}, enjoy your lunch! Started at {now.strftime('%H:%M:%S')}",
                "time": now.strftime('%H:%M:%S')
            }

        elif action == "RESUME":
            cursor.execute("""
                           UPDATE attendance_records
                           SET lunch_end    = ?,
                               last_updated = ?
                           WHERE employee_id = ? AND date = ?
                           """, (str(now), str(now), employee_id, str(date.today())))
            conn.commit()
            return {
                "success": True,
                "action": "RESUME",
                "message": f"Welcome back {emp_name}! Resumed at {now.strftime('%H:%M:%S')}",
                "time": now.strftime('%H:%M:%S')
            }

        elif action == "CHECK_OUT":
            rec = get_today_record(employee_id)
            start_time = datetime.fromisoformat(rec['start_time']) if rec.get('start_time') else None
            lunch_start = datetime.fromisoformat(rec['lunch_start']) if rec.get('lunch_start') else None
            lunch_end = datetime.fromisoformat(rec['lunch_end']) if rec.get('lunch_end') else None

            # Use smart payroll calculation
            payroll = calculate_smart_payroll(
                start_time, now, lunch_start, lunch_end,
                emp_info['rate'], emp_info['shift_hours'],
                date.today(), conn
            )

            # Get pending advances
            cursor.execute("""
                           SELECT SUM(remaining_balance) as total_advance
                           FROM advance_payments
                           WHERE employee_id = ?
                             AND status = 'pending'
                           """, (employee_id,))
            advance_result = cursor.fetchone()
            total_advance = advance_result['total_advance'] if advance_result and advance_result[
                'total_advance'] else 0.0
            net_pay = round(payroll['total_pay'] - total_advance, 2)

            cursor.execute("""
                           UPDATE attendance_records
                           SET end_time       = ?,
                               total_hours    = ?,
                               break_hours    = ?,
                               overtime_hours = ?,
                               total_pay      = ?,
                               last_updated   = ?
                           WHERE employee_id = ? AND date = ?
                           """, (str(now), payroll['total_hours'], payroll['break_hours'],
                                 payroll['overtime_hours'], payroll['total_pay'], str(now), employee_id,
                                 str(date.today())))
            conn.commit()

            # Auto-deduct advances
            if total_advance > 0:
                cursor.execute("""
                               UPDATE advance_payments
                               SET remaining_balance = 0,
                                   status            = 'fully_deducted'
                               WHERE employee_id = ?
                                 AND status = 'pending'
                               """, (employee_id,))
                conn.commit()

            response_msg = f"Goodbye {emp_name}! Checked out at {now.strftime('%H:%M:%S')}"
            if payroll.get('is_sunday'):
                response_msg += " (Sunday - OT Applied)"
            elif payroll.get('is_holiday'):
                response_msg += " (Holiday - OT Applied)"

            return {
                "success": True,
                "action": "CHECK_OUT",
                "message": response_msg,
                "time": now.strftime('%H:%M:%S'),
                "hours": payroll['total_hours'],
                "break": payroll['break_hours'],
                "paid_break": payroll.get('paid_break_hours', 0),
                "excess_break": payroll.get('excess_break_hours', 0),
                "overtime": payroll['overtime_hours'],
                "before_office_ot": payroll.get('before_office_ot', 0),
                "after_office_ot": payroll.get('after_office_ot', 0),
                "gross_pay": payroll['total_pay'],
                "break_deduction": payroll.get('break_deduction', 0),
                "advance_deducted": total_advance,
                "net_pay": net_pay,
                "is_sunday": payroll.get('is_sunday', False),
                "is_holiday": payroll.get('is_holiday', False)
            }

        elif action == "COMPLETED":
            return {
                "success": True,
                "action": "COMPLETED",
                "message": f"{emp_name} has already completed today's shift!",
                "time": now.strftime('%H:%M:%S')
            }

    except Exception as e:
        print(f"Action error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}
    finally:
        cursor.close()
        conn.close()


# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')


@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/employees')
def employees_page():
    return render_template('employees.html')


@app.route('/advances')
def advances_page():
    return render_template('advances.html')


@app.route('/settings')
def settings_page():
    return render_template('settings.html')


@app.route('/payslip')
def payslip_page():
    return render_template('payslip.html')


# ---------------- API ENDPOINTS ----------------

@app.route('/api/register', methods=['POST'])
def api_register():
    """Register new employee with face"""
    data = request.json

    if not all(k in data for k in ['employee_id', 'emp_name', 'pay_type', 'rate', 'shift_hours', 'image']):
        return jsonify({"success": False, "message": "Missing required fields"})

    employee_id = data['employee_id'].strip()
    emp_name = data['emp_name'].strip()
    pay_type = data['pay_type'].lower()

    try:
        rate = float(data['rate'])
        shift_hours = float(data['shift_hours'])
    except ValueError:
        return jsonify({"success": False, "message": "Invalid rate or shift hours"})

    # Check if employee ID already exists
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id FROM employee_faces WHERE employee_id = ?", (employee_id,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": f"Employee ID '{employee_id}' already exists!"})
    cursor.close()
    conn.close()

    # Convert image and detect face
    image = base64_to_image(data['image'])
    if image is None:
        return jsonify({"success": False, "message": "Invalid image"})

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return jsonify({"success": False, "message": "No face detected! Please try again."})

    if len(face_locations) > 1:
        return jsonify(
            {"success": False, "message": "Multiple faces detected! Only one person should be in the frame."})

    face_encodings = face_recognition.face_encodings(image, face_locations)
    if not face_encodings:
        return jsonify({"success": False, "message": "Could not encode face"})

    face_encoding = face_encodings[0]

    # ANTI-SCAM: Check for duplicate face
    duplicate = check_duplicate_face(face_encoding)
    if duplicate:
        return jsonify({
            "success": False,
            "message": f"🚫 SCAM ALERT! This face is already registered as '{duplicate['emp_name']}' (ID: {duplicate['employee_id']}). Don't try to scam the system!"
        })

    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    encoding_blob = pickle.dumps(face_encoding)

    try:
        cursor.execute("""
                       INSERT INTO employee_faces
                       (employee_id, emp_name, face_encoding, pay_type, rate, shift_hours, registered_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       """, (employee_id, emp_name, encoding_blob, pay_type, rate, shift_hours, str(now_ist())))
        conn.commit()
        return jsonify({
            "success": True,
            "message": f"✅ {emp_name} registered successfully!",
            "employee_id": employee_id,
            "emp_name": emp_name
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Database error: {str(e)}"})
    finally:
        cursor.close()
        conn.close()


@app.route('/api/mark_attendance', methods=['POST'])
def api_mark_attendance():
    """Mark attendance using face recognition"""
    data = request.json

    if 'image' not in data:
        return jsonify({"success": False, "message": "No image provided"})

    image = base64_to_image(data['image'])
    if image is None:
        return jsonify({"success": False, "message": "Invalid image"})

    known_encodings, known_ids, known_names = load_known_faces()

    if not known_encodings:
        return jsonify({"success": False, "message": "No employees registered yet!"})

    result = recognize_face_from_image(image, known_encodings, known_ids, known_names)

    if not result:
        return jsonify({"success": False, "message": "Face not recognized! Please register first."})

    employee_id = result['employee_id']
    emp_name = result['emp_name']

    # Get employee info
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employee_faces WHERE employee_id = ?", (employee_id,))
    emp_info = dict(cursor.fetchone())
    cursor.close()
    conn.close()

    rec = get_today_record(employee_id)
    action = determine_action(rec)

    response = perform_action(employee_id, emp_name, action, emp_info)
    return jsonify(response)


@app.route('/api/today_records')
def api_today_records():
    """Get today's attendance records"""
    conn = get_db()
    if not conn:
        return jsonify({"success": False, "message": "Database error"})

    cursor = conn.cursor()
    cursor.execute("""
                   SELECT *
                   FROM attendance_records
                   WHERE date = ?
                   ORDER BY emp_name
                   """, (str(date.today()),))
    records = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return jsonify({"success": True, "records": records})


@app.route('/api/employees')
def api_employees():
    """Get all employees"""
    conn = get_db()
    if not conn:
        return jsonify({"success": False, "message": "Database error"})

    cursor = conn.cursor()
    cursor.execute("""
                   SELECT employee_id, emp_name, pay_type, rate, shift_hours, registered_date
                   FROM employee_faces
                   ORDER BY emp_name
                   """)
    employees = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return jsonify({"success": True, "employees": employees})


@app.route('/api/delete_record', methods=['POST'])
def api_delete_record():
    """Delete today's record for an employee"""
    data = request.json
    employee_id = data.get('employee_id')

    if not employee_id:
        return jsonify({"success": False, "message": "Employee ID required"})

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
                   DELETE
                   FROM attendance_records
                   WHERE employee_id = ? AND date = ?
                   """, (employee_id, str(date.today())))
    affected = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()

    if affected > 0:
        return jsonify({"success": True, "message": f"Deleted {affected} record(s)"})
    else:
        return jsonify({"success": False, "message": "No record found for today"})


@app.route('/api/export_csv')
def api_export_csv():
    """Export today's records to CSV"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT emp_name,
                          employee_id, date, start_time, lunch_start, lunch_end, end_time, total_hours, break_hours, overtime_hours, pay_type, rate, shift_hours, total_pay
                   FROM attendance_records
                   WHERE date = ?
                   ORDER BY emp_name
                   """, (str(date.today()),))
    records = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    if not records:
        return jsonify({"success": False, "message": "No records to export"})

    if not os.path.exists('static'):
        os.makedirs('static')

    filename = f"attendance_{date.today().isoformat()}.csv"
    filepath = os.path.join('static', filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    return send_file(filepath, as_attachment=True)


@app.route('/api/give_advance', methods=['POST'])
def api_give_advance():
    """Record advance payment to employee"""
    data = request.json

    required_fields = ['employee_id', 'amount', 'payment_mode']
    if not all(k in data for k in required_fields):
        return jsonify({"success": False, "message": "Missing required fields"})

    employee_id = data['employee_id']
    amount = float(data['amount'])
    payment_mode = data['payment_mode']
    reference_no = data.get('reference_no', '')
    reason = data.get('reason', '')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_name FROM employee_faces WHERE employee_id = ?", (employee_id,))
    emp = cursor.fetchone()

    if not emp:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": "Employee not found"})

    emp_name = emp['emp_name']

    try:
        cursor.execute("""
                       INSERT INTO advance_payments
                       (employee_id, emp_name, amount, date, payment_mode, reference_no, reason,
                        status, remaining_balance, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                       """, (employee_id, emp_name, amount, str(date.today()), payment_mode,
                             reference_no, reason, amount, str(now_ist())))
        conn.commit()

        return jsonify({
            "success": True,
            "message": f"Advance of ₹{amount} given to {emp_name}",
            "employee_name": emp_name
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        cursor.close()
        conn.close()


@app.route('/api/employee_advances/<employee_id>')
def api_employee_advances(employee_id):
    """Get all advances for a specific employee"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT *
                   FROM advance_payments
                   WHERE employee_id = ?
                   ORDER BY date DESC
                   """, (employee_id,))
    advances = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    total_pending = sum(adv['remaining_balance'] for adv in advances if adv['status'] == 'pending')

    return jsonify({
        "success": True,
        "advances": advances,
        "total_pending": total_pending
    })


@app.route('/api/all_advances')
def api_all_advances():
    """Get all advance payments"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM advance_payments ORDER BY date DESC")
    advances = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return jsonify({"success": True, "advances": advances})


@app.route('/api/deduct_advance', methods=['POST'])
def api_deduct_advance():
    """Manually deduct advance from employee"""
    data = request.json
    advance_id = data.get('advance_id')
    deduct_amount = float(data.get('deduct_amount', 0))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM advance_payments WHERE id = ?", (advance_id,))
    advance = cursor.fetchone()

    if not advance:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": "Advance not found"})

    advance = dict(advance)
    new_balance = advance['remaining_balance'] - deduct_amount
    new_status = 'fully_deducted' if new_balance <= 0 else 'pending'

    cursor.execute("""
                   UPDATE advance_payments
                   SET remaining_balance = ?,
                       status            = ?
                   WHERE id = ?
                   """, (max(0, new_balance), new_status, advance_id))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Deducted ₹{deduct_amount}",
        "remaining_balance": max(0, new_balance)
    })


@app.route('/api/get_settings')
def api_get_settings():
    """Get company settings"""
    conn = get_db()
    settings = get_company_settings(conn)
    conn.close()

    return jsonify({"success": True, "settings": settings})


@app.route('/api/update_settings', methods=['POST'])
def api_update_settings():
    """Update company settings"""
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
                       UPDATE company_settings
                       SET office_start_time  = ?,
                           office_end_time    = ?,
                           ot_rate_multiplier = ?,
                           sunday_auto_ot     = ?
                       WHERE id = 1
                       """, (data['office_start_time'], data['office_end_time'],
                             data['ot_rate_multiplier'], 1 if data['sunday_auto_ot'] else 0))
        conn.commit()
        return jsonify({"success": True, "message": "Settings updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        cursor.close()
        conn.close()


@app.route('/api/get_breaks')
def api_get_breaks():
    """Get break types"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM break_types ORDER BY id")
    breaks = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return jsonify({"success": True, "breaks": breaks})


@app.route('/api/add_break', methods=['POST'])
def api_add_break():
    """Add break type"""
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
                       INSERT INTO break_types
                           (break_name, duration_minutes, is_paid, deduct_rate_per_minute)
                       VALUES (?, ?, ?, ?)
                       """, (data['break_name'], data['duration_minutes'],
                             1 if data['is_paid'] else 0, data.get('deduct_rate_per_minute', 0)))
        conn.commit()
        return jsonify({"success": True, "message": "Break type added"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        cursor.close()
        conn.close()


@app.route('/api/delete_break/<int:break_id>', methods=['DELETE'])
def api_delete_break(break_id):
    """Delete break type"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM break_types WHERE id = ?", (break_id,))
    affected = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()

    if affected > 0:
        return jsonify({"success": True, "message": "Break deleted"})
    return jsonify({"success": False, "message": "Break not found"})


@app.route('/api/get_holidays')
def api_get_holidays():
    """Get all holidays"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM holidays ORDER BY holiday_date")
    holidays = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return jsonify({"success": True, "holidays": holidays})


@app.route('/api/add_holiday', methods=['POST'])
def api_add_holiday():
    """Add holiday"""
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
                       INSERT INTO holidays (holiday_date, holiday_name, is_ot_applicable)
                       VALUES (?, ?, ?)
                       """,
                       (data['holiday_date'], data['holiday_name'], 1 if data.get('is_ot_applicable', True) else 0))
        conn.commit()
        return jsonify({"success": True, "message": "Holiday added"})
    except Exception as e:
        if 'UNIQUE' in str(e):
            return jsonify({"success": False, "message": "Holiday already exists for this date"})
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        cursor.close()
        conn.close()


@app.route('/api/delete_holiday/<int:holiday_id>', methods=['DELETE'])
def api_delete_holiday(holiday_id):
    """Delete holiday"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM holidays WHERE id = ?", (holiday_id,))
    affected = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()

    if affected > 0:
        return jsonify({"success": True, "message": "Holiday deleted"})
    return jsonify({"success": False, "message": "Holiday not found"})


@app.route('/api/export_csv_range', methods=['POST'])
def api_export_csv_range():
    """Export records for date range"""
    data = request.json
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT emp_name,
                          employee_id, date, start_time, lunch_start, lunch_end, end_time, total_hours, break_hours, overtime_hours, pay_type, rate, shift_hours, total_pay
                   FROM attendance_records
                   WHERE date BETWEEN ? AND ?
                   ORDER BY date, emp_name
                   """, (from_date, to_date))
    records = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    if not records:
        return jsonify({"success": False, "message": "No records in this date range"})

    if not os.path.exists('static'):
        os.makedirs('static')

    filename = f"attendance_{from_date}_to_{to_date}.csv"
    filepath = os.path.join('static', filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    return send_file(filepath, as_attachment=True)


@app.route('/api/attendance_records', methods=['POST'])
def api_attendance_records():
    """Get attendance records for specific employee and date range"""
    data = request.json
    employee_id = data.get('employee_id')
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT *
                   FROM attendance_records
                   WHERE employee_id = ? AND date BETWEEN ? AND ?
                   ORDER BY date
                   """, (employee_id, from_date, to_date))
    records = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return jsonify({"success": True, "records": records})

@app.route('/api/mark_attendance_manual', methods=['POST'])
def api_mark_attendance_manual():
    """Mark attendance manually (in case scanner fails)"""
    data = request.json
    employee_id = data.get('employee_id')
    action = data.get('action')
    time_str = data.get('time')

    if not all([employee_id, action, time_str]):
        return jsonify({"success": False, "message": "Missing required fields"})

    conn = get_db()
    if not conn:
        return jsonify({"success": False, "message": "Database error"})

    cursor = conn.cursor()

    # Get employee info
    cursor.execute("SELECT * FROM employee_faces WHERE employee_id = ?", (employee_id,))
    emp = cursor.fetchone()

    if not emp:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": "Employee not found"})

    emp_info = dict(emp)
    emp_name = emp_info['emp_name']

    # Convert time string to datetime
    try:
        time_obj = datetime.strptime(time_str, '%H:%M').time()
        now = now_ist().replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
    except:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": "Invalid time format"})

    try:
        if action == "CHECK_IN":
            cursor.execute("""
                INSERT OR REPLACE INTO attendance_records
                (employee_id, emp_name, date, start_time, pay_type, rate, shift_hours, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (employee_id, emp_name, str(date.today()), str(now),
                  emp_info['pay_type'], emp_info['rate'], emp_info['shift_hours'], str(now_ist())))
            conn.commit()
            msg = f"✅ Manual Check-in recorded for {emp_name} at {time_str}"

        elif action == "LUNCH_START":
            cursor.execute("""
                UPDATE attendance_records
                SET lunch_start = ?, last_updated = ?
                WHERE employee_id = ? AND date = ?
            """, (str(now), str(now_ist()), employee_id, str(date.today())))
            conn.commit()
            msg = f"✅ Lunch start recorded for {emp_name} at {time_str}"

        elif action == "RESUME":
            cursor.execute("""
                UPDATE attendance_records
                SET lunch_end = ?, last_updated = ?
                WHERE employee_id = ? AND date = ?
            """, (str(now), str(now_ist()), employee_id, str(date.today())))
            conn.commit()
            msg = f"✅ Lunch end recorded for {emp_name} at {time_str}"

        elif action == "CHECK_OUT":
            rec = get_today_record(employee_id)
            if not rec:
                cursor.close()
                conn.close()
                return jsonify({"success": False, "message": "No check-in record found for today"})

            start_time = datetime.fromisoformat(rec['start_time']) if rec.get('start_time') else None
            lunch_start = datetime.fromisoformat(rec['lunch_start']) if rec.get('lunch_start') else None
            lunch_end = datetime.fromisoformat(rec['lunch_end']) if rec.get('lunch_end') else None

            payroll = calculate_smart_payroll(
                start_time, now, lunch_start, lunch_end,
                emp_info['rate'], emp_info['shift_hours'],
                date.today(), conn
            )

            cursor.execute("""
                UPDATE attendance_records
                SET end_time = ?, total_hours = ?, break_hours = ?,
                    overtime_hours = ?, total_pay = ?, last_updated = ?
                WHERE employee_id = ? AND date = ?
            """, (str(now), payroll['total_hours'], payroll['break_hours'],
                  payroll['overtime_hours'], payroll['total_pay'], str(now_ist()),
                  employee_id, str(date.today())))
            conn.commit()
            msg = f"✅ Manual Check-out recorded for {emp_name} at {time_str}"

        cursor.close()
        conn.close()
        return jsonify({"success": True, "message": msg})

    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@app.route('/api/update_employee', methods=['POST'])
def api_update_employee():
    """Update employee details (salary, shift hours, pay type)"""
    data = request.json
    employee_id = data.get('employee_id')

    if not employee_id:
        return jsonify({"success": False, "message": "Employee ID required"})

    conn = get_db()
    if not conn:
        return jsonify({"success": False, "message": "Database error"})

    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE employee_faces
            SET pay_type = ?, rate = ?, shift_hours = ?
            WHERE employee_id = ?
        """, (data.get('pay_type'), float(data.get('rate', 0)),
              float(data.get('shift_hours', 0)), employee_id))

        conn.commit()

        cursor.execute("SELECT emp_name FROM employee_faces WHERE employee_id = ?", (employee_id,))
        emp = cursor.fetchone()

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"✅ {emp['emp_name']} details updated successfully!"
        })

    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
# ---------------- MAIN ----------------
if __name__ == '__main__':
    print("=" * 60)
    print("  🎭 FACE RECOGNITION ATTENDANCE (SQLite)")
    print("=" * 60)
    print("\n📋 Initializing database...")
    init_db()
    print("\n🚀 Starting server...")
    print(f"\n📱 Access from your phone:")
    print(f"   http://172.30.71.193:5000")
    print(f"\n💻 Access from this PC:")
    print(f"   http://localhost:5000")
    print(f"\n⚠️  Make sure phone is on SAME WiFi!")
    print(f"\n🛑 Press Ctrl+C to stop server\n")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        log_error(e)
        raise
