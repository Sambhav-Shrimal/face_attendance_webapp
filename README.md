# 🎭 Face Recognition Attendance System v2.0

Automated attendance system using face recognition with **Advance Payment Management**. Employees mark attendance by showing their face to the camera from any phone/device browser.

## 📋 Features

- ✅ **Automated Attendance**: Check-in → Lunch → Resume → Check-out
- ✅ **Anti-Scam Protection**: Prevents duplicate face registration
- ✅ **Phone Camera Support**: Works with any smartphone browser
- ✅ **Payroll Calculation**: Automatic hourly/daily pay calculation
- ✅ **💰 Advance Payment Tracking**: Record and auto-deduct advances
- ✅ **Real-time Dashboard**: View today's attendance instantly
- ✅ **CSV Export**: Export attendance records
- ✅ **No App Installation**: Just a web browser needed

## 🆕 NEW: Advance Payment System

### Features:
- 💸 **Give Advance**: Record advance payments to employees
- 💳 **Multiple Payment Modes**: Cash, UPI, Bank Transfer, Cheque, etc.
- 📊 **Track Balance**: See pending advances per employee
- 🔄 **Auto Deduction**: Advances automatically deducted on checkout
- 📈 **Complete History**: View all advance transactions
- 💰 **Net Pay Calculation**: Gross Pay - Advances = Net Pay

### How It Works:
1. **Admin gives advance** to employee (e.g., ₹5,000)
2. System records: Amount, Date, Payment Mode, Reference
3. **On checkout**, system calculates:
   ```
   Gross Pay: ₹10,000
   Advance Deducted: -₹5,000
   Net Pay: ₹5,000
   ```
4. Advance status changes to "fully_deducted"

## 🛠️ Setup Instructions

### Prerequisites

1. **Python 3.11 or 3.14** installed
2. **MySQL** installed and running
3. **Same WiFi network** for PC and phone

### Step 1: Install Dependencies

```powershell
# Activate virtual environment (if using)
.\.venv\Scripts\Activate.ps1

# Install all packages
pip install -r requirements.txt
```

### Step 2: Configure Database

Edit `app.py` and update MySQL password:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD",  # Change this
    "database": "attendance_system"
}
```

### Step 3: Create Folders

Create these folders in the project directory:

```
mkdir templates
mkdir static
```

### Step 4: Place Files

```
face_attendance_web/
├── app.py
├── templates/
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   ├── dashboard.html
│   └── employees.html
├── static/
├── requirements.txt
├── START_SERVER.bat
└── README.md
```

### Step 5: Start Server

**Windows:**
```powershell
# Double-click START_SERVER.bat
# OR run manually:
python app.py
```

**Mac/Linux:**
```bash
python app.py
```

### Step 6: Find Your PC's IP Address

**Windows:**
```powershell
ipconfig
```
Look for **IPv4 Address** (e.g., `192.168.1.5` or `172.30.71.193`)

**Mac/Linux:**
```bash
ifconfig
```

### Step 7: Access from Phone

1. Connect phone to **SAME WiFi** as PC
2. Open browser (Chrome/Safari)
3. Go to: `http://YOUR_PC_IP:5000`
4. Example: `http://172.30.71.193:5000`

## 📱 Usage Guide

### Register Employee (First Time)

1. Open: `http://YOUR_PC_IP:5000/register`
2. Fill employee details
3. Click "Start Camera"
4. Position face in frame
5. Click "Capture Face"
6. Click "Register Employee"

**Anti-Scam Feature**: System prevents registering the same face twice!

### Mark Attendance (Daily)

1. Open: `http://YOUR_PC_IP:5000/attendance`
2. Click "Scan Face"
3. System automatically detects:
   - **1st scan today** → Check-in
   - **2nd scan** → Lunch start
   - **3rd scan** → Resume work
   - **4th scan** → Check-out (with pay calculation)

### View Dashboard

1. Open: `http://YOUR_PC_IP:5000/dashboard`
2. See all today's attendance
3. Export to CSV
4. Delete records if needed

### View All Employees

1. Open: `http://YOUR_PC_IP:5000/employees`
2. See all registered employees with details

## 🔧 Troubleshooting

### Camera Not Working

- **Allow camera permissions** in browser
- Try using **Chrome** or **Safari**
- Check if another app is using camera

### Can't Connect from Phone

- ✅ Both devices on **same WiFi**?
- ✅ Check **firewall** (temporarily disable Windows Firewall)
- ✅ Correct **IP address**?
- ✅ Server **running**?

### Database Error

- ✅ MySQL **running**?
- ✅ Correct **password** in `app.py`?
- ✅ Check MySQL service: `services.msc` (Windows)

### Face Not Recognized

- ✅ Good lighting
- ✅ Face clearly visible
- ✅ Look at camera directly
- ✅ Employee registered first?

## 📊 Database Structure

### employee_faces
- `employee_id` (Primary Key)
- `emp_name`
- `face_encoding` (BLOB)
- `pay_type`
- `rate`
- `shift_hours`
- `registered_date`

### attendance_records
- `employee_id`
- `emp_name`
- `date`
- `start_time`
- `lunch_start`
- `lunch_end`
- `end_time`
- `total_hours`
- `break_hours`
- `overtime_hours`
- `total_pay`

## 🚀 Deployment Tips

### For Factory Use:

1. **Dedicated PC** at entrance
2. **Mount phone/tablet** at eye level
3. **Good lighting** at scanning area
4. **Keep attendance page open** all day
5. **Backup database** regularly

### For Testing at Home:

1. Run server on laptop
2. Use your phone to test
3. Register yourself first
4. Test all 4 scans (check-in, lunch, resume, checkout)

## 📝 Notes

- **Tolerance**: Face matching tolerance is set to `0.6` (adjustable in code)
- **Overtime**: Automatically calculated as 1.5x rate
- **Daily pay**: Fixed amount regardless of hours
- **Data storage**: All data stored locally in MySQL

## 🆘 Support

For issues or questions, check:
1. All files in correct folders
2. Virtual environment activated
3. All packages installed
4. MySQL running
5. Correct IP address

## 📄 License

Free to use for personal and commercial projects.

---

**Built with ❤️ using Python Flask + Face Recognition**