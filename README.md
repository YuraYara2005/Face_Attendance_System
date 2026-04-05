# Face Attendance System

A desktop application that uses your webcam and facial recognition to automatically mark attendance and export reports to Excel.

---

## 📁 Folder Structure

```
face_attendance/
│
├── app.py                    ← Main entry point. Run this to launch the app.
│
├── requirements.txt          ← All Python packages needed
│
├── core/
│   ├── database.py           ← All database operations (SQLite)
│   └── face_engine.py        ← All face detection & recognition logic
│
├── utils/
│   └── exporter.py           ← Excel report generator
│
└── data/
    ├── known_faces/          ← Photos of registered people are stored here
    ├── attendance_logs/      ← Exported Excel files go here
    ├── attendance.db         ← SQLite database (auto-created on first run)
    └── encodings.pkl         ← Saved face encodings (auto-created on first run)
```

---

## 🛠️ Technologies Used

| Library | What it does in this project |
|---|---|
| `face_recognition` | Detects faces in images/video and creates a unique 128-number fingerprint (encoding) for each face |
| `opencv-python` | Captures webcam video frames, draws bounding boxes and labels on detected faces |
| `customtkinter` | Builds the modern dark-theme desktop UI (buttons, tabs, scrollable frames) |
| `sqlite3` | Built-in Python database — stores person info and attendance records |
| `pandas` | Converts attendance data to structured tables for export |
| `openpyxl` | Creates formatted Excel (.xlsx) files with styled headers |
| `Pillow (PIL)` | Converts between OpenCV image format and what Tkinter can display |
| `numpy` | Handles the math on face encoding arrays |
| `pickle` | Saves/loads face encodings to disk so you don't re-encode on every startup |

---

## ⚙️ Installation (Step by Step)

### Step 1 — Prerequisites
Make sure you have Python 3.9 or higher installed.
```
python --version
```

### Step 2 — Install cmake and dlib (required by face_recognition)

**On Windows:**
```
pip install cmake
pip install dlib
```
If dlib fails on Windows, download a prebuilt wheel from:
https://github.com/z-mahmud22/Dlib_Windows_Python3.x
Then install it with: `pip install dlib‑19.xx.x‑cp3x‑cp3x‑win_amd64.whl`

**On Mac:**
```
brew install cmake
pip install dlib
```

**On Linux (Ubuntu/Debian):**
```
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib
```

### Step 3 — Install all requirements
```
cd face_attendance
pip install -r requirements.txt
```

### Step 4 — Run the app
```
python app.py
```

---

## 🚀 How to Use

### Registering a person
1. Click **➕ Register Person** in the sidebar
2. Enter the person's full name and their role (Student, Doctor, Staff, etc.)
3. Click **📂 Choose Photo** and select a clear frontal photo of them
4. Click **✔ Register Person**
5. The app encodes their face and saves it. Done!

**Tips for good registration photos:**
- Use a clear, well-lit frontal face photo
- No sunglasses, masks, or hats
- One face per photo only

### Taking Attendance
1. Click **📷 Live Attendance** in the sidebar
2. Click **▶ Start Camera**
3. Point the camera at people's faces
4. The app automatically:
   - Detects and recognizes faces
   - Draws a **green box** for known people, **red** for unknowns
   - Marks attendance once per person per day
   - Shows real-time log on the right panel

### Viewing Reports
1. Click **📋 Attendance Log**
2. The default view shows today's records
3. Type any date (YYYY-MM-DD) and click **Filter** to see a specific day
4. Click **All** to see all records
5. Click **⬇ Export Excel** to download a formatted Excel report

### Managing People
- Click **👥 Manage People** to see everyone registered
- Use the **Delete** button to remove someone and all their data

---

## 🧠 How Face Recognition Works (Simplified)

1. **Registration**: When you register someone, the app runs their photo through a deep learning model that converts their face into 128 numbers (called an "encoding" or "face fingerprint"). These numbers are saved to disk.

2. **Recognition**: For each camera frame, the app:
   - Detects all faces in the frame
   - Converts each detected face into 128 numbers
   - Compares those numbers to every saved encoding using a distance formula
   - If the distance is below the tolerance threshold (0.50), it's a match

3. **Tolerance**: Lower = stricter (fewer false positives, might miss some). Higher = looser. 0.50 is a good default balance.

---

## 🔧 Common Issues & Fixes

**"No face detected in photo"**
→ Use a clearer, brighter, more frontal photo. The face must be clearly visible.

**"Camera Error"**
→ Make sure no other app (Teams, Zoom) is using the camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` if you have multiple cameras.

**App is slow when recognizing**
→ The app already skips every 3rd frame. You can increase this in `app.py` (change `% 3` to `% 5`).

**dlib fails to install**
→ See Step 2 above for platform-specific instructions.

---

## 💼 For Your Freelance Portfolio

When showing this to clients, highlight:
- Automatic, contactless attendance (no manual input)
- One-click Excel export for HR/admin use
- Works offline (no internet needed after setup)
- One-time marked per day (prevents duplicates)
- Easy to add/remove people at any time
