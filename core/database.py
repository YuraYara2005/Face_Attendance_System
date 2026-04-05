import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'attendance.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            role        TEXT DEFAULT 'Student',
            image_path  TEXT,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL,
            person_name TEXT NOT NULL,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL,
            status      TEXT DEFAULT 'Present',
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
    ''')

    conn.commit()
    conn.close()


def register_person(name: str, role: str, image_path: str) -> int:
    """Insert a new person and return their ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO persons (name, role, image_path) VALUES (?, ?, ?)',
        (name, role, image_path)
    )
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id


def get_all_persons():
    """Return all registered persons."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM persons ORDER BY name')
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_person(person_id: int):
    """Remove a person and their attendance records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance WHERE person_id = ?', (person_id,))
    cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
    conn.commit()
    conn.close()


def mark_attendance(person_id: int, person_name: str) -> bool:
    """
    Mark attendance only once per person per day.
    Returns True if marked, False if already marked today.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    now_time = datetime.now().strftime('%H:%M:%S')

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        'SELECT id FROM attendance WHERE person_id = ? AND date = ?',
        (person_id, today)
    )
    if cursor.fetchone():
        conn.close()
        return False  # already marked

    cursor.execute(
        'INSERT INTO attendance (person_id, person_name, date, time) VALUES (?, ?, ?, ?)',
        (person_id, person_name, today, now_time)
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_report(date_filter: str = None):
    """Return attendance records, optionally filtered by date."""
    conn = get_connection()
    cursor = conn.cursor()
    if date_filter:
        cursor.execute(
            'SELECT * FROM attendance WHERE date = ? ORDER BY time DESC',
            (date_filter,)
        )
    else:
        cursor.execute('SELECT * FROM attendance ORDER BY date DESC, time DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_today_count():
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM attendance WHERE date = ?', (today,))
    count = cursor.fetchone()[0]
    conn.close()
    return count
