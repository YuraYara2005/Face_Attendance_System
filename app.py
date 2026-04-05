"""
Face Attendance System — Premium Native Streamlit UI
Run with:  streamlit run app.py
"""

import os, sys, queue, threading
from datetime import datetime
from io import BytesIO

import av
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.database import (
    init_db, register_person, get_all_persons, delete_person,
    mark_attendance, get_attendance_report, get_today_count,
)
from core.face_engine import register_face, remove_face, recognize_faces

KNOWN_FACES_DIR = os.path.join(ROOT, "data", "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FaceAttend | Biometric Dashboard", page_icon="🛡️", layout="wide")

# ── Session state init ────────────────────────────────────────────────────────
if "det_log" not in st.session_state:
    st.session_state.det_log = []
if "camera_was_playing" not in st.session_state:
    st.session_state.camera_was_playing = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Live Attendance"

# ─────────────────────────────────────────────────────────────────────────────
# WebRTC video frame callback (Backend Logic Preserved 100%)
# ─────────────────────────────────────────────────────────────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=40)
_recently_marked: dict    = {}
_mark_lock                 = threading.Lock()

def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    rgb = frame.to_ndarray(format="rgb24")
    try:
        results = recognize_faces(rgb)
    except Exception:
        results = []

    now = datetime.now()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for r in results:
        top, right_x, bottom, left = r["location"]
        is_match = r["matched"]
        color = (110, 231, 183) if is_match else (113, 113, 248)

        cv2.rectangle(bgr, (left, top), (right_x, bottom), color, 2)

        label = f"{r['name']}  {r['confidence']}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        pill_top = max(0, top - th - 14)
        cv2.rectangle(bgr, (left, pill_top), (left + tw + 12, top), color, cv2.FILLED)
        cv2.putText(bgr, label, (left + 6, top - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (13, 15, 26), 1, cv2.LINE_AA)

        if is_match and r["person_id"] is not None:
            pid = r["person_id"]
            with _mark_lock:
                last     = _recently_marked.get(pid)
                too_soon = last and (now - last).seconds < 10
            if not too_soon:
                marked = mark_attendance(pid, r["name"])
                with _mark_lock:
                    _recently_marked[pid] = now
                ev = {"name": r["name"], "matched": True,
                      "marked": marked, "time": now.strftime("%H:%M:%S")}
                try: _event_queue.put_nowait(ev)
                except queue.Full:
                    try: _event_queue.get_nowait()
                    except queue.Empty: pass
                    try: _event_queue.put_nowait(ev)
                    except queue.Full: pass
        else:
            ev = {"name": "Unknown", "matched": False,
                  "marked": False, "time": now.strftime("%H:%M:%S")}
            try: _event_queue.put_nowait(ev)
            except queue.Full: pass

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(out_rgb, format="rgb24")


# ── Excel export helper (Backend Logic Preserved) ─────────────────────────────
def _build_excel_bytes(date_filter=None):
    records = get_attendance_report(date_filter)
    persons = get_all_persons()
    if not records:
        return None
    pm = {p["id"]: p["role"] for p in persons}
    df = pd.DataFrame(records)
    df["role"] = df["person_id"].map(pm).fillna("—")
    df = df[["date","time","person_name","role","status"]]
    df.columns = ["Date","Time","Name","Role","Status"]
    df = df.sort_values(["Date","Time"], ascending=[False,True])

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Professional Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ FaceAttend")
    st.caption("Secure Enterprise Biometrics")
    st.divider()

    # Custom Full-Width Menu Buttons
    if st.button("📷 Live Attendance", use_container_width=True, type="primary" if st.session_state.current_page == "Live Attendance" else "secondary"):
        st.session_state.current_page = "Live Attendance"
        st.rerun()

    if st.button("➕ Register Person", use_container_width=True, type="primary" if st.session_state.current_page == "Register Person" else "secondary"):
        st.session_state.current_page = "Register Person"
        st.rerun()

    if st.button("👥 Manage People", use_container_width=True, type="primary" if st.session_state.current_page == "Manage People" else "secondary"):
        st.session_state.current_page = "Manage People"
        st.rerun()

    if st.button("📋 Attendance Log", use_container_width=True, type="primary" if st.session_state.current_page == "Attendance Log" else "secondary"):
        st.session_state.current_page = "Attendance Log"
        st.rerun()

    st.divider()
    with st.container(border=True):
        st.metric(label="Today's Check-ins", value=get_today_count())


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Attendance
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "Live Attendance":
    st.header("Live Monitoring Terminal")
    st.caption("Real-time video processing and automated attendance logging.")

    col_cam, col_panel = st.columns([2.5, 1], gap="large")

    with col_cam:
        with st.container(border=True):
            ctx = webrtc_streamer(
                key="fa-live",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=_video_frame_callback,
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                              "frameRate": {"ideal": 30, "max": 30}},
                    "audio": False,
                },
                async_processing=True,
                rtc_configuration=RTCConfiguration(iceServers=[
                    {"urls": ["stun:stun.l.google.com:19302"]},
                ])
            )

    with col_panel:
        st.subheader("Activity Log")

        is_playing = bool(ctx.state.playing)

        if is_playing:
            drained = 0
            while drained < 20:
                try:
                    ev = _event_queue.get_nowait()
                    if ev["matched"] and ev["marked"]:
                        badge = ("success", ev["name"], f"✓ {ev['time']}")
                    elif ev["matched"]:
                        badge = ("info",  ev["name"], "Already marked")
                    else:
                        badge = ("error", "Unknown Face",  ev["time"])
                    st.session_state.det_log.insert(0, badge)
                    drained += 1
                except queue.Empty:
                    break
            st.session_state.det_log = st.session_state.det_log[:20]
            st.session_state.camera_was_playing = True

        elif st.session_state.camera_was_playing:
            st.session_state.camera_was_playing = False

        with st.container(border=True, height=450):
            if not st.session_state.det_log:
                st.info("System standby. Start camera to begin.")
            else:
                for badge_type, name, time in st.session_state.det_log:
                    if badge_type == "success":
                        st.success(f"**{name}** logged at {time}", icon="✅")
                    elif badge_type == "info":
                        st.warning(f"**{name}** (Skipped: {time})", icon="ℹ️")
                    else:
                        st.error(f"**{name}** detected at {time}", icon="🚫")

        if is_playing:
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Register Person
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Register Person":
    st.header("Personnel Onboarding")
    st.caption("Enroll a new individual into the biometric database.")

    col_form, col_prev = st.columns([1.2, 1], gap="large")

    with col_form:
        with st.container(border=True):
            with st.form("register_form", clear_on_submit=True):
                st.subheader("Identity Details")
                name = st.text_input("Full Legal Name", placeholder="e.g. Yara Mohamed")
                role = st.selectbox("Designation", ["Student", "Staff", "Doctor", "Administrator", "Other"])
                st.divider()
                st.subheader("Biometric Profile")
                uploaded = st.file_uploader("Upload a clear, well-lit, frontal face photo", type=["jpg","jpeg","png"])

                submit = st.form_submit_button("Enroll Person", type="primary", use_container_width=True)

                if submit:
                    if not name.strip() or not uploaded:
                        st.error("Please provide both a name and a photo.")
                    else:
                        ext  = os.path.splitext(uploaded.name)[1] or ".jpg"
                        safe = name.strip().replace(" ", "_")
                        dest = os.path.join(KNOWN_FACES_DIR, f"{safe}{ext}")
                        with open(dest, "wb") as f:
                            f.write(uploaded.getbuffer())

                        pid = register_person(name.strip(), role, dest)
                        success = register_face(pid, name.strip(), dest)

                        if success:
                            st.success(f"Successfully enrolled **{name.strip()}**!")
                            st.balloons()
                        else:
                            delete_person(pid)
                            if os.path.exists(dest): os.remove(dest)
                            st.error("No face detected in the photo. Please try a different one.")

    with col_prev:
        with st.container(border=True):
            st.subheader("Photo Preview")
            if uploaded:
                st.image(Image.open(uploaded), use_container_width=True)
            else:
                st.info("Upload an image to render preview.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Manage People
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Manage People":
    persons = get_all_persons()
    st.header("Database Management")
    st.caption(f"Currently managing {len(persons)} active biometric profiles.")

    with st.container(border=True):
        if not persons:
            st.info("No personnel registered in the database yet.")
        else:
            for p in persons:
                col1, col2, col3 = st.columns([1, 6, 2])
                with col1:
                    st.write("👤")
                with col2:
                    st.write(f"**{p['name']}**")
                    st.caption(f"Role: {p['role']} | Enrolled: {p['registered_at'][:10]}")
                with col3:
                    if st.button("Revoke Access", key=f"del_{p['id']}", type="secondary", use_container_width=True):
                        delete_person(p["id"])
                        remove_face(p["id"])
                        st.rerun()
                st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Attendance Log
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Attendance Log":
    st.header("Attendance & Reporting")
    st.caption("View and export historical biometric logs.")

    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            date_val = st.date_input("Filter by Specific Date", value=datetime.now())
        with col2:
            st.write("") # Spacing
            st.write("")
            filter_today = st.button("Today Only", use_container_width=True)
        with col3:
            st.write("")
            st.write("")
            show_all = st.button("Show All Records", use_container_width=True)

    active_filter = datetime.now().strftime("%Y-%m-%d") if filter_today else (None if show_all else date_val.strftime("%Y-%m-%d"))
    records = get_attendance_report(active_filter)

    with st.container(border=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Records Found", len(records))
        col_m2.metric("Today's Check-ins", get_today_count())
        col_m3.metric("Total Profiles", len(get_all_persons()))

    st.write("") # Spacing

    with st.container(border=True):
        if not records:
            st.info("No attendance records found for this timeframe.")
        else:
            df = pd.DataFrame(records)[["date","time","person_name","status"]].rename(
                columns={"date":"Date","time":"Time","person_name":"Name","status":"Status"}
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            excel_bytes = _build_excel_bytes(active_filter)
            if excel_bytes:
                st.download_button(
                    label="⬇ Export to Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"attendance_report_{active_filter or 'all'}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )