import streamlit as st
import pandas as pd
import joblib
import os
import cv2
import time
from PIL import Image
import pytesseract
import sqlite3

# ----------------------------
# Database setup
# ----------------------------
conn = sqlite3.connect("student_hub.db", check_same_thread=False)
cursor = conn.cursor()

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")



# Create notes table
cursor.execute("""
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    data BLOB,
    uploaded_by TEXT,
    timestamp TEXT
)
""")

# Create chat table
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    message TEXT,
    timestamp TEXT
)
""")

conn.commit()

# ----------------------------
# Session state for login
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ----------------------------
# Configure Tesseract path
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------
# Set page configuration
# ----------------------------
st.set_page_config(page_title="Student Focus Hub 🎓", layout="wide")

# ----------------------------
# ----------------------------
# Sidebar navigation and login
# ----------------------------
if not st.session_state.logged_in:
    st.title("🔑 Sign In / Sign Up")

    tab = st.radio("Choose Action:", ["Sign In", "Sign Up"])

    if tab == "Sign In":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign In"):
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            user = cursor.fetchone()
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid credentials!")

    elif tab == "Sign Up":
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, new_pass))
                conn.commit()
                st.success("Account created! Please sign in.")
            except:
                st.error("Username already exists.")

else:
    # User is logged in → show the main app pages
    st.sidebar.title("📚 Student Focus & Collaboration Hub")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home Dashboard", "📂 Share Notes", "💬 Community Chat", "🎥 Focus Tracker", "✏️ Assignment OCR"]
    )

    # --------- Your existing page code goes here ---------


# ----------------------------
# Load ML model
# ----------------------------
    model_path = "student_model.pkl"
    model = joblib.load(model_path)

    # ========== PAGE 1: HOME DASHBOARD ==========
    if page == "🏠 Home Dashboard":
        st.title("📈 Student Focus Prediction Dashboard")
        st.write("Predict your expected final score based on your daily performance.")

        hours = st.slider("Hours Studied per Day", 1.0, 10.0, 3.5)
        attendance = st.slider("Attendance (%)", 50, 100, 85)
        previous = st.slider("Previous Exam Score", 30, 100, 75)
        assignments = st.slider("Assignments Submitted", 0, 10, 8)
        attention = st.slider("Average Attention (%)", 30, 100, 80)

        if st.button("🎯 Predict Final Score"):
            features = pd.DataFrame([{
                "Hours_Studied_per_day": hours,
                "Attendance_pct": attendance,
                "Previous_Score": previous,
                "Assignments_Submitted": assignments,
                "Avg_Attention_pct": attention
            }])
            predicted = model.predict(features)[0]
            st.success(f"✅ Predicted Final Score: **{predicted:.2f}** / 100")
            if attention < 60:
                st.warning("⚠️ Your attention is quite low. Try to stay focused!")

    # ========== PAGE 2: SHARE NOTES ==========
    # --------- PAGE 2: SHARE NOTES (Database version) ---------
    elif page == "📂 Share Notes":
        st.title("📚 Share Notes with Classmates")
        
        # Upload notes
        uploaded_file = st.file_uploader(
            "Upload your notes (PDF, DOCX, ZIP)",
            type=["pdf", "docx", "zip"]
        )

        if uploaded_file:
            file_data = uploaded_file.read()
            cursor.execute("""
                INSERT INTO notes (filename, data, uploaded_by, timestamp)
                VALUES (?, ?, ?, ?)
            """, (uploaded_file.name, file_data, st.session_state.username, time.strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")

        # Display available notes with download buttons
        st.subheader("📁 Available Notes:")
        cursor.execute("SELECT id, filename, uploaded_by, timestamp FROM notes ORDER BY id DESC")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                note_id, filename, uploaded_by, ts = row
                st.write(f"📄 {filename} (by {uploaded_by} at {ts})")
                cursor.execute("SELECT data FROM notes WHERE id=?", (note_id,))
                file_data = cursor.fetchone()[0]
                st.download_button(
                    label=f"Download {filename}",
                    data=file_data,
                    file_name=filename,
                    mime="application/octet-stream"
                )
        else:
            st.info("No shared notes yet.")


    # ========== PAGE 3: COMMUNITY CHAT ==========
    # --------- PAGE 3: COMMUNITY CHAT (Database version) ---------
    elif page == "💬 Community Chat":
        st.title("💬 Student Community Chat")
        
        user = st.session_state.username
        message = st.text_area("💭 Your Message:")
        if st.button("📤 Post"):
            if user and message:
                cursor.execute("""
                    INSERT INTO chat (username, message, timestamp)
                    VALUES (?, ?, ?)
                """, (user, message, time.strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                st.success("✅ Message posted!")
            else:
                st.warning("Please enter your message.")

        # Display recent messages
        st.subheader("🗨️ Recent Messages:")
        cursor.execute("SELECT username, message, timestamp FROM chat ORDER BY id DESC LIMIT 10")
        messages = cursor.fetchall()
        if messages:
            for username, message, ts in reversed(messages):  # show oldest first
                st.write(f"**{username}** [{ts}]: {message}")
        else:
            st.info("No messages yet. Be the first to post!")


    # ========== PAGE 4: FOCUS TRACKER ==========
    elif page == "🎥 Focus Tracker":
        st.title("🎥 Real-time Attention Tracker")
        st.write("This feature uses your webcam to track attention level.")
        st.info("⚠️ Click below to start camera in a new window.")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        if st.button("▶️ Start Tracking"):
            cap = cv2.VideoCapture(0)
            attention_score = 100
            last_check_time = time.time()
            frame_placeholder = st.empty()
            log = []
            stop = False

            st.info("Press **Stop Tracking** below to end session.")
            stop_button = st.button("🛑 Stop Tracking")

            while not stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Cannot access camera.")
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        if len(eyes) >= 1:
                            attention_score = min(attention_score + 1, 100)
                        else:
                            attention_score = max(attention_score - 2, 0)
                else:
                    attention_score = max(attention_score - 2, 0)

                cv2.putText(frame, f"Attention: {attention_score:.0f}%", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                if time.time() - last_check_time > 5:
                    last_check_time = time.time()
                    features = pd.DataFrame([{
                        "Hours_Studied_per_day": 3.5,
                        "Attendance_pct": 85,
                        "Previous_Score": 75,
                        "Assignments_Submitted": 8,
                        "Avg_Attention_pct": attention_score
                    }])
                    predicted_score = model.predict(features)[0]
                    log.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "attention": attention_score,
                        "predicted_score": predicted_score
                    })
                    st.write(f"📊 Predicted Final Score: **{predicted_score:.2f}** | Attention: **{attention_score:.0f}%**")
                    if attention_score < 60:
                        st.warning("⚠️ Low attention detected! Please focus!")

                if stop_button:
                    stop = True
                    break

            cap.release()
            st.success("✅ Tracking Stopped.")
            st.write("### Session Log")
            st.dataframe(pd.DataFrame(log))

    # ========== PAGE 5: ASSIGNMENT OCR ==========
    elif page == "✏️ Assignment OCR":
        st.title("✏️ Assignment OCR Checker")
        st.write("Upload a handwritten or printed assignment image to extract text.")

        uploaded_file = st.file_uploader(
            "Upload your assignment image (JPG, PNG, JPEG)",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.subheader("📄 Uploaded Assignment")
            st.image(img, use_container_width=True)

            # Minimal OCR (best for clean images)
            extracted_text = pytesseract.image_to_string(img)

            st.subheader("📝 Extracted Text")
            st.text_area("Text from Assignment", extracted_text, height=300)

            # Optional auto-grading
            st.subheader("✅ Auto-Grading Example")
            solution_key = "This is the correct answer for comparison."  # Replace with real solution
            score = 0
            if solution_key.lower() in extracted_text.lower():
                score = 100
            st.write(f"Score: {score} / 100")
