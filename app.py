import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Directories and files
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Streamlit UI
st.title("🎥 Face Recognition Attendance System")

menu = ["🏠 Home", "🧍 Register Face", "📋 View Attendance"]
choice = st.sidebar.selectbox("Select Option", menu)

# ---------------- HOME PAGE ----------------
if choice == "🏠 Home":
    st.write("Welcome, Nandini 💖")
    st.write("Use the sidebar to register faces or check attendance.")

# ---------------- REGISTER FACE ----------------
elif choice == "🧍 Register Face":
    st.subheader("Register New Face")

    name = st.text_input("Enter Your Name:")
    run_camera = st.button("Start Camera")

    if run_camera:
        if not name:
            st.warning("Please enter your name before starting the camera.")
        else:
            st.info("Press 's' to save your face or 'q' to quit camera.")

            cap = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                    cv2.imwrite(face_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    st.success(f"✅ Face registered as {name}")
                    break
                elif key == ord('q'):
                    st.write("Camera closed.")
                    break

            cap.release()
            cv2.destroyAllWindows()

# ---------------- VIEW ATTENDANCE ----------------
elif choice == "📋 View Attendance":
    st.subheader("Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance data found yet.")

