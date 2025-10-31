import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Paths
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize attendance file
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Streamlit UI
st.title("🎥 Face Recognition Attendance System")
st.write("Developed by **Nandini 💙**")

menu = ["🏠 Home", "🧍 Register Face", "📋 View Attendance"]
choice = st.sidebar.selectbox("Select Option", menu)

# -------------- HOME ----------------
if choice == "🏠 Home":
    st.subheader("Welcome 👋")
    st.write("Use the sidebar to register new faces or view attendance records.")
    st.image("https://i.imgur.com/3E5z4FB.png", caption="AI-based Attendance System", use_container_width=True)

# -------------- REGISTER FACE ----------------
elif choice == "🧍 Register Face":
    st.subheader("📸 Register Your Face")

    name = st.text_input("Enter your name:")
    uploaded_image = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])

    if uploaded_image and name:
        # Save image to known_faces folder
        face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        with open(face_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.success(f"✅ {name}'s face registered successfully!")

        # Mark attendance instantly
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        df = pd.read_csv(ATTENDANCE_FILE)
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.info("Attendance recorded successfully! 🕒")

    elif name and not uploaded_image:
        st.warning("Please upload an image file of your face.")

# -------------- VIEW ATTENDANCE ----------------
elif choice == "📋 View Attendance":
    st.subheader("📅 Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Attendance CSV", csv, "attendance.csv", "text/csv")
    else:
        st.info("No attendance records found yet.")
