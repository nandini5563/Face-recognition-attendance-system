import streamlit as st
import pandas as pd
import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime

st.set_page_config(page_title="Face Recognition Attendance", page_icon="📸", layout="wide")

st.title("📸 Face Recognition Attendance System")

# === Utility Functions ===

def register_face(name):
    st.info("Camera will open. Press 'S' to capture, 'Q' to quit.")
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    os.makedirs("known_faces", exist_ok=True)
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite(f"known_faces/{name}.jpg", frame[y:y+h, x:x+w])
                st.success(f"✅ Face registered for {name}")
                cam.release()
                cv2.destroyAllWindows()
                return
        cv2.imshow("Register Face - Press S to Save", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def mark_attendance():
    known_faces = []
    known_names = []

    for file in os.listdir("known_faces"):
        img = face_recognition.load_image_file(f"known_faces/{file}")
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(file)[0])

    cap = cv2.VideoCapture(0)
    attendance = pd.DataFrame(columns=["Name", "Date", "Time"])
    st.info("Camera active! Press 'Q' to quit recognition.")

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for encodeFace, faceLoc in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_faces, encodeFace)
            faceDis = face_recognition.face_distance(known_faces, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = known_names[matchIndex].upper()
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                if not os.path.exists("attendance.csv"):
                    attendance.to_csv("attendance.csv", index=False)

                df = pd.read_csv("attendance.csv") if os.path.exists("attendance.csv") else pd.DataFrame(columns=["Name", "Date", "Time"])
                if not ((df["Name"] == name) & (df["Date"] == date)).any():
                    new_row = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv("attendance.csv", index=False)
                    print(f"{name} marked present at {time}")

                (top, right, bottom, left) = faceLoc
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Mark Attendance - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Attendance Marked Successfully!")


def view_attendance():
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("No attendance data found!")


# === Streamlit Sidebar ===
st.sidebar.title("Options")
option = st.sidebar.radio("Select Task:", ["Register Face", "Mark Attendance", "View Attendance"])

if option == "Register Face":
    name = st.text_input("Enter name:")
    if st.button("Register"):
        if name.strip() != "":
            register_face(name.strip())
        else:
            st.warning("Please enter a valid name!")

elif option == "Mark Attendance":
    if st.button("Start Camera"):
        mark_attendance()

elif option == "View Attendance":
    view_attendance()

st.markdown("---")
st.caption("Made with ❤️ by Nandini")


