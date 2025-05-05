import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pinecone
import threading
import pygame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

# Load alarm sounds
WELCOME_SOUND = "alarm2.mp3"  # Welcome sound at system startup
ALERT_SOUND = "alarm1.wav"    # Drowsiness alert sound

# Set Page Configuration
st.set_page_config(page_title="🚗 Driver Drowsiness Detection", layout="wide")

# Initialize Pinecone for data storage
pc = pinecone.Pinecone(api_key="pcsk_2f5C9U_6k7rXahFdZDUB3wN4kqRKLiwBeUbWTthR2mWvHS5hKYJGLPdNBVANYKTFtDwxBt")
index = pc.Index("drowsiness-detection")  # Ensure this index has 384 dimensions

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define Eye Landmark Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Play welcome sound once at startup
def play_welcome_sound():
    pygame.mixer.music.load(WELCOME_SOUND)
    pygame.mixer.music.play()

# Function to play alert sound
def play_alert():
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play()

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_points):
    points = np.array([landmarks[p] for p in eye_points])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

# Store Drowsiness Data in Pinecone
def store_data(user_id, drowsiness_score, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vector = np.random.rand(384).tolist()  # Generate a random vector for indexing

    try:
        index.upsert([
            {
                "id": f"{user_id}_{timestamp}",
                "values": vector,
                "metadata": {
                    "drowsiness_score": float(drowsiness_score),
                    "confidence": float(confidence),
                    "timestamp": timestamp
                }
            }
        ])
    except Exception as e:
        st.error(f"❌ Error storing data in Pinecone: {e}")

# Function to detect drowsiness
def detect_drowsiness():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam access failed! Check permissions.")
        return

    stop_button = st.button("❌ Stop Detection")
    last_alert_time = 0  # Track last alert time
    welcome_played = False  # Track if welcome sound has played

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if not welcome_played:
            threading.Thread(target=play_welcome_sound).start()
            welcome_played = True

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = {i: (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                         for i, lm in enumerate(face_landmarks.landmark)}
            
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2
            confidence = round((1 - avg_ear) * 100, 2)

            current_time = time.time()
            if avg_ear < 0.22 and (current_time - last_alert_time > 5):  # Alert every 20 sec
                st.warning(f"⚠️ Drowsiness Detected! Confidence: {confidence}%")
                store_data("user_123", avg_ear, confidence)
                threading.Thread(target=play_alert).start()
                last_alert_time = current_time

        stframe.image(frame, channels="RGB")

    cap.release()

# Admin Dashboard for Data Analytics
def admin_dashboard():
    st.subheader("📊 Admin Dashboard - Drowsiness Analysis")
    try:
        stored_data = index.query(vector=np.random.rand(384).tolist(), top_k=100, include_metadata=True)
        
        if 'matches' not in stored_data or not stored_data['matches']:
            st.warning("No data available!")
            return

        df = pd.DataFrame([x['metadata'] for x in stored_data['matches']])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(df['drowsiness_score'], bins=10, kde=True, ax=ax)
            ax.set_title("Drowsiness Score Distribution")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error retrieving data: {e}")

# Sidebar Navigation for Authentication
st.sidebar.title("🔑 Authentication")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if login_button:
    if username == "admin" and password == "admin123":
        st.session_state['authenticated'] = True
        st.sidebar.success("Logged in as Admin")
    elif username == "user" and password == "user123":
        st.session_state['authenticated'] = True
        st.sidebar.success("Logged in as User")
    else:
        st.sidebar.error("Invalid credentials")

# Main Page Navigation
if 'authenticated' in st.session_state and st.session_state['authenticated']:
    page = st.sidebar.radio("Select Page:", ["User", "Admin"])
    
    if page == "User":
        st.title("🚗 Driver Drowsiness Detection")
        st.write("Detects driver drowsiness using real-time video and alerts when drowsiness is detected.")
        if st.button("▶️ Start Detection"):
            detect_drowsiness()
    
    elif page == "Admin":
        st.title("📊 Admin Dashboard")
        admin_dashboard()
else:
    st.warning("Please login to continue.")
