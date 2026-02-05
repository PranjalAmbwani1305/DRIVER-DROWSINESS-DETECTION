# ============================================
# DROWSINESS DETECTION
# STREAMLIT + OPENCV + CNN (NO MEDIAPIPE)
# ============================================

import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import os


# ================= CONFIG ==================

MODEL_PATH = "eye_model.h5"
IMG_SIZE = 64
SLEEP_TIME = 2


# ================= LOAD MODEL ==============

@st.cache_resource
def load_cnn():

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found!")
        st.stop()

    return load_model(MODEL_PATH)

model = load_cnn()


# ================= OPENCV ==================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# ================= UI ======================

st.set_page_config("Drowsiness Detection")

st.title("🚗 Driver Drowsiness Detection")
st.markdown("CNN + OpenCV + Streamlit (No MediaPipe)")


# ================= PREPROCESS ==============

def preprocess(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    return img


# ================= PREDICT =================

def predict_eye(eye):

    eye = preprocess(eye)

    prob = model.predict(eye, verbose=0)[0][0]

    if prob > 0.5:
        return "Closed", prob
    else:
        return "Open", 1 - prob


# ================= MAIN ====================

def run_app():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not accessible")
        return


    frame_box = st.empty()

    start_sleep = None
    drowsy = False


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]
            gray_face = gray[y:y+h, x:x+w]


            eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)


            if len(eyes) == 0:
                continue


            ex, ey, ew, eh = eyes[0]

            eye = face[ey:ey+eh, ex:ex+ew]


            if eye.size == 0:
                continue


            state, conf = predict_eye(eye)


            # Drowsiness logic
            if state == "Closed":

                if start_sleep is None:
                    start_sleep = time.time()

                elif time.time() - start_sleep > SLEEP_TIME:
                    drowsy = True

            else:
                start_sleep = None
                drowsy = False


            # Display
            color = (0,255,0) if state=="Open" else (0,0,255)

            cv2.rectangle(
                frame,
                (x+ex, y+ey),
                (x+ex+ew, y+ey+eh),
                color, 2
            )

            cv2.putText(
                frame,
                f"{state} {round(conf*100,1)}%",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )


            if drowsy:

                cv2.putText(
                    frame,
                    "DROWSY ALERT!",
                    (40,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,(0,0,255),3
                )

                st.warning("⚠️ Driver is Drowsy!")


        frame_box.image(frame, channels="BGR")


    cap.release()
    cv2.destroyAllWindows()


# ================= BUTTON ==================

if st.button("▶ Start Detection"):
    run_app()
