import numpy as np
import math
import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define labels corresponding to your model's output classes
labels = ["A", "B", "C"]

# Load the hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Initialize Streamlit
st.title("ASL Sign Language Translator")

# Initialize session state
if 'video_running' not in st.session_state:
    st.session_state.video_running = False

# Function to toggle video state
def toggle_video():
    st.session_state.video_running = not st.session_state.video_running

# Button to start/stop video
st.button("Start/Stop Video", on_click=toggle_video)

# Placeholders for video feed and prediction
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Initialize label variable
label = "Waiting for prediction..."

if st.session_state.video_running:
    # Video processing loop
    while st.session_state.video_running:
        success, img = cap.read()
        if not success:
            st.write("Error: Unable to access webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, w, y, h = hand['bbox']

            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]
            imgCropShape = imgCrop.shape

            aspectratio = h / w
            if aspectratio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Ensure index is within bounds
            if 0 <= index < len(labels):
                label = labels[index]
            else:
                label = "Unknown"

            cv2.rectangle(imgOutput, (x - 20, y - 20 - 50), (x - 20 + 150, y - 20 + 50), (255, 255, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 255), 4)

        # Display the video feed and prediction
        frame_placeholder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), channels="RGB")
        prediction_placeholder.write(f"Prediction: {label}")

        # Break the loop if stop button is pressed
        if not st.session_state.video_running:
            break

    cap.release()
    cv2.destroyAllWindows()

# Message to guide users
if not st.session_state.video_running:
    st.write("Press 'Start/Stop Video' to start or stop the video feed.")