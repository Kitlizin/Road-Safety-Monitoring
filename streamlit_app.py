import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import os

# Load YOLOv8 model
model = YOLO("yolov8_reckless2.pt")

# Custom styling
st.set_page_config(page_title="🚦 Reckless Driving Detector", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>Reckless Driving Behavior Recognition</h1>"
    "<h4 style='text-align: center;'>Road Safety Monitoring System</h4><hr>",
    unsafe_allow_html=True
)

st.sidebar.title("Upload Media")
media_type = st.sidebar.radio("Choose input type:", ("Image", "Video"))

if media_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_file).convert("RGB")
        results = model.predict(image)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Prediction", use_column_width=True)

elif media_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            res_plotted = results[0].plot()
            stframe.image(res_plotted, channels="BGR", use_column_width=True)
        cap.release()
