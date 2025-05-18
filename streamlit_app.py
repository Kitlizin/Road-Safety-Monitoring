import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8_reckless_best.pt")

# Streamlit page setup
st.set_page_config(page_title="🚦 Reckless Driving Detector", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #ff4b4b;'>Reckless Driving Behavior Recognition</h1>
    <h4 style='text-align: center;'>Road Safety Monitoring System</h4><hr>
    """,
    unsafe_allow_html=True
)

# Sidebar for input selection
st.sidebar.title("📂 Upload Media")
media_type = st.sidebar.radio("Choose input type:", ("Image", "Video"))

def draw_boxes(image_np, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        label_text = f"{label} {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return image_np

if media_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        results = model(image_np)[0]
        processed_image = draw_boxes(image_np, results)
        st.image(processed_image, caption="🖼️ Detection Result", use_container_width=True)

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
            results = model(frame)[0]
            annotated_frame = draw_boxes(frame, results)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        cap.release()
