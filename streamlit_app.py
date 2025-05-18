import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile

# Load your custom YOLOv8 model
model = YOLO("yolov8_reckless_best.pt")

# Streamlit page configuration
st.set_page_config(page_title="🚗 Reckless Driving Detector", layout="centered")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #d62728;'>🚦 Reckless Driving Behavior Recognition</h1>
    <h4 style='text-align: center;'>Road Safety Monitoring System</h4><hr>
    """,
    unsafe_allow_html=True
)

# Sidebar for media upload
st.sidebar.title("📂 Upload Media")
media_type = st.sidebar.radio("Select Input Type:", ("Image", "Video"))

# Bounding box drawer with default style
def draw_boxes(image_np, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        label_text = f"{label} {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Default thickness
        cv2.putText(image_np, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Default font size and thickness
    return image_np

# Image handler
if media_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        results = model(image_np)[0]
        processed = draw_boxes(image_np.copy(), results)
        st.image(processed, caption="🖼️ Detection Result", use_container_width=True)

# Video handler
elif media_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = draw_boxes(frame.copy(), results)
            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
