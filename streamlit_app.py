import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import random

# Load your custom YOLOv8 model
model = YOLO("yolov8_reckless_best.pt")

# Set Streamlit page configuration
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

# Assign random colors to class names for consistency
def get_class_colors(class_names):
    random.seed(42)
    return {name: [random.randint(0, 255) for _ in range(3)] for name in class_names}

class_colors = get_class_colors(model.names.values())

# Bounding box drawing with background label and per-class color
def draw_boxes(image_np, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = results.names[class_id]
        conf = float(box.conf[0])
        label_text = f"{label} {conf:.2f}"
        color = class_colors.get(label, (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        # Get text size
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw filled background
        cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

        # Put text over background
        cv2.putText(image_np, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
