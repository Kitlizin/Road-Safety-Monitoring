import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import random
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

model = YOLO("FinalModel_yolov8.pt")

st.set_page_config(page_title="🚗 Reckless Driving Detector", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #d62728;'>🚦Reckless Driving Behavior Recognition For Road Safety Monitoring🚦</h1>
    <h4 style='text-align: center;'>⚠️ Road Safety Monitoring System ⚠️</h4><hr> 
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📂 Choose Input Type")
media_type = st.sidebar.radio("Select input 🎯:", ("🖼️ Image", "🎥 Video"))

def get_class_colors(class_names):
    random.seed(42)
    return {name: [random.randint(0, 255) for _ in range(3)] for name in class_names}

class_colors = get_class_colors(model.names.values())

custom_colors = {
    "Vehicle": (137, 207, 240),     # Light Blue
    "Pedestrian": (255, 179, 71),   # Light Orange
}

def draw_boxes(image_np, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[class_id] if hasattr(results, 'names') else model.names[class_id]
        label_text = f"{label} {conf:.2f}"
        color = custom_colors.get(label, (255, 255, 255))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image_np, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image_np

if media_type == "🖼️ Image":
    uploaded_image = st.sidebar.file_uploader("Upload your image", type=["jpg", "jpeg", "png"]) 
    if uploaded_image:
        from PIL import Image
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        results = model(image_np)[0]
        processed = draw_boxes(image_np.copy(), results)
        st.image(processed, caption="🖼️ Detection Result — Stay safe out there!", use_container_width=True)

elif media_type == "🎥 Video":
    import tempfile
    uploaded_video = st.sidebar.file_uploader("Upload your video", type=["mp4", "avi", "mov"]) 
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = draw_boxes(frame.copy(), results)
            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processing complete! Drive safe! 🚗💨")
