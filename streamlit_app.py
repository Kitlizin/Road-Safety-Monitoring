import streamlit as st
import numpy as np
from PIL import Image
import joblib  # or import pickle
import cv2

# Load your model
model = joblib.load('yolov8_reckless2.pt')  # Replace with your model path

st.set_page_config(page_title="Road Safety Monitoring", layout="wide")

# Title section
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>🚦 Reckless Driving Behavior Recognition</h1>
    <h4 style='text-align: center; color: #666;'>For Road Safety Monitoring</h4>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛣️ Navigation")
option = st.sidebar.selectbox("Select Input Type", ["Upload Image", "Upload Video"])

# Class labels
class_labels = ['Pedestrian', 'Vehicle', 'Unsafe', 'Safe']

# Upload & Predict
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = img_array.reshape(1, -1)  # Adjust as per model input

        if st.button("Predict"):
            prediction = model.predict(img_array)
            st.success(f"Prediction: {class_labels[int(prediction[0])]}")
            
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.info("Video processing will be available in the extended version of this app.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made with ❤️ for safer roads — Streamlit Gorgeous like You 😎</p>
""", unsafe_allow_html=True)
