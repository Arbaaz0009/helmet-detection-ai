"""
Safety Helmet Detection Web Application
======================================

A Streamlit web application for detecting safety helmets in images using YOLOv8.
This app allows users to upload images and get real-time helmet detection results.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

# Try to import cv2 with better error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing OpenCV (cv2): {str(e)}")
    st.error("Please ensure opencv-python-headless is properly installed.")
    CV2_AVAILABLE = False

# Try to import ultralytics with better error handling
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing ultralytics: {str(e)}")
    st.error("Please ensure ultralytics is properly installed. Check requirements.txt")
    ULTRALYTICS_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load your YOLOv8 trained model with error handling
@st.cache_resource
def load_model():
    if not ULTRALYTICS_AVAILABLE:
        return None
    
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV is required for image processing but not available.")
        return None
        
    try:
        # Try different possible model paths
        model_paths = [
            "runs/detect/train4/weights/best.pt",  # Your trained model
            "runs/detect/train3/weights/best.pt",
            "runs/detect/train2/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "yolov8n.pt"  # Fallback to default model
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                st.success(f"✅ Model loaded from: {path}")
                return YOLO(path)
        
        # If no local model found, try to download a pre-trained model
        st.warning("⚠️ No local model found. Downloading pre-trained YOLOv8n model...")
        try:
            model = YOLO("yolov8n.pt")
            st.success("✅ Pre-trained YOLOv8n model loaded successfully!")
            st.info("ℹ️ Note: This is a general-purpose model, not specifically trained for helmets.")
            return model
        except Exception as download_error:
            st.error(f"❌ Failed to download model: {str(download_error)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

st.title("🪖 Helmet Compliance Detection App")
st.write("Upload an image below to detect helmets.")

# Upload file widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Convert PIL image to numpy array for processing
        img_array = np.array(image)
        
        # Run detection without saving to file
        with st.spinner("🔍 Running detection..."):
            results = model.predict(source=img_array, conf=0.5, save=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Get the annotated image
            annotated_img = result.plot()
            
            # Convert BGR to RGB for display
            if CV2_AVAILABLE:
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            else:
                # Fallback: assume image is already in RGB format
                annotated_img_rgb = annotated_img
            
            # Display the result image
            st.image(annotated_img_rgb, caption='Detection Result', use_container_width=True)
            
            # Display detection information
            if result.boxes is not None:
                num_detections = len(result.boxes)
                st.success(f"✅ Detection complete! Found {num_detections} object(s)")
                
                # Show detection details
                if num_detections > 0:
                    st.subheader("Detection Details:")
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        st.write(f"Object {i+1}: {class_name} (Confidence: {conf:.2f})")
            else:
                st.warning("⚠️ No objects detected in the image.")
                
        else:
            st.error("❌ Detection failed. Please try again.")
            
    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")
        st.error("Please check if the image is valid and try again.")

elif uploaded_file is not None and model is None:
    st.error("❌ Model not loaded. Please check the model file and restart the app.")

# Add some helpful information
with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses a YOLOv8 model trained to detect helmets in images. 
    
    **How to use:**
    1. Upload an image (JPG, JPEG, or PNG)
    2. The app will automatically detect helmets in the image
    3. Results will be displayed with bounding boxes around detected objects
    
    **Model Information:**
    - The model is trained to detect helmet compliance
    - Confidence threshold is set to 0.5
    - Results show bounding boxes and confidence scores
    """)

