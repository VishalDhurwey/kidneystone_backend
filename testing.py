import requests
import os
import torch
from PIL import Image
from ultralytics import YOLO

# Flask server URL
UPLOAD_URL = "http://127.0.0.1:5000/upload"
OUTPUT_DIR = "test_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to test image
TEST_IMAGE_PATH = "testing_image.jpg"  # Change this to an actual image path

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
    exit()

# Load the model
model_path = "yolo11s.pt"
try:
    model = YOLO(model_path)  # Load YOLO model
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load an image
try:
    image = Image.open(TEST_IMAGE_PATH)
    print(f"✅ Image '{TEST_IMAGE_PATH}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit()

# Perform inference
try:
    results = model(image)
    output_path = os.path.join(OUTPUT_DIR, "output.jpg")
    
    # Save the output image with detections
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        im.save(output_path)  # save image
    
    print(f"✅ Inference completed successfully. Output saved as '{output_path}'.")
except Exception as e:
    print(f"❌ Error during inference: {e}")
