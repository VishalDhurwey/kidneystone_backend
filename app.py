import psutil
from flask import Flask, request, jsonify
import cv2
import torch
import base64
import os
import gc
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO

# ✅ Select Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on {device}")

# ✅ Force Memory Cleanup


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleaned up")


def log_memory_usage(msg):
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"🔍 {msg} Memory Usage: {memory:.2f} MB")


# ✅ Constants
MODEL_PATH = 'last (1).pt'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
DPI = 300  # Define the DPI of the image for mm conversion

# ✅ Flask Setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app, resources={
     r"/*": {"origins": ["http://localhost:3000", "https://kidneystonedetection.netlify.app"]}})

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Prediction Function (Loads Model Every Time)


def predict_image(image, conf=0.5, dpi=300):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]

        # ✅ Load Model for Each Request
        print("🔄 Reloading YOLO Model...")
        yolo_model = YOLO(MODEL_PATH).to(device)

        with torch.no_grad():
            results = yolo_model(image, conf=conf, max_det=3,
                                 imgsz=416, verbose=False)

        detected_stones = []
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])

                if conf_score > conf:
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width_mm = round((x_max - x_min) / dpi * 25.4, 2)
                    height_mm = round((y_max - y_min) / dpi * 25.4, 2)
                    size_mm = round((width_mm + height_mm) / 2, 2)

                    location = (
                        "Top-left" if center_x < image_width / 2 and center_y < image_height / 2 else
                        "Top-right" if center_x >= image_width / 2 and center_y < image_height / 2 else
                        "Bottom-left" if center_x < image_width / 2 and center_y >= image_height / 2 else
                        "Bottom-right"
                    )

                    detected_stones.append({
                        "confidence": conf_score,
                        "coordinates": [x_min, y_min, x_max, y_max],
                        "location": location,
                        "size_mm": size_mm
                    })

                    cv2.rectangle(image, (x_min, y_min),
                                  (x_max, y_max), (255, 0, 0), 2)
                    label = f"Stone {len(detected_stones)}: {location} ({size_mm}mm)"
                    cv2.putText(image, label, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ✅ Delete Model After Use
        del yolo_model
        free_memory()

        return image, detected_stones

    except Exception as e:
        print(f"❌ Error in predict_image: {str(e)}")
        raise

# ✅ Image Upload & Processing API


@app.route('/upload', methods=['POST'])
def upload_image():
    log_memory_usage("Before Prediction")

    try:
        print("🟢 Received image upload request")
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']
        if image_file.filename == '' or not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # ✅ Run Detection
        pred_img, detections = predict_image(image, dpi=DPI)

        largest_size = max([stone['size_mm']
                           for stone in detections]) if detections else 0

        for stone in detections:
            size = stone['size_mm']
            stone['severity'] = (
                "Mild" if size < 2 else
                "Moderate" if size < 4 else
                "Serious" if size <= 10 else
                "Critical"
            )

        surgery_suggestion = (
            "Home remedies" if largest_size < 2 else
            "Home remedies, medication" if largest_size < 4 else
            "Home remedies, medication, medical procedures (ESWL or Ureteroscopy)" if largest_size <= 10 else
            "Surgery (URSL, RIRS, PCNL)"
        )

        _, buffer = cv2.imencode(
            '.jpg', cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        base64_img = base64.b64encode(buffer).decode('utf-8')

        free_memory()
        log_memory_usage("After Prediction")

        return jsonify({
            'status': 'success',
            'predicted_image': base64_img,
            'detections': detections,
            'surgerySuggestion': surgery_suggestion,
            'message': f'Successfully processed image with {len(detections)} detections'
        })

    except Exception as e:
        print(f"🔴 Error in /upload: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)