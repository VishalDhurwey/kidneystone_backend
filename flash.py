from flask import Flask, request, jsonify, send_file
import cv2
import torch
import os
import uuid
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import gc

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Load YOLO Model
MODEL_PATH = 'last.pt'
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Flask Setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
LABELS_FOLDER = 'labels'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory, max_files=100):
    """Remove old files if directory has more than max_files"""
    files = os.listdir(directory)
    if len(files) > max_files:
        for file in sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))[:-max_files]:
            try:
                os.remove(os.path.join(directory, file))
            except Exception as e:
                print(f"Error cleaning up file {file}: {e}")

def stone_detection(img_path, model, labels_dir):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        detections = results[0].boxes.data

        print(f"\n🔍 Processing image: {os.path.basename(img_path)}")
        print(f"📊 Found {len(detections)} detections")

        img_with_pred_boxes = img_rgb.copy()
        detected_stones = []

        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, class_id = map(float, det.tolist())
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                class_name = model.names[int(class_id)].lower()
                print(f"📌 Detected {class_name} with confidence {conf:.2f}")

                # Draw bounding box
                color = (255, 0, 0)  # Red for predicted stones
                cv2.rectangle(img_with_pred_boxes, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img_with_pred_boxes, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detected_stones.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': round(conf, 2),
                    'class': class_name
                })

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return img_with_pred_boxes, detected_stones

    except Exception as e:
        print(f"❌ Error in stone detection: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(image_file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Secure the filename and save
        image_filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image_file.save(image_path)

        # Run detection
        pred_img, detections = stone_detection(image_path, model, LABELS_FOLDER)

        # Save processed image
        pred_img_path = os.path.join(RESULT_FOLDER, f"pred_{image_filename}")
        cv2.imwrite(pred_img_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

        # Cleanup old files
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(RESULT_FOLDER)

        return jsonify({
            'status': 'success',
            'predicted_image_url': f'http://127.0.0.1:5000/download/{os.path.basename(pred_img_path)}',
            'detections': detections,
            'message': f'Successfully processed image with {len(detections)} detections'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_file(
            os.path.join(RESULT_FOLDER, secure_filename(filename)), 
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

if __name__ == '__main__':
    app.run(debug=True)
