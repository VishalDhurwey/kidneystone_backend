from flask import Flask, request, jsonify, send_file, url_for
import cv2
import torch
import os
import uuid
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import gc
import google.generativeai as genai

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
DPI = 300

MODEL_PATH = 'last (1).pt'

try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory, max_files=100):
    files = os.listdir(directory)
    if len(files) > max_files:
        for file in sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))[:-max_files]:
            try:
                os.remove(os.path.join(directory, file))
            except Exception as e:
                print(f"Error cleaning up file {file}: {e}")

def pixels_to_mm(pixels, dpi):
    return round((pixels / dpi) * 25.4, 2)

def predict_image(image_path, dpi=300, conf=0.3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    results = model(image)
    detected_stones = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            if conf_score > conf:
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width_mm = pixels_to_mm(x_max - x_min, dpi)
                height_mm = pixels_to_mm(y_max - y_min, dpi)
                size_mm = round((width_mm + height_mm) / 2, 2)

                if center_x < image_width / 2 and center_y < image_height / 2:
                    location = "Top-left"
                elif center_x >= image_width / 2 and center_y < image_height / 2:
                    location = "Top-right"
                elif center_x < image_width / 2 and center_y >= image_height / 2:
                    location = "Bottom-left"
                else:
                    location = "Bottom-right"

                detected_stones.append({
                    "confidence": conf_score,
                    "coordinates": [x_min, y_min, x_max, y_max],
                    "center": [center_x, center_y],
                    "location": location,
                    "size_mm": size_mm,
                    "dimensions_mm": {"width": width_mm, "height": height_mm}
                })

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                label = f"Stone {len(detected_stones)}: {location} ({size_mm}mm)"
                cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return image, detected_stones

genai.configure(api_key="YOUR_GEMINI_API_KEY")

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    prompt = f"User: {user_message}\nAssistant:" 

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    answer = response.text.strip()

    return jsonify({"response": answer})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    image_filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image_file.save(image_path)

    pred_img, detections = predict_image(image_path, dpi=DPI)
    pred_img_path = os.path.join(RESULT_FOLDER, f"pred_{image_filename}")
    cv2.imwrite(pred_img_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

    cleanup_old_files(UPLOAD_FOLDER)
    cleanup_old_files(RESULT_FOLDER)

    download_url = url_for('download_file', filename=f"pred_{image_filename}", _external=True)

    return jsonify({
        'status': 'success',
        'predicted_image_url': download_url,
        'detections': detections
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
