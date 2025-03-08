from flask import Flask, request, jsonify, send_file
import cv2
import torch
import base64
import os
import uuid
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import gc
import google.generativeai as genai

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
DPI = 300  # Define the DPI of the image for mm conversion

# Load YOLO Model
MODEL_PATH = 'last (1).pt'
device = torch.device("cpu")  # Force CPU

try:
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.to(device)  # Move model to CPU
    print(f"✅ Model loaded successfully on {device}")
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

def pixels_to_mm(pixels, dpi):
    return round((pixels / dpi) * 25.4, 2)  # Convert pixels to mm

def predict_image(image_path=None, image=None, conf=0.3, dpi=300):
    try:
        if image is None and image_path is not None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"❌ Image not found at {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("❌ Failed to load the image. Check the file format and path.")
        elif image is None:
            raise ValueError("❌ Either image_path or image must be provided")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        # Run inference
        results = yolo_model(image)
        
        detected_stones = []
        
        # Process detections
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
                        "confidence": float(conf_score),
                        "coordinates": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "center": [float(center_x), float(center_y)],
                        "location": location,
                        "size_mm": size_mm,
                        "dimensions_mm": {
                            "width": round(width_mm, 2),
                            "height": round(height_mm, 2)
                        }
                    })
                    
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    stone_number = len(detected_stones)
                    label = f"Stone {stone_number}: {location} ({size_mm}mm)"
                    cv2.putText(image, label, (x_min, y_min - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if detected_stones:
            print(f"✅ Number of Kidney Stones Detected: {len(detected_stones)}")
            for i, stone in enumerate(detected_stones, 1):
                print(f"  {i}. Location: {stone['location']}, Size: {stone['size_mm']}mm")
        else:
            print("❌ No kidney stones detected above the confidence threshold.")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return image, detected_stones

    except Exception as e:
        print(f"❌ Error in predict_image: {str(e)}")
        raise

# Configure Gemini
genai.configure(api_key="AIzaSyAx9pUTUhIRVWQjIMOsR6_oxl8vBkXLXOg")
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        prompt = f"""
You are a polite kidney health assistant.
Answer the user's question briefly in 1 short and clear sentence.
Keep the tone friendly and supportive.

User: {user_message}
Assistant:
"""
        response = gemini_model.generate_content(prompt)
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        print(f"Chat API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        print("🟢 Received image upload request")
        print(f"Request Files: {request.files}")
        print(f"Request Headers: {request.headers}")

        if 'image' not in request.files:
            print("❌ No image file in request")
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']
        print(f"📁 Received file: {image_file.filename}")
        
        if image_file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(image_file.filename):
            print(f"❌ Invalid file type: {image_file.filename}")
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        try:
            image_np = np.frombuffer(image_file.read(), np.uint8)
            print(f"📊 Image buffer size: {len(image_np)}")
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            print(f"🖼️ Decoded image shape: {image.shape if image is not None else 'None'}")
        except Exception as e:
            print(f"❌ Image decode error: {str(e)}")
            return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400

        if image is None:
            print("❌ Image decoding failed")
            return jsonify({'error': 'Failed to decode image'}), 400

        print("🟢 Running detection on the image")
        try:
            pred_img, detections = predict_image(image=image, dpi=DPI)
            
            largest_size = max([stone['size_mm'] for stone in detections]) if detections else 0
            
            for stone in detections:
                size = stone['size_mm']
                if size < 2:
                    stone['severity'] = "Mild"
                elif size < 4:
                    stone['severity'] = "Moderate"
                elif size <= 10:
                    stone['severity'] = "Serious"
                else:
                    stone['severity'] = "Critical"

            if largest_size < 2:
                surgery_suggestion = "Home remedies"
            elif largest_size < 4:
                surgery_suggestion = "Home remedies, medication"
            elif largest_size <= 10:
                surgery_suggestion = "Home remedies, medication, medical procedures (ESWL or Ureteroscopy)"
            else:
                surgery_suggestion = "Surgery (URSL, RIRS, PCNL)"

        except Exception as e:
            print(f"🔴 Detection error: {str(e)}")
            return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'predicted_image': base64_img,
            'detections': detections,
            'surgerySuggestion': surgery_suggestion,
            'message': f'Successfully processed image with {len(detections)} detections'
        })

    except Exception as e:
        print(f"🔴 Error in /upload endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    port = int(os.getenv("PORT", 5000))  # Use PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)