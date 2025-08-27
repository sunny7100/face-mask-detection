from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class FaceMaskDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return faces
    
    def predict_mask(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Focus on mouth/nose area (lower part of face)
        mouth_region = gray[int(h*0.55):int(h*0.85), int(w*0.2):int(w*0.8)]
        
        if mouth_region.size == 0:
            return "No Mask", 0.50
        
        # Calculate multiple features
        mean_val = np.mean(mouth_region)
        std_val = np.std(mouth_region)
        
        # Edge detection in mouth area
        edges = cv2.Canny(mouth_region, 30, 100)
        edge_ratio = np.sum(edges > 0) / mouth_region.size
        
        # Texture analysis using Laplacian variance
        laplacian_var = cv2.Laplacian(mouth_region, cv2.CV_64F).var()
        
        # Decision logic (inverted from previous)
        # No mask: More edges, higher texture variance, typical skin tone
        # With mask: Fewer edges, lower texture variance, uniform appearance
        
        no_mask_score = 0
        
        # High edge density suggests visible mouth/nose features
        if edge_ratio > 0.05:
            no_mask_score += 2
        elif edge_ratio > 0.02:
            no_mask_score += 1
            
        # High texture variance suggests natural skin/lip texture
        if laplacian_var > 100:
            no_mask_score += 2
        elif laplacian_var > 50:
            no_mask_score += 1
            
        # Typical skin tone range for mouth area
        if 60 < mean_val < 200 and std_val > 15:
            no_mask_score += 1
        
        # Decision: Higher score = No Mask
        if no_mask_score >= 3:
            confidence = min(0.95, 0.75 + (no_mask_score * 0.05))
            return "No Mask", confidence
        else:
            confidence = min(0.92, 0.70 + ((5-no_mask_score) * 0.04))
            return "With Mask", confidence
    
    def process_image(self, image):
        faces = self.detect_faces(image)
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y+h, x:x+w]
            label, confidence = self.predict_mask(face_roi)
            
            # Green for mask, Red for no mask
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            cv2.putText(image, f"{label}: {confidence:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            results.append({
                'face_id': i+1,
                'label': label,
                'confidence': confidence,
                'bbox': [x, y, w, h]
            })
        
        return image, results

detector = FaceMaskDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        result_image, detections = detector.process_image(image.copy())
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        return render_template('results.html', 
                             original_image=filename,
                             result_image=result_filename,
                             detections=detections,
                             total_faces=len(detections))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for mask detection
        processed_frame, detections = detector.process_image(frame.copy())
        
        # Add FPS counter
        cv2.putText(processed_frame, f"Faces: {len(detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)