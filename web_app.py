from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
        
        # Focus on lower face (mouth/nose area)
        lower_face = gray[int(h*0.5):int(h*0.85), int(w*0.25):int(w*0.75)]
        
        if lower_face.size == 0:
            return "No Mask", 0.75
        
        # Calculate brightness in mouth area
        brightness = np.mean(lower_face)
        
        # Edge detection for facial features
        edges = cv2.Canny(lower_face, 30, 100)
        edge_count = np.sum(edges > 0)
        edge_ratio = edge_count / lower_face.size
        
        # Standard deviation (texture)
        texture = np.std(lower_face)
        
        # Decision logic: visible mouth/nose features = no mask
        # High edges + high texture + normal brightness = no mask
        if edge_ratio > 0.04 and texture > 25 and 50 < brightness < 200:
            return "No Mask", 0.88
        # Low edges + low texture = mask covering face
        elif edge_ratio < 0.02 and texture < 15:
            return "With Mask", 0.91
        # Medium values - use brightness as tie-breaker
        elif brightness > 120:
            return "No Mask", 0.82
        else:
            return "With Mask", 0.85
    
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
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        result_image, detections = detector.process_image(image.copy())
        
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

# Global variables for stable detection
stable_detections = {}
frame_counter = 0

def generate_frames():
    global stable_detections, frame_counter
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Process every frame for real-time accuracy
        _, current_detections = detector.process_image(frame.copy())
        
        # Use current detections directly for accuracy
        stable_detections = {}
        for i, detection in enumerate(current_detections):
            face_key = f"face_{i}"
            stable_detections[face_key] = detection
        
        # Draw stable detections
        display_frame = frame.copy()
        for detection in stable_detections.values():
            x, y, w, h = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(display_frame, f"{label}: {confidence:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(display_frame, f"Faces: {len(stable_detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

if __name__ == '__main__':
    app.run(debug=True, port=8080)