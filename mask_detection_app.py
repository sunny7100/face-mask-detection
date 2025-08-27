import cv2
import numpy as np
from PIL import Image
import os
import time

class FaceMaskDetector:
    def __init__(self):
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def predict_mask(self, face_img):
        """Rule-based mask detection"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Focus on lower face region (where mask would be)
        h, w = gray.shape
        lower_face = gray[int(h*0.6):, :]
        
        # Calculate statistics
        mean_intensity = np.mean(lower_face)
        std_intensity = np.std(lower_face)
        
        # Mask detection logic
        if mean_intensity < 100 and std_intensity > 15:
            return "Mask", 0.92
        elif mean_intensity < 120 and std_intensity > 25:
            return "Mask", 0.85
        else:
            return "No Mask", 0.88
    
    def process_image(self, image_path):
        """Process image and detect masks"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, []
        
        # Detect faces
        faces = self.detect_faces(image)
        results = []
        
        print(f"Detected {len(faces)} faces")
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Predict mask
            label, confidence = self.predict_mask(face_roi)
            
            # Draw bounding box and label
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            results.append({
                'face_id': i+1,
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence
            })
            
            print(f"Face {i+1}: {label} ({confidence:.2%})")
        
        return image, results
    
    def live_detection(self):
        """Real-time mask detection using webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting live detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict mask
                label, confidence = self.predict_mask(face_roi)
                
                # Draw bounding box and label
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Face Mask Detection System")
    print("AI/ML Project using Haar Cascades + Computer Vision")
    print("=" * 50)
    
    detector = FaceMaskDetector()
    
    while True:
        print("\nOptions:")
        print("1. Process Image")
        print("2. Live Camera Detection")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            
            if os.path.exists(image_path):
                print(f"\nProcessing {image_path}...")
                start_time = time.time()
                
                result_image, detections = detector.process_image(image_path)
                processing_time = time.time() - start_time
                
                if result_image is not None:
                    print(f"\nProcessing completed in {processing_time:.3f}s")
                    print(f"Total faces detected: {len(detections)}")
                    
                    # Save result
                    output_path = f"result_{os.path.basename(image_path)}"
                    cv2.imwrite(output_path, result_image)
                    print(f"Result saved as: {output_path}")
                    
                    # Display results
                    for detection in detections:
                        print(f"Face {detection['face_id']}: {detection['label']} "
                              f"({detection['confidence']:.2%})")
                else:
                    print("Error: Could not process image")
            else:
                print("Error: Image file not found")
        
        elif choice == '2':
            detector.live_detection()
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()