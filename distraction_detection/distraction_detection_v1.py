import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import pygame
import json
from time import time

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CompleteDistractionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Initialize sound system
        pygame.mixer.init()
        self.sounds = self._load_sounds()
        
        # Initialize tracking variables
        self.start_time = time()
        self.distraction_log = []
        self.current_distraction = None
        self.distraction_start = None
        self.alert_cooldown = 3.0  # seconds
        self.last_alert_time = 0
        
        # Create log directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configuration
        self.config = {
            'mouth_threshold': 0.6,
            'looking_away_threshold': 0.3,
            'eyes_closed_threshold': 0.2,
            'alert_cooldown': 3.0
        }

    def _load_sounds(self):
        """Load sound files for different alerts"""
        sounds = {}
        sound_files = {
            'eating': 'sounds/eating_alert.wav',
            'looking_away': 'sounds/looking_away_alert.wav',
            'drowsy': 'sounds/drowsy_alert.wav',
            'general': 'sounds/general_alert.wav'
        }
        
        for key, file in sound_files.items():
            try:
                sounds[key] = pygame.mixer.Sound(file)
            except:
                print(f"Warning: Could not load sound file {file}")
                sounds[key] = None
        return sounds

    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # Vertical eye distances
        v1 = distance(eye_landmarks[1], eye_landmarks[5])
        v2 = distance(eye_landmarks[2], eye_landmarks[4])
        # Horizontal eye distance
        h = distance(eye_landmarks[0], eye_landmarks[3])
        
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def detect_distractions(self, face_landmarks, frame_shape):
        """Comprehensive distraction detection"""
        if not face_landmarks:
            return "No Face Detected", (0, 0, 255)

        image_height, image_width = frame_shape[:2]
        distractions = []
        
        # Get key landmarks
        nose_tip = face_landmarks.landmark[1]
        nose_x = int(nose_tip.x * image_width)
        
        # Check head position (looking away)
        if abs(nose_x - image_width/2) > image_width * self.config['looking_away_threshold']:
            distractions.append("Looking Away")
        
        # Check mouth for eating/drinking
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_distance = abs(upper_lip.y - lower_lip.y)
        if mouth_distance > self.config['mouth_threshold']:
            distractions.append("Eating/Drinking")
        
        # Check eyes for drowsiness
        left_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        if avg_ear < self.config['eyes_closed_threshold']:
            distractions.append("Drowsy")
        
        # Determine final status and color
        if not distractions:
            return "Focused", (0, 255, 0)
        else:
            self._log_distraction("/".join(distractions))
            return "/".join(distractions), (0, 0, 255)

    def _log_distraction(self, distraction_type):
        """Log distraction events"""
        current_time = time()
        
        # Start new distraction period
        if self.current_distraction != distraction_type:
            if self.current_distraction:
                # Log previous distraction
                duration = current_time - self.distraction_start
                self.distraction_log.append({
                    'type': self.current_distraction,
                    'start_time': datetime.fromtimestamp(self.distraction_start).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': round(duration, 2)
                })
            
            self.current_distraction = distraction_type
            self.distraction_start = current_time
            
            # Play alert sound if cooldown has passed
            if current_time - self.last_alert_time > self.alert_cooldown:
                self._play_alert(distraction_type)
                self.last_alert_time = current_time

    def _play_alert(self, distraction_type):
        """Play appropriate alert sound"""
        if 'eating' in distraction_type.lower() and self.sounds.get('eating'):
            self.sounds['eating'].play()
        elif 'looking' in distraction_type.lower() and self.sounds.get('looking_away'):
            self.sounds['looking_away'].play()
        elif 'drowsy' in distraction_type.lower() and self.sounds.get('drowsy'):
            self.sounds['drowsy'].play()
        elif self.sounds.get('general'):
            self.sounds['general'].play()

    def draw_ui(self, frame, status, color):
        """Draw the user interface"""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw status box
        cv2.rectangle(frame, (10, 10), (frame_width-10, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw recent distractions log
        y_offset = 90
        cv2.putText(frame, "Recent Distractions:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, log in enumerate(reversed(self.distraction_log[-5:])):
            y_offset += 25
            text = f"{log['start_time']} - {log['type']} ({log['duration']}s)"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw session duration
        session_duration = int(time() - self.start_time)
        cv2.putText(frame, f"Session Duration: {session_duration}s", 
                   (frame_width-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def save_session_log(self):
        """Save session log to file"""
        session_data = {
            'session_start': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration': round(time() - self.start_time, 2),
            'distractions': self.distraction_log
        }
        
        filename = f"logs/session_{int(self.start_time)}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=4)
        print(f"Session log saved to {filename}")

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw the detections
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
                
                # Detect distractions
                status, color = self.detect_distractions(face_landmarks, frame.shape)
                
                # Draw UI
                self.draw_ui(frame, status, color)
        else:
            # No face detected
            self.draw_ui(frame, "No Face Detected", (0, 0, 255))
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize detector
    detector = CompleteDistractionDetector()
    
    print("Starting distraction detection system...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current session log")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        frame = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Distraction Detection System', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            detector.save_session_log()
    
    # Clean up
    detector.save_session_log()  # Auto-save on exit
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Details:", str(e))