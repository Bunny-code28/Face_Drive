import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from collections import deque
import json
import csv
from pathlib import Path

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DistractionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Hands detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Object Detection
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name='Cup')  # Cup detection for drinking
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, 
            circle_radius=1,
            color=(0, 255, 0)
        )
        
        # Landmark indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.MOUTH_INDICES = [61, 291, 39, 181, 0, 17]
        
        # Initialize counters
        self.eye_close_counter = 0
        self.mouth_movement_counter = 0
        self.looking_down_counter = 0
        self.hand_near_face_counter = 0
        self.food_detected_counter = 0
        self.eating_movement_counter = 0
        
        # Initialize detection history
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        self.detections_log = []
        
        # Configuration thresholds
        self.config = {
            'ear_threshold': 0.25,         # Eye Aspect Ratio threshold
            'mar_threshold': 0.5,          # Mouth Aspect Ratio threshold
            'drowsy_frames': 15,           # Number of frames for drowsy detection
            'eating_frames': 10,           # Number of frames for eating detection
            'looking_down_frames': 20,     # Number of frames for phone usage detection
            'hand_face_distance': 0.3,     # Threshold for hand-face proximity
            'food_frames': 10,             # Frames to confirm food presence
            'eating_mar_threshold': 0.6,   # Mouth aspect ratio for eating detection
            'eating_pattern_frames': 15    # Frames to confirm eating pattern
        }

        # Initialize logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Session information
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.full_session_log = []
        
        # Create log files
        self.log_files = {
            'json': self.log_dir / f"session_{self.session_id}.json",
            'csv': self.log_dir / f"session_{self.session_id}.csv"
        }
        
        # Initialize CSV file
        with open(self.log_files['csv'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Event Type', 'Duration', 'Details'])

    def get_landmarks_coords(self, landmarks, indices):
        """Extract landmark coordinates for given indices"""
        return np.array([[landmarks.landmark[idx].x, landmarks.landmark[idx].y] for idx in indices])

    def calculate_ratio(self, points):
        """Calculate aspect ratio for eyes or mouth"""
        try:
            # Vertical distances
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            # Horizontal distance
            h = np.linalg.norm(points[0] - points[3])
            
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except:
            return 0.0

    def analyze_eating_pattern(self, mouth_points):
        """Analyze mouth movements to detect eating patterns"""
        try:
            # Calculate current mouth movement
            mar = self.calculate_ratio(mouth_points)
            self.mar_history.append(mar)
            
            if len(self.mar_history) >= 10:
                # Check for rhythmic movement pattern
                movement_diff = np.diff(list(self.mar_history)[-10:])
                movement_std = np.std(movement_diff)
                
                # Rhythmic movement will have consistent differences
                if 0.05 < movement_std < 0.2:
                    return True
            
            return False
        except:
            return False

    def detect_food(self, frame, face_landmarks):
        """Detect food-related objects and eating behavior"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            food_results = self.objectron.process(rgb_frame)
            food_detected = False
            
            # Check for food-related objects (cups, containers)
            if food_results.detected_objects:
                for detected_object in food_results.detected_objects:
                    confidence = detected_object.score
                    if confidence > 0.5:
                        food_detected = True
                        # Draw bounding box
                        self.mp_drawing.draw_landmarks(
                            frame,
                            detected_object.landmarks_2d,
                            self.mp_objectron.BOX_CONNECTIONS)
            
            # Check for eating movements using mouth
            if face_landmarks:
                mouth_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] 
                                       for i in self.MOUTH_INDICES])
                
                # Detect eating pattern
                if self.analyze_eating_pattern(mouth_points):
                    self.eating_movement_counter += 1
                else:
                    self.eating_movement_counter = max(0, self.eating_movement_counter - 1)
                
                if self.eating_movement_counter >= self.config['eating_pattern_frames']:
                    food_detected = True
            
            if food_detected:
                self.food_detected_counter += 1
            else:
                self.food_detected_counter = max(0, self.food_detected_counter - 1)
            
            return self.food_detected_counter >= self.config['food_frames']
            
        except Exception as e:
            print(f"Error in food detection: {str(e)}")
            return False

    def detect_hands(self, frame, face_landmarks):
        """Detect hands and check if they're near the face"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            hands_near_face = False
            
            if results.multi_hand_landmarks and face_landmarks:
                # Get face center
                nose_tip = face_landmarks.landmark[1]
                face_x, face_y = nose_tip.x, nose_tip.y
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand center (using palm center)
                    palm_x = hand_landmarks.landmark[0].x
                    palm_y = hand_landmarks.landmark[0].y
                    
                    # Calculate distance between hand and face
                    distance = np.sqrt((face_x - palm_x)**2 + (face_y - palm_y)**2)
                    
                    if distance < self.config['hand_face_distance']:
                        hands_near_face = True
                        # Draw hand landmarks when near face
                        self.mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.drawing_spec,
                            self.drawing_spec
                        )
            
            return hands_near_face
        except Exception as e:
            print(f"Error in hand detection: {str(e)}")
            return False

    def log_event(self, event_type, details=""):
        """Log a single event"""
        timestamp = datetime.now()
        event = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event_type,
            'details': details
        }
        self.full_session_log.append(event)
        
        with open(self.log_files['csv'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                event['timestamp'],
                event['event_type'],
                '',
                details
            ])

    def save_session_log(self):
        """Save complete session log"""
        session_end = datetime.now()
        session_duration = (session_end - self.session_start).total_seconds()
        
        session_data = {
            'session_id': self.session_id,
            'start_time': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': session_end.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': session_duration,
            'events': self.full_session_log,
            'statistics': {
                'total_events': len(self.full_session_log),
                'drowsy_events': sum(1 for event in self.full_session_log if 'Drowsy' in event['event_type']),
                'phone_events': sum(1 for event in self.full_session_log if 'Phone' in event['event_type']),
                'eating_events': sum(1 for event in self.full_session_log if 'Eating' in event['event_type'])
            }
        }
        
        with open(self.log_files['json'], 'w') as f:
            json.dump(session_data, f, indent=4)
        
        print(f"\nSession log saved:")
        print(f"JSON: {self.log_files['json']}")
        print(f"CSV: {self.log_files['csv']}")
        print("\nSession Summary:")
        print(f"Duration: {session_duration:.2f} seconds")
        print(f"Total Events: {session_data['statistics']['total_events']}")
        print(f"Drowsy Events: {session_data['statistics']['drowsy_events']}")
        print(f"Phone Usage Events: {session_data['statistics']['phone_events']}")
        print(f"Eating Events: {session_data['statistics']['eating_events']}")

    def detect_distractions(self, face_landmarks, frame_shape, frame):
        """Detect various types of distractions"""
        if not face_landmarks:
            return "No Face Detected", (0, 0, 255)
        
        h, w = frame_shape[:2]
        
        # Get eye coordinates
        left_eye = self.get_landmarks_coords(face_landmarks, self.LEFT_EYE_INDICES)
        right_eye = self.get_landmarks_coords(face_landmarks, self.RIGHT_EYE_INDICES)
        
        # Calculate eye aspect ratios
        ear_left = self.calculate_ratio(left_eye)
        ear_right = self.calculate_ratio(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0
        self.ear_history.append(ear_avg)
        
        # Initialize distractions list
        distractions = []
        
        # Detect drowsiness
        if ear_avg < self.config['ear_threshold']:
            self.eye_close_counter += 1
            if self.eye_close_counter >= self.config['drowsy_frames']:
                distractions.append("Drowsy")
        else:
            self.eye_close_counter = max(0, self.eye_close_counter - 1)
        
        # Check head position (for phone usage)
        nose_y = face_landmarks.landmark[1].y
        if nose_y > 0.7:  # If nose is in lower part of frame
            self.looking_down_counter += 1
            if self.looking_down_counter >= self.config['looking_down_frames']:
                distractions.append("Looking Down")
        else:
            self.looking_down_counter = max(0, self.looking_down_counter - 1)
        
        # Check for hand-face proximity (possible phone usage)
        if self.detect_hands(frame, face_landmarks):
            self.hand_near_face_counter += 1
            if self.hand_near_face_counter >= self.config['looking_down_frames']:
                distractions.append("Possible Phone Usage")
        else:
            self.hand_near_face_counter = max(0, self.hand_near_face_counter - 1)

        # Check for food and eating behavior
        if self.detect_food(frame, face_landmarks):
            distractions.append("Eating/Drinking")
        
        # Log detection
        if distractions:
            self.detections_log.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'types': distractions
            })
            if len(self.detections_log) > 10:  # Keep last 10 detections
                self.detections_log.pop(0)
            return " + ".join(distractions), (0, 0, 255)
        
        return "Focused", (0, 255, 0)

    def draw_ui(self, frame, status, color):
        """Draw user interface elements"""
        h, w = frame.shape[:2]
        
        # Draw status box
        cv2.rectangle(frame, (10, 10), (w-10, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw metrics
        y_offset = 100
        metrics = [
            f"Eye Closure: {self.eye_close_counter}/{self.config['drowsy_frames']}",
            f"Looking Down: {self.looking_down_counter}/{self.config['looking_down_frames']}",
            f"Hand Near Face: {self.hand_near_face_counter}/{self.config['looking_down_frames']}",
            f"Eating Detection: {self.food_detected_counter}/{self.config['food_frames']}"
        ]
        for metric in metrics:
            cv2.putText(frame, metric, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
        
        # Draw recent detections
        y_offset += 20
        cv2.putText(frame, "Recent Detections:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        for detection in reversed(self.detections_log[-5:]):
            text = f"{detection['timestamp']} - {'+'.join(detection['types'])}"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

    def process_frame(self, frame):
        """Process each frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
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
                status, color = self.detect_distractions(face_landmarks, frame.shape, frame)  # Pass frame
                
                # Draw UI
                self.draw_ui(frame, status, color)
                break  # Process only first face
        else:
            self.draw_ui(frame, "No Face Detected", (0, 0, 255))
        
        return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize detector
    detector = DistractionDetector()
    
    print("\nStarting distraction detection system...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset counters")
    print("- Press 's' to save session log")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = detector.process_frame(frame)
            cv2.imshow('Distraction Detection System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nSaving final session log...")
                detector.save_session_log()
                break
            elif key == ord('r'):
                detector.eye_close_counter = 0
                detector.looking_down_counter = 0
                detector.hand_near_face_counter = 0
                detector.food_detected_counter = 0
                detector.eating_movement_counter = 0
                detector.detections_log.clear()
                detector.log_event("Reset", "Counters reset by user")
                print("Counters reset")
            elif key == ord('s'):
                detector.save_session_log()
                print("Session log saved")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()