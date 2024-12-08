import cv2
import numpy as np
from datetime import datetime
import time
from ultralytics import YOLO
import os
import json

class VirtualProctor:
    def __init__(self):
        # Initialize YOLOv8 face detection model
        self.model = YOLO('yolov8n-face.pt')
        self.cap = None
        self.violation_count = 0
        self.last_violation_time = time.time()
        self.violation_cooldown = 3  # Seconds between logging violations
        self.confidence_threshold = 0.5  # Minimum confidence score for face detection
        
        # Add new tracking variables
        self.face_history = []
        self.max_history = 30  # Track last 30 frames
        self.movement_threshold = 0.3  # Threshold for significant movement (30% of frame width)
        self.absence_threshold = 15  # Frames before counting as absent
        self.frames_without_face = 0
        
        # Add violation types and statistics
        self.violation_types = {
            'no_face': {'severity': 'HIGH', 'message': 'No face detected'},
            'multiple_faces': {'severity': 'HIGH', 'message': 'Multiple faces detected'},
            'face_position': {'severity': 'MEDIUM', 'message': 'Face not centered'},
            'movement': {'severity': 'MEDIUM', 'message': 'Suspicious head movement'},
            'absence': {'severity': 'HIGH', 'message': 'Extended face absence'}
        }
        self.violation_stats = {vtype: 0 for vtype in self.violation_types.keys()}
        
        # Add logging directory setup
        self.log_dir = "proctoring_logs"
        self.screenshots_dir = os.path.join(self.log_dir, "screenshots")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.log")
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "violations": []
        }
        self._setup_logging_directories()
        
    def _setup_logging_directories(self):
        """Create necessary directories for logging"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def save_screenshot(self, frame, violation_type):
        """Save a screenshot of the violation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{violation_type}_{timestamp}.jpg"
        filepath = os.path.join(self.screenshots_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def start_monitoring(self):
        """Start the webcam monitoring session"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not access webcam")
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the processed frame
            cv2.imshow('Virtual Proctor', processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
        
    def process_frame(self, frame):
        """Process frame with screenshot capability"""
        # Store current frame for screenshot purposes
        self.current_frame = frame.copy()
        
        results = self.model(frame, conf=self.confidence_threshold)
        frame_width = frame.shape[1]
        
        # Get detected faces
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        
        # Track face position and movement
        if len(faces) == 1:
            self.frames_without_face = 0
            face = faces[0]
            face_center_x = face[0] + face[2]//2
            
            # Add face position to history
            self.face_history.append(face_center_x)
            if len(self.face_history) > self.max_history:
                self.face_history.pop(0)
            
            # Check for significant head movement
            if len(self.face_history) > 5:  # Need some history to detect movement
                movement = abs(self.face_history[-1] - self.face_history[-5])
                if movement > frame_width * self.movement_threshold:
                    self.log_violation("Suspicious head movement detected")
                    cv2.putText(frame, "Warning: Head movement!", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
            # Check if face is too far to the side
            if face_center_x < frame_width * 0.2 or face_center_x > frame_width * 0.8:
                self.log_violation("Face too far to the side")
                cv2.putText(frame, "Warning: Face not centered!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        else:
            self.frames_without_face += 1
            self.face_history.clear()  # Reset history when face is lost
            
        # Check for extended absence
        if self.frames_without_face >= self.absence_threshold:
            self.log_violation("Extended face absence detected")
            cv2.putText(frame, "Warning: Face absent for too long!", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Existing violation checks
        if len(faces) == 0:
            self.log_violation("No face detected")
            cv2.putText(frame, "No face detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(faces) > 1:
            self.log_violation("Multiple faces detected")
            cv2.putText(frame, "WARNING: Multiple faces detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame = cv2.rectangle(frame, (0, 0), 
                                (frame.shape[1]-1, frame.shape[0]-1), 
                                (0, 0, 255), 3)
            
        # Draw rectangles and confidence scores
        for (x, y, w, h, conf) in faces:
            border_color = (0, 0, 255) if len(faces) > 1 else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
            
        return frame
        
    def log_violation(self, violation_type):
        """Modified log_violation to work with GUI"""
        current_time = time.time()
        if current_time - self.last_violation_time >= self.violation_cooldown:
            self.violation_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Map violation type
            violation_key = None
            if "No face" in violation_type:
                violation_key = 'no_face'
            elif "Multiple faces" in violation_type:
                violation_key = 'multiple_faces'
            elif "Face too far" in violation_type:
                violation_key = 'face_position'
            elif "movement" in violation_type.lower():
                violation_key = 'movement'
            elif "absence" in violation_type.lower():
                violation_key = 'absence'
            
            if violation_key:
                severity = self.violation_types[violation_key]['severity']
                self.violation_stats[violation_key] += 1
                
                # Create violation record
                violation_record = {
                    "timestamp": timestamp,
                    "violation_type": violation_key,
                    "severity": severity,
                    "message": violation_type,
                    "violation_number": self.violation_count
                }
                
                # Save screenshot if we have a frame
                if hasattr(self, 'current_frame'):
                    screenshot_path = self.save_screenshot(self.current_frame, violation_key)
                    violation_record["screenshot"] = screenshot_path
                
                # Add to session data
                self.session_data["violations"].append(violation_record)
                
                # Write to log file
                with open(self.log_file, 'a') as f:
                    f.write(f"[{timestamp}] ALERT [{severity}] Violation #{self.violation_count}: "
                           f"{violation_type} (Total {violation_key}: {self.violation_stats[violation_key]})\n")
                
                # If GUI queue exists, add alert to it
                if hasattr(self, 'alert_queue'):
                    self.alert_queue.put({
                        'message': f"{violation_type} (#{self.violation_count})",
                        'severity': severity
                    })
                
                # Print to console
                print(f"[{timestamp}] ALERT [{severity}] Violation #{self.violation_count}: "
                      f"{violation_type} (Total {violation_key}: {self.violation_stats[violation_key]})")
                
            self.last_violation_time = current_time
        
    def get_violation_summary(self):
        """Get a summary of all violations detected during the session"""
        summary = "\nViolation Summary:\n" + "="*50 + "\n"
        for vtype, count in self.violation_stats.items():
            if count > 0:
                severity = self.violation_types[vtype]['severity']
                message = self.violation_types[vtype]['message']
                summary += f"- {message} [{severity}]: {count} occurrences\n"
        return summary
        
    def cleanup(self):
        """Enhanced cleanup with session summary"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Add end time to session data
        self.session_data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_data["total_violations"] = self.violation_count
        self.session_data["violation_stats"] = self.violation_stats
        
        # Save session summary
        summary_file = os.path.join(self.log_dir, f"session_{self.session_id}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(self.session_data, f, indent=4)
        
        # Print violation summary
        print(self.get_violation_summary())
        print(f"\nSession logs saved to: {self.log_dir}")

if __name__ == "__main__":
    proctor = VirtualProctor()
    try:
        proctor.start_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proctor.cleanup()
