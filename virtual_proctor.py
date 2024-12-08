import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
import logging

class VirtualProctor:
    def __init__(self):
        # Load face detection model files from the models directory
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Download and load face detection model if not exists
        self.load_face_detector()
        
        self.cap = None
        
        # Define violation types
        self.violation_types = {
            'no_face': {'message': 'No Face Detected', 'severity': 'HIGH'},
            'multiple_faces': {'message': 'Multiple Faces Detected', 'severity': 'HIGH'},
            'looking_away': {'message': 'Looking Away', 'severity': 'MEDIUM'},
            'phone_detected': {'message': 'Phone Detected', 'severity': 'HIGH'},
            'book_detected': {'message': 'Book Detected', 'severity': 'HIGH'},
            'cheating_suspected': {'message': 'Cheating Suspected', 'severity': 'HIGH'}
        }
        
        # Add frames counter
        self.frames_without_face = 0
        self.frames_with_multiple_faces = 0
        
        # Initialize violation stats
        self.violation_stats = {vtype: 0 for vtype in self.violation_types}
        
        # Setup logging system
        self.setup_logging()
        
        # Session information
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = datetime.now()
        
        # Violation tracking
        self.violation_count = 0
        self.last_violation_time = datetime.now()
        self.violation_cooldown = 3  # seconds between similar violations
        
        # Create session directory
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.session_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Initialize session log
        self.session_log = {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "violations": [],
            "statistics": {}
        }
        
        # Enhanced phone detection parameters
        self.head_position_history = []
        self.looking_down_frames = 0
        self.looking_down_threshold = 5      # Reduced threshold for quicker detection
        self.head_tilt_threshold = 0.65      # Adjusted threshold for head tilt (lower value = more sensitive)
        self.max_history = 30
        self.phone_detection_cooldown = 20   # Frames between phone detections
        
        # Load both face and object detection models
        self.load_face_detector()
        self.load_object_detector()  # New YOLO model for objects
        
        # Add object detection classes
        self.object_classes = ['cell phone', 'book', 'laptop', 'tablet']
        
        # Add looking away detection parameters
        self.face_center_history = []
        self.looking_away_frames = 0
        self.looking_away_threshold = 5  # Frames threshold for looking away detection
        self.center_threshold = 0.2      # How far from center before considered looking away
        self.stable_position_frames = 0   # Counter for stable head position

    def setup_logging(self):
        """Setup logging configuration"""
        self.log_dir = Path("proctoring_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "proctor.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("VirtualProctor")

    def load_face_detector(self):
        """Load DNN face detector model"""
        try:
            # Model files
            model_file = self.models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = self.models_dir / "deploy.prototxt"
            
            # Download model files if they don't exist
            if not model_file.exists() or not config_file.exists():
                print("Downloading face detection model files...")
                self.download_model_files(model_file, config_file)
            
            # Load model
            self.face_net = cv2.dnn.readNet(str(model_file), str(config_file))
            print("Face detection model loaded successfully")
            
        except Exception as e:
            print(f"Error loading face detector: {str(e)}")
            raise

    def download_model_files(self, model_file, config_file):
        """Download required model files"""
        import urllib.request
        
        # URLs for model files
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        
        # Download files
        urllib.request.urlretrieve(model_url, model_file)
        urllib.request.urlretrieve(config_url, config_file)

    def detect_faces(self, frame):
        """Detect faces using DNN model"""
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        h, w = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        
        return faces

    def save_screenshot(self, frame, violation_type):
        """Save screenshot of violation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{violation_type}_{timestamp}.jpg"
        filepath = self.screenshots_dir / filename
        cv2.imwrite(str(filepath), frame)
        return str(filepath)

    def log_violation(self, violation_type, frame, details=None):
        """Log a violation with screenshot and details"""
        current_time = datetime.now()
        
        # Check cooldown period
        if (current_time - self.last_violation_time).total_seconds() < self.violation_cooldown:
            return
        
        self.violation_count += 1
        self.last_violation_time = current_time
        
        # Save screenshot
        screenshot_path = self.save_screenshot(frame, violation_type)
        
        # Create violation record
        violation_record = {
            "violation_id": self.violation_count,
            "timestamp": current_time.isoformat(),
            "type": violation_type,
            "severity": self.violation_types[violation_type]['severity'],
            "message": self.violation_types[violation_type]['message'],
            "screenshot": screenshot_path,
            "details": details or {}
        }
        
        # Add to session log
        self.session_log["violations"].append(violation_record)
        
        # Log to file
        self.logger.warning(
            f"Violation #{self.violation_count} - {violation_type}: "
            f"{self.violation_types[violation_type]['message']}"
        )
        
        # Save updated session log
        self.save_session_log()
        
        return violation_record

    def save_session_log(self):
        """Save current session log to file"""
        try:
            # Update statistics before saving
            self.session_log["statistics"] = {
                "total_violations": self.violation_count,
                "violation_counts": self.violation_stats,
                "session_duration": (datetime.now() - self.session_start_time).total_seconds()
            }
            
            log_file = self.session_dir / "session_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.session_log, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Error saving session log: {str(e)}")

    def load_object_detector(self):
        """Load YOLO object detection model"""
        try:
            from ultralytics import YOLO
            self.object_model = YOLO('yolov8n.pt')  # Load the general object detection model
            print("Object detection model loaded successfully")
        except Exception as e:
            print(f"Error loading object detector: {str(e)}")
            raise

    def detect_objects(self, frame):
        """Detect objects (phones, books, etc.) using YOLO"""
        results = self.object_model(frame, conf=0.4)  # Lower confidence threshold for objects
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                if class_name in self.object_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_objects.append({
                        'class': class_name,
                        'confidence': conf,
                        'box': (x1, y1, x2-x1, y2-y1)
                    })
        
        return detected_objects

    def process_frame(self, frame):
        """Process frame with both face and object detection"""
        try:
            if frame is None:
                return frame, [], []
            
            processed_frame = frame.copy()
            faces = self.detect_faces(frame)
            violations = []
            
            frame_height, frame_width = processed_frame.shape[:2]
            frame_center_x = frame_width / 2
            
            # Process faces and check for looking away
            if len(faces) == 1:  # Only check looking away for single face
                face = faces[0]
                face_x, face_y, face_w, face_h = face[:4]
                face_center_x = face_x + face_w/2
                
                # Calculate how far face is from center (normalized)
                center_offset = abs(face_center_x - frame_center_x) / frame_width
                
                # Track face center position
                self.face_center_history.append(face_center_x)
                if len(self.face_center_history) > 10:  # Keep last 10 positions
                    self.face_center_history.pop(0)
                
                # Check if looking away
                if center_offset > self.center_threshold:
                    self.looking_away_frames += 1
                    if self.looking_away_frames >= self.looking_away_threshold:
                        # Determine direction
                        direction = "right" if face_center_x > frame_center_x else "left"
                        self.violation_stats['looking_away'] += 1
                        violation = self.log_violation('looking_away', frame, {
                            'direction': direction,
                            'offset': center_offset,
                            'frames': self.looking_away_frames
                        })
                        if violation:
                            violations.append(violation)
                            
                        # Draw looking away indicator
                        color = (0, 0, 255)  # Red
                        arrow_length = 50
                        arrow_x = face_center_x + (arrow_length if direction == "right" else -arrow_length)
                        cv2.arrowedLine(processed_frame, 
                                      (int(face_center_x), int(face_y + face_h//2)),
                                      (int(arrow_x), int(face_y + face_h//2)),
                                      color, 2, tipLength=0.3)
                else:
                    self.looking_away_frames = max(0, self.looking_away_frames - 1)
                    self.stable_position_frames += 1
                
                # Draw face center and guidelines
                cv2.circle(processed_frame, (int(face_center_x), int(face_y + face_h//2)), 
                          3, (0, 255, 0), -1)
                
                # Draw center region boundaries
                left_boundary = int(frame_center_x - frame_width * self.center_threshold)
                right_boundary = int(frame_center_x + frame_width * self.center_threshold)
                cv2.line(processed_frame, (left_boundary, 0), (left_boundary, frame_height), 
                        (100, 100, 100), 1)
                cv2.line(processed_frame, (right_boundary, 0), (right_boundary, frame_height), 
                        (100, 100, 100), 1)
            
            else:
                self.looking_away_frames = 0
                self.stable_position_frames = 0
                self.face_center_history.clear()
            
            # Detect objects (phones, books, etc.)
            objects = self.detect_objects(frame)
            
            # Process detected objects
            for obj in objects:
                x, y, w, h = obj['box']
                class_name = obj['class']
                conf = obj['confidence']
                
                # Draw object box
                color = (0, 0, 255)  # Red for prohibited items
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(processed_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Log violations for detected objects
                if class_name == 'cell phone':
                    self.violation_stats['phone_detected'] += 1
                    violation = self.log_violation('phone_detected', frame, {
                        'confidence': conf,
                        'location': f'x:{x}, y:{y}'
                    })
                    if violation:
                        violations.append(violation)
                
                elif class_name == 'book':
                    self.violation_stats['book_detected'] += 1
                    violation = self.log_violation('book_detected', frame, {
                        'confidence': conf,
                        'location': f'x:{x}, y:{y}'
                    })
                    if violation:
                        violations.append(violation)
                
                # Log general cheating violation
                if class_name in ['cell phone', 'book', 'laptop', 'tablet']:
                    self.violation_stats['cheating_suspected'] += 1
                    violation = self.log_violation('cheating_suspected', frame, {
                        'object_type': class_name,
                        'confidence': conf
                    })
                    if violation:
                        violations.append(violation)
            
            # Process faces (existing code)
            if len(faces) == 0:
                self.frames_without_face += 1
                self.violation_stats['no_face'] += 1
                violation = self.log_violation('no_face', frame, {
                    "frames_without_face": self.frames_without_face
                })
                if violation:
                    violations.append(violation)
            
            elif len(faces) > 1:
                self.violation_stats['multiple_faces'] += 1
                violation = self.log_violation('multiple_faces', frame, {
                    "face_count": len(faces)
                })
                if violation:
                    violations.append(violation)
            
            # Draw faces
            for (x, y, w, h, conf) in faces:
                color = (0, 0, 255) if len(faces) > 1 else (255, 0, 0)
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(processed_frame, f"Face: {conf:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return processed_frame, faces, violations
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {str(e)}")
            return frame, [], []

    def cleanup(self):
        """Cleanup and save final session data"""
        try:
            if self.cap:
                self.cap.release()
            
            # Add session end time
            self.session_log["end_time"] = datetime.now().isoformat()
            
            # Calculate statistics
            self.session_log["statistics"] = {
                "total_violations": self.violation_count,
                "violation_counts": self.violation_stats,
                "session_duration": (datetime.now() - self.session_start_time).total_seconds()
            }
            
            # Save final session log
            self.save_session_log()
            
            # Generate session summary
            summary = self.generate_session_summary()
            summary_file = self.session_dir / "session_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Session {self.session_id} completed and saved")
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {str(e)}")

    def generate_session_summary(self):
        """Generate a human-readable session summary"""
        summary = [
            f"Session Summary - {self.session_id}",
            "=" * 50,
            f"Start Time: {self.session_start_time}",
            f"End Time: {datetime.now()}",
            f"Total Violations: {self.violation_count}",
            "\nViolation Statistics:",
            "-" * 20
        ]
        
        for vtype, count in self.violation_stats.items():
            if count > 0:
                summary.append(f"{self.violation_types[vtype]['message']}: {count}")
        
        return "\n".join(summary)