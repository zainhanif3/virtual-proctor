# Virtual Proctor System Documentation

## 1. System Overview

The Virtual Proctor is an AI-powered monitoring system designed to detect and prevent cheating during remote examinations. The system uses computer vision techniques to monitor student behavior in real-time through their webcam.

### 1.1 Key Features
- Real-time face detection and tracking
- Multiple violation detection types
- GUI-based monitoring interface
- Comprehensive logging system
- Screenshot capture of violations
- Statistical analysis of violations

## 2. Technical Architecture

### 2.1 Core Components
1. **Face Detection Engine**
   - Uses YOLOv8 model (yolov8n-face.pt)
   - Real-time processing at frame level
   - Confidence threshold: 0.5

2. **Violation Detection System**
   - Multiple face detection
   - Face absence detection
   - Head movement tracking
   - Position monitoring

3. **Logging System**
   - JSON-based session logging
   - Screenshot capture
   - Violation statistics
   - Real-time alerts

4. **Graphical Interface**
   - Live video feed display
   - Real-time violation alerts
   - Statistical dashboard
   - Status indicators

### 2.2 Class Structure 