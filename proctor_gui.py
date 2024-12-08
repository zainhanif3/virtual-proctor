import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from virtual_proctor import VirtualProctor
from datetime import datetime
import queue
import numpy as np

class ProctorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Proctor Monitor")
        self.root.state('zoomed')  # Start maximized
        
        # Create queue for thread-safe communication
        self.alert_queue = queue.Queue()
        
        # Initialize the VirtualProctor
        self.proctor = VirtualProctor()
        
        self.last_face_detected = True
        
        self._setup_gui()
        self._setup_video()
        
        # Start alert processing
        self.process_alerts()
        
    def _setup_gui(self):
        """Setup the GUI layout"""
        # Main container with two panels
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for video
        self.left_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.left_panel, weight=2)
        
        # Video label
        self.video_label = ttk.Label(self.left_panel)
        self.video_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Right panel for controls and alerts
        self.right_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.right_panel, weight=1)
        
        # Status section
        status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: ")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.status_indicator = ttk.Label(status_frame, text="●", font=('Arial', 16))
        self.status_indicator.pack(side=tk.LEFT)
        self.update_status("OK")
        
        # Control buttons
        control_frame = ttk.LabelFrame(self.right_panel, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start Monitoring", 
                  command=self.start_monitoring).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_monitoring).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(self.right_panel, text="Violation Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stat_labels = {}
        for vtype, vinfo in self.proctor.violation_types.items():
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{vinfo['message']}: ").pack(side=tk.LEFT, padx=5)
            self.stat_labels[vtype] = ttk.Label(frame, text="0")
            self.stat_labels[vtype].pack(side=tk.LEFT)
        
        # Total violations
        total_frame = ttk.Frame(stats_frame)
        total_frame.pack(fill=tk.X, pady=2)
        ttk.Label(total_frame, text="Total Violations: ").pack(side=tk.LEFT, padx=5)
        self.total_violations_label = ttk.Label(total_frame, text="0")
        self.total_violations_label.pack(side=tk.LEFT)
        
        # Alerts section
        alerts_frame = ttk.LabelFrame(self.right_panel, text="Live Alerts")
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Alert text widget with scrollbar
        self.alert_text = tk.Text(alerts_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(alerts_frame, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=scrollbar.set)
        
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_video(self):
        """Initialize video capture"""
        self.running = False
        self.video_thread = None
    
    def update_status(self, status):
        """Update status indicator"""
        if status == "OK":
            self.status_indicator.configure(text="●", foreground="green")
        elif status == "WARNING":
            self.status_indicator.configure(text="●", foreground="orange")
        else:  # ERROR
            self.status_indicator.configure(text="●", foreground="red")
        
        self.status_label.configure(text=f"Status: {status}")
    
    def add_alert(self, alert_text, severity="INFO"):
        """Add alert to the alert text box"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {
            "HIGH": "red",
            "MEDIUM": "orange",
            "LOW": "blue",
            "INFO": "black"
        }.get(severity, "black")
        
        self.alert_text.tag_config(severity, foreground=color)
        self.alert_text.insert(tk.END, f"[{timestamp}] ", "INFO")
        self.alert_text.insert(tk.END, f"{alert_text}\n", severity)
        self.alert_text.see(tk.END)
    
    def update_stats(self):
        """Update violation statistics"""
        try:
            for vtype, count in self.proctor.violation_stats.items():
                if vtype in self.stat_labels:
                    self.stat_labels[vtype].configure(text=str(count))
                    
            # Update total violations
            total = sum(self.proctor.violation_stats.values())
            if hasattr(self, 'total_violations_label'):
                self.total_violations_label.configure(text=str(total))
                
        except Exception as e:
            print(f"Error updating stats: {str(e)}")
    
    def process_alerts(self):
        """Process alerts from the queue"""
        try:
            while not self.alert_queue.empty():
                try:
                    alert = self.alert_queue.get_nowait()
                    self.add_alert(alert['message'], alert['severity'])
                    self.update_stats()
                    
                    if alert['severity'] == "HIGH":
                        self.update_status("WARNING")
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error processing alert: {str(e)}")
                    
        except Exception as e:
            print(f"Alert processing error: {str(e)}")
        finally:
            # Schedule next check
            if not self.root.winfo_exists():
                return
            self.root.after(100, self.process_alerts)
    
    def update_frame(self, frame):
        """Update the video frame"""
        try:
            if frame is None:
                return
            
            # Convert frame to numpy array if it isn't already
            if not isinstance(frame, np.ndarray):
                self.add_alert("Invalid frame format", "HIGH")
                return
            
            # Convert only if frame is not already in RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure frame dimensions are valid
            height, width = frame.shape[:2]
            if height > 0 and width > 0:
                frame = cv2.resize(frame, (800, 600))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference!
            
        except Exception as e:
            print(f"Frame update error: {str(e)}")  # Add print for debugging
            self.add_alert(f"Error updating frame: {str(e)}", "HIGH")
    
    def video_loop(self):
        """Main video processing loop"""
        try:
            while self.running:
                if not self.proctor.cap or not self.proctor.cap.isOpened():
                    self.update_status("ERROR")
                    self.add_alert("Camera disconnected", "HIGH")
                    break
                    
                ret, frame = self.proctor.cap.read()
                if not ret:
                    continue
                    
                try:
                    # Unpack the returned tuple from process_frame
                    processed_frame, faces, violations = self.proctor.process_frame(frame)
                    
                    # Handle face detection status - check length of faces list
                    if len(faces) == 0:
                        if self.last_face_detected:  # Only alert once when face disappears
                            self.alert_queue.put({
                                'message': 'No face detected in frame',
                                'severity': 'HIGH'
                            })
                            self.last_face_detected = False
                    else:
                        if not self.last_face_detected:  # Face has reappeared
                            self.alert_queue.put({
                                'message': 'Face detected',
                                'severity': 'INFO'
                            })
                        self.last_face_detected = True

                    # Handle multiple faces
                    if len(faces) > 1:
                        self.alert_queue.put({
                            'message': f'Multiple faces detected ({len(faces)} faces)',
                            'severity': 'HIGH'
                        })

                    # Handle other violations
                    if violations:
                        for violation in violations:
                            self.alert_queue.put(violation)

                    # Update the display with the processed frame
                    self.update_frame(processed_frame)
                    
                except Exception as e:
                    self.add_alert(f"Frame processing error: {str(e)}", "HIGH")
                    self.update_status("ERROR")
                    continue
                    
        except Exception as e:
            self.add_alert(f"Video loop error: {str(e)}", "HIGH")
            self.update_status("ERROR")
        finally:
            self.running = False
    
    def start_monitoring(self):
        """Start video monitoring"""
        if self.running:
            return
        
        try:
            self.proctor.cap = cv2.VideoCapture(0)
            if not self.proctor.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Test first frame
            ret, frame = self.proctor.cap.read()
            if not ret:
                raise Exception("Failed to read from camera")
            
            self.running = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            self.update_status("OK")
            self.add_alert("Monitoring started", "INFO")
            
        except Exception as e:
            self.update_status("ERROR")
            self.add_alert(f"Failed to start monitoring: {str(e)}", "HIGH")
            if self.proctor.cap:
                self.proctor.cap.release()
    
    def stop_monitoring(self):
        """Stop video monitoring"""
        if not self.running:
            return
        
        try:
            self.running = False
            if self.video_thread:
                self.video_thread.join(timeout=1.0)  # Wait max 1 second
            
            if self.proctor.cap:
                self.proctor.cap.release()
            
            self.update_status("OK")
            self.add_alert("Monitoring stopped", "INFO")
            
        except Exception as e:
            self.update_status("ERROR")
            self.add_alert(f"Error stopping monitoring: {str(e)}", "HIGH")
        finally:
            self.running = False
            if self.proctor.cap:
                self.proctor.cap.release()
    
    def on_closing(self):
        """Cleanup on window closing"""
        try:
            self.stop_monitoring()
        except:
            pass
        finally:
            self.root.quit()
            self.root.destroy()

def main():
    try:
        root = tk.Tk()
        app = ProctorGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")

if __name__ == "__main__":
    main() 