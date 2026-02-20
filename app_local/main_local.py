import sys
import os
import cv2
import time
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.detector import ViolenceDetector

from collections import deque
import datetime
import json
import requests

# Ensure results directory exists
RESULTS_DIR = r"e:\Violence_Detection_System\results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Ensure logs directory exists
LOGS_DIR = r"e:\Violence_Detection_System\logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
LOG_FILE = os.path.join(LOGS_DIR, "violence_history.json")

# Web API endpoint (used when app_web server is running)
API_URL = "http://127.0.0.1:5000/api/logs"

def log_event(event: dict):
    """POST event to web API. Falls back to direct file write if server is offline."""
    try:
        r = requests.post(API_URL, json=event, timeout=2)
        if r.status_code == 201:
            print(f"[LOG] Event sent to API: {event['event_type']}")
            return
    except Exception:
        pass  # Server not running, fall back to local write

    # Fallback: write directly to JSON file
    history = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    history.append(event)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"[LOG] Event saved locally (server offline): {event['event_type']}")

def calculate_motion(frames):
    """Calculate average pixel difference between consecutive frames"""
    if not frames or len(frames) < 2:
        return 0.0
    
    diff_sum = 0.0
    # Convert frames to grayscale for simpler motion calc
    # frames are RGB numpy arrays
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    for i in range(len(gray_frames) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
        diff_sum += np.mean(diff)
        
    return diff_sum / (len(frames) - 1)

def main():
    MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
    CAMERA_ID = 0 
    
    print("Initializing Violence Detection System...")
    detector = ViolenceDetector(model_path=MODEL_PATH)
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return

    print("Camera started. Press 'q' to quit.")
    
    frames_buffer = []
    FPS = 0
    frame_count = 0
    
    # Smoothing parameters
    prediction_history = deque(maxlen=5) # Keep last 5 predictions
    current_label = "Normal"
    current_conf = 0.0
    
    # Recording state
    is_recording = False
    out = None
    save_path = ""
    start_time = None
    max_conf = 0.0
    event_label = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = cv2.resize(frame, (800, 600))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Temporal Sampling: Every 3rd frame
        if frame_count % 3 == 0:
            frames_buffer.append(rgb_frame)
        
        if len(frames_buffer) > 16:
            frames_buffer.pop(0)
            
        # Inference Logic
        if len(frames_buffer) == 16 and frame_count % 5 == 0:
             # 1. Motion Gate
             motion_score = calculate_motion(frames_buffer)
             
             if motion_score < 5.0: # Threshold for "Stillness"
                 # If practically no motion, force "Normal"
                 pred_label = "Normal (Static)"
                 pred_conf = 1.0
             else:
                 # 2. Run AI Model
                 label, conf = detector.predict(frames_buffer)
                 pred_label = label
                 pred_conf = conf
                 
             # 3. Smoothing / Voting
             prediction_history.append((pred_label, pred_conf))
             
             # Count "Violence" votes in history
             violence_votes = sum(1 for p in prediction_history if p[0] in ['danh_nhau', 'nga'])
             
             if violence_votes >= 3: # Majority vote
                 current_label = prediction_history[-1][0] # Use latest violence label
                 current_conf = prediction_history[-1][1]
             else:
                 current_label = "Normal"
                 current_conf = 0.0
                 
             print(f"Frame {frame_count}: Motion={motion_score:.2f} | Raw={pred_label} ({pred_conf:.2f}) | Final={current_label}")

        # Draw Output
        color = (0, 255, 0) # Green
        if current_label in ['danh_nhau', 'nga']:
            color = (0, 0, 255) # Red
            cv2.rectangle(display_frame, (0, 0), (800, 600), color, 10)
            cv2.putText(display_frame, f"ALERT: {current_label.upper()}!", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        cv2.putText(display_frame, f"Status: {current_label}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Recording Logic
        if current_label in ['danh_nhau', 'nga']:
            if not is_recording:
                # Start Recording
                start_time = datetime.datetime.now()
                timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(RESULTS_DIR, f"violence_{timestamp}.mp4")
                # Use avc1 (H.264) codec – natively supported by all browsers
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (800, 600))
                is_recording = True
                max_conf = current_conf
                event_label = current_label
                print(f"[REC] Recording started: {save_path}")
            
            if out is not None:
                out.write(display_frame)
                cv2.circle(display_frame, (750, 50), 10, (0, 0, 255), -1) # Red recording dot
                if current_conf > max_conf:
                    max_conf = current_conf
        
        else:
            # excessive logic: Stop recording if it was recording
            if is_recording:
                if out is not None:
                    out.release()
                    out = None
                is_recording = False
                
                # Save JSON Metadata to Central Log
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                new_event = {
                    "source":           "webcam",
                    "event_type":       event_label,
                    "start_time":       start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time":         end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": round(duration, 2),
                    "max_confidence":   float(round(max_conf, 4)),
                    "video_file":       os.path.basename(save_path)
                }
                
                # Submit event via API (or fallback to local file)
                log_event(new_event)
                print(f"[REC] Recording saved to {save_path}")
        
        cv2.imshow('Violence Detection System - Local', display_frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




