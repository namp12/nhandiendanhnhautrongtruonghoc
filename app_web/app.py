import sys
import os
import cv2
import time
import json
import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.detector import ViolenceDetector

app = Flask(__name__)

# Configuration
MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
RESULTS_DIR = r"e:\Violence_Detection_System\results"
LOGS_DIR = r"e:\Violence_Detection_System\logs"
LOG_FILE = os.path.join(LOGS_DIR, "violence_history.json")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Global Variables
camera = None
detector = None

def get_detector():
    global detector
    if detector is None:
        print("Loading Model...")
        detector = ViolenceDetector(model_path=MODEL_PATH)
    return detector

def calculate_motion(frames):
    """Calculate average pixel difference between consecutive frames"""
    if not frames or len(frames) < 2:
        return 0.0
    diff_sum = 0.0
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    for i in range(len(gray_frames) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
        diff_sum += np.mean(diff)
    return diff_sum / (len(frames) - 1)

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return

    frames_buffer = []
    prediction_history = deque(maxlen=5)
    frame_count = 0
    
    # State variables
    current_label = "Normal"
    # Recording
    is_recording = False
    out = None
    save_path = ""
    start_time = None
    max_conf = 0.0
    event_label = ""
    
    detector_instance = get_detector()
    import numpy as np # Ensure numpy is available inside generator if needed

    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Resize for consistent processing
        display_frame = cv2.resize(frame, (800, 600))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Buffer logic
        if frame_count % 3 == 0:
            frames_buffer.append(rgb_frame)
        if len(frames_buffer) > 16:
            frames_buffer.pop(0)
            
        # Inference Logic
        if len(frames_buffer) == 16 and frame_count % 5 == 0:
             # Motion Gate
             motion_score = calculate_motion(frames_buffer)
             if motion_score < 5.0:
                 pred_label = "Normal (Static)"
                 pred_conf = 1.0
             else:
                 label, conf = detector_instance.predict(frames_buffer)
                 pred_label = label
                 pred_conf = conf
                 
             # Smoothing
             prediction_history.append((pred_label, pred_conf))
             violence_votes = sum(1 for p in prediction_history if p[0] in ['danh_nhau', 'nga'])
             
             if violence_votes >= 3:
                 current_label = prediction_history[-1][0]
                 current_conf = prediction_history[-1][1]
             else:
                 current_label = "Normal"
                 current_conf = 0.0

        # Draw UI
        color = (0, 255, 0)
        if current_label in ['danh_nhau', 'nga']:
            color = (0, 0, 255)
            cv2.rectangle(display_frame, (0, 0), (800, 600), color, 10)
            cv2.putText(display_frame, f"ALERT: {current_label.upper()}!", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        cv2.putText(display_frame, f"Status: {current_label}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Recording Logic
        if current_label in ['danh_nhau', 'nga']:
            if not is_recording:
                start_time = datetime.datetime.now()
                timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(RESULTS_DIR, f"violence_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (800, 600))
                is_recording = True
                max_conf = current_conf
                event_label = current_label
                print(f"[WEB] Recording started: {save_path}")
            
            if out is not None:
                out.write(display_frame)
                cv2.circle(display_frame, (750, 50), 10, (0, 0, 255), -1)
                if current_conf > max_conf:
                    max_conf = current_conf
        else:
            if is_recording:
                if out is not None:
                    out.release()
                    out = None
                is_recording = False
                
                # Log Event
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                new_event = {
                    "event_type": event_label,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": round(duration, 2),
                    "max_confidence": float(round(max_conf, 4)),
                    "video_file": os.path.basename(save_path)
                }
                
                history = []
                if os.path.exists(LOG_FILE):
                    try:
                        with open(LOG_FILE, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                            if not isinstance(history, list): history = []
                    except: history = []
                
                history.append(new_event)
                with open(LOG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
                print(f"[WEB] Event logged: {save_path}")

        frame_count += 1
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        except:
            return jsonify([])
    return jsonify([])

@app.route('/results/<path:filename>')
def serve_video(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
