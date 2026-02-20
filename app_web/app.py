import sys
import os
import cv2
import time
import json
import datetime
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from collections import deque
from werkzeug.utils import secure_filename

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.detector import ViolenceDetector

app = Flask(__name__)

# Configuration
MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
RESULTS_DIR = r"e:\Violence_Detection_System\results"
LOGS_DIR = r"e:\Violence_Detection_System\logs"
LOG_FILE = os.path.join(LOGS_DIR, "violence_history.json")
UPLOADS_DIR = r"e:\Violence_Detection_System\uploads"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global Variables
camera = None
detector = None

def get_detector():
    global detector
    if detector is None:
        print("Loading Model...")
        detector = ViolenceDetector(model_path=MODEL_PATH)
    return detector

import subprocess
import imageio_ffmpeg

def convert_to_h264(src_path):
    """Convert mp4v video to H.264 for browser playback using ffmpeg."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_path = src_path + ".tmp.mp4"
        subprocess.run([
            ffmpeg_exe, "-y", "-i", src_path,
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-movflags", "+faststart",
            tmp_path
        ], capture_output=True, timeout=60)
        os.replace(tmp_path, src_path)
        print(f"[CONVERT] {os.path.basename(src_path)} -> H.264 OK")
    except Exception as e:
        print(f"[CONVERT] Failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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
                 # Show the actual predicted action (di, dung, ngoi...)
                 current_label = pred_label
                 current_conf = pred_conf

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
                save_path = os.path.join(RESULTS_DIR, f"violence_{timestamp}.mp4")
                # Use mp4v codec (available on all OpenCV installations)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                # Convert to H.264 for browser playback
                convert_to_h264(save_path)

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

@app.route('/api/logs', methods=['GET', 'POST'])
def api_logs():
    # ── GET: return all events ──────────────────────────────────────────────
    if request.method == 'GET':
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify(data)
            except Exception:
                return jsonify([])
        return jsonify([])

    # ── POST: accept a new event from any client (app_local, app_web, etc.) ─
    event = request.get_json(silent=True)
    if not event or not isinstance(event, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    required = {'event_type', 'start_time', 'video_file'}
    if not required.issubset(event.keys()):
        return jsonify({"error": f"Missing fields: {required - event.keys()}"}), 400

    _append_log(event)
    return jsonify({"status": "ok", "logged": event}), 201

@app.route('/results/<path:filename>')
def serve_video(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)

# ── Video Upload & Analysis ──────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _append_log(event: dict):
    """Append an event dict to the central JSON log file (thread-safe enough for Flask dev)."""
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

def analyze_video_file(video_path: str, source_filename: str) -> dict:
    """
    Analyse a video file for violence events.
    Returns a summary dict with list of detected events.
    """
    det = get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video", "events": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_size = 16
    step = max(1, clip_size // 2)   # 50 % sliding window

    frames = []
    frame_idx = 0
    events = []
    in_event = False
    event_start_frame = 0
    max_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        frame_idx += 1

        if len(frames) == clip_size:
            label, conf = det.predict(frames)
            is_violence = label in ('danh_nhau', 'nga') and conf >= 0.55

            if is_violence and not in_event:
                in_event = True
                event_start_frame = frame_idx - clip_size
                max_conf = conf
                event_label = label
            elif is_violence and in_event:
                if conf > max_conf:
                    max_conf = conf
                    event_label = label
            elif not is_violence and in_event:
                in_event = False
                start_sec = round(event_start_frame / fps, 2)
                end_sec   = round(frame_idx / fps, 2)
                new_event = {
                    "source":      "upload",
                    "video_file":  source_filename,
                    "event_type":  event_label,
                    "start_time":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "start_sec":   start_sec,
                    "end_sec":     end_sec,
                    "duration_seconds": round(end_sec - start_sec, 2),
                    "max_confidence":   float(round(max_conf, 4)),
                }
                events.append(new_event)
                _append_log(new_event)

            frames = frames[step:]   # slide window

    # If video ended while still in an event
    if in_event:
        start_sec = round(event_start_frame / fps, 2)
        end_sec   = round(frame_idx / fps, 2)
        new_event = {
            "source":      "upload",
            "video_file":  source_filename,
            "event_type":  event_label,
            "start_time":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_sec":   start_sec,
            "end_sec":     end_sec,
            "duration_seconds": round(end_sec - start_sec, 2),
            "max_confidence":   float(round(max_conf, 4)),
        }
        events.append(new_event)
        _append_log(new_event)

    cap.release()
    return {"events": events, "total_frames": total_frames, "fps": fps}

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOADS_DIR, filename)
    file.save(save_path)

    result = analyze_video_file(save_path, filename)
    result["filename"] = filename
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
