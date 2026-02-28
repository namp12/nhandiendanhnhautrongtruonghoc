import sys
import os
import cv2
import time
import datetime
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from collections import deque
from werkzeug.utils import secure_filename
import subprocess
import imageio_ffmpeg
import mysql.connector
import threading
import shutil

# Mẫu cấu hình kết nối Database
DB_CONFIG = {
    'host': 'localhost',      
    'user': 'root',           
    'password': 'phuongnam@3333', 
    'database': 'violence_db' 
}

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.detector import ViolenceDetector

app = Flask(__name__)

# Configuration
MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
RESULTS_DIR = r"e:\Violence_Detection_System\results"
LOGS_DIR = r"e:\Violence_Detection_System\logs"
UPLOADS_DIR = r"e:\Violence_Detection_System\uploads"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global Variables
camera = None
detector_instance = None
last_prediction = {"label": "Normal", "confidence": 0, "inference_time": 0, "probs": {}, "status": "Normal"}

def log_event_to_db(event_type, conf, start, end, filename, path):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """INSERT INTO EVENT_LOGS 
                 (event_type, confidence, start_time, end_time, video_filename, video_path) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        values = (event_type, float(conf), start, end, filename, path)
        cursor.execute(sql, values)
        conn.commit()
    except Exception as err:
        print(f"[DB ERROR] Không lưu được DB (Có thể chưa bật MySQL hoặc chưa tạo DB): {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_logs_from_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT event_type, start_time, end_time, video_filename as video_file FROM EVENT_LOGS ORDER BY id ASC")
        logs = cursor.fetchall()
        for log in logs:
            if isinstance(log['start_time'], datetime.datetime):
                log['start_time'] = log['start_time'].strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(log['end_time'], datetime.datetime):
                log['end_time'] = log['end_time'].strftime("%Y-%m-%d %H:%M:%S")
        return logs
    except Exception as err:
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_detector():
    global detector_instance
    if detector_instance is None:
        print("Loading Model...")
        detector_instance = ViolenceDetector(model_path=MODEL_PATH)
    return detector_instance

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_offline(video_path):
    detector = get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Cannot open video file"}

    frames_buffer = deque(maxlen=16)
    prediction_history = deque(maxlen=5)
    
    start_time = datetime.datetime.now()
    max_conf = 0.0
    detected_danger = False
    danger_label = "normal"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame_rgb)
        
        if len(frames_buffer) == 16:
            label, conf, probs_dict = detector.predict(frames_buffer)
            prediction_history.append(probs_dict)
            
            if len(prediction_history) == 5:
                avg_probs = {}
                for c in detector.classes:
                    avg_probs[c] = sum(p_dict.get(c, 0.0) for p_dict in prediction_history) / 5.0
                
                best_label = max(avg_probs, key=avg_probs.get)
                best_conf = avg_probs[best_label]
                
                if best_label in ['danh_nhau', 'nga'] and best_conf > 0.85:
                    detected_danger = True
                    if best_conf > max_conf:
                        max_conf = best_conf
                        danger_label = best_label

    cap.release()
    end_time = datetime.datetime.now()
    
    result_data = {
        "status": "success",
        "detected_danger": detected_danger,
        "danger_label": danger_label,
        "max_conf": round(max_conf * 100, 1) if detected_danger else 100.0
    }
    if detected_danger:
        filename = os.path.basename(video_path)
        dest_path = os.path.join(RESULTS_DIR, f"uploaded_danger_{filename}")
        shutil.copy(video_path, dest_path)
        convert_to_h264(dest_path)
        
        video_url = f'/results/{os.path.basename(dest_path)}'
        
        log_event_to_db(
            danger_label,
            max_conf,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(dest_path),
            video_url
        )
        result_data["video_url"] = video_url
        
    return result_data

def convert_to_h264(input_path):
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_path = input_path + ".tmp.mp4"
        subprocess.run([
            ffmpeg_exe, "-y", "-i", input_path,
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-movflags", "+faststart",
            tmp_path
        ], capture_output=True, timeout=60)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.replace(tmp_path, input_path)
        else:
            print("[CONVERT] FFmpeg failed to produce a valid file")
    except Exception as e:
        print(f"[CONVERT] Failed: {e}")

def calculate_motion(frames):
    if not frames or len(frames) < 2: return 0.0
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    diff_sum = 0.0
    for i in range(len(gray_frames) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
        diff_sum += np.mean(diff)
    return diff_sum / (len(frames) - 1)

def generate_frames():
    global camera, last_prediction
    if camera is None or not camera.isOpened():
        for idx in [0, 1, 2]:
            cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cam.isOpened():
                # Read a few frames to skip buffer and ensure it's not a broken virtual cam
                ret, _ = cam.read()
                if ret:
                    camera = cam
                    break
            cam.release()
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not found", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            _, buf = cv2.imencode('.jpg', blank)
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            return

    detector = get_detector()
    frames_buffer = []
    prediction_history = deque(maxlen=5)
    frame_count = 0
    current_label = "Normal"
    current_conf = 0.0
    is_recording = False
    out = None
    save_path = ""
    start_time = None
    max_conf = 0.0
    recorded_danger_label = "normal"

    while True:
        success, frame = camera.read()
        if not success:
            camera.release()
            for idx in [0, 1, 2]:
                cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cam.isOpened():
                    ret, _ = cam.read()
                    if ret:
                        camera = cam
                        break
                cam.release()
            else: 
                break
            continue

        display_frame = cv2.resize(frame, (800, 600))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % 3 == 0:
            frames_buffer.append(rgb_frame)
        if len(frames_buffer) > 16:
            frames_buffer.pop(0)

        if len(frames_buffer) == 16 and frame_count % 5 == 0:
            start_inf = time.time()
            motion_score = calculate_motion(frames_buffer)
            
            if motion_score < 5.0:
                pred_label, pred_conf = "normal", 1.0
                probs_dict = {c: 0.0 for c in detector.classes}
                probs_dict["normal"] = 1.0
            else:
                label, conf, probs_dict = detector.predict(frames_buffer)
                pred_label, pred_conf = label, conf
            
            inf_time = (time.time() - start_inf) * 1000
            
            # 1. Store the full probability vector
            prediction_history.append(probs_dict)
            
            # 2. Calculate Weighted Average Probabilities
            avg_probs = {}
            for cls in detector.classes:
                avg_probs[cls] = sum(p.get(cls, 0) for p in prediction_history) / len(prediction_history)
            
            # 3. Decision Logic
            best_label = max(avg_probs, key=avg_probs.get)
            best_conf = avg_probs[best_label]
            
            # --- Anti-False Positive Rule ---
            # If the winner is 'danh_nhau' or 'nga' but confidence is low, 
            # fall back to the best non-violent class
            VIOLENCE_THRESHOLD = 0.85
            if best_label in ['danh_nhau', 'nga'] and best_conf < VIOLENCE_THRESHOLD:
                non_violent_probs = {c: v for c, v in avg_probs.items() if c not in ['danh_nhau', 'nga']}
                if non_violent_probs:
                    best_label = max(non_violent_probs, key=non_violent_probs.get)
                    best_conf = avg_probs[best_label]
                else:
                    best_label = "normal"
                    best_conf = 1.0
            
            current_label = best_label
            current_conf = best_conf

            last_prediction = {
                "label": current_label,
                "confidence": round(current_conf * 100, 1),
                "inference_time": round(inf_time, 1),
                "motion": round(motion_score, 2),
                "status": "DANGER" if current_label in ['danh_nhau', 'nga'] else "SAFE"
            }

        # Draw UI
        color = (0, 0, 255) if current_label in ['danh_nhau', 'nga'] else (0, 255, 0)
        cv2.putText(display_frame, f"Status: {current_label} ({current_conf:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if current_label in ['danh_nhau', 'nga']:
            if not is_recording:
                start_time = datetime.datetime.now()
                save_path = os.path.join(RESULTS_DIR, f"violence_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4")
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (800, 600))
                is_recording = True
                max_conf = current_conf
                recorded_danger_label = current_label
            else:
                if current_conf > max_conf:
                    max_conf = current_conf
                    recorded_danger_label = current_label
            out.write(display_frame)
        else:
            if is_recording:
                if out: out.release(); out = None
                is_recording = False
                end_time = datetime.datetime.now()
                
                # Gọi hàm lưu DB trực tiếp
                video_url = f'/results/{os.path.basename(save_path)}'
                log_event_to_db(
                    recorded_danger_label, 
                    max_conf, 
                    start_time.strftime("%Y-%m-%d %H:%M:%S"), 
                    end_time.strftime("%Y-%m-%d %H:%M:%S"), 
                    os.path.basename(save_path), 
                    video_url
                )
                        
                convert_to_h264(save_path)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        frame_count += 1

@app.route('/')
def index(): return render_template('index.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route('/upload')
def upload(): return render_template('upload.html')

@app.route('/history')
def history(): return render_template('history.html')

@app.route('/testing')
def testing(): return render_template('testing.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return render_template('upload.html', error="Không có file nộp lên")
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error="Chưa chọn file")
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOADS_DIR, f"user_upload_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}")
        file.save(save_path)
        
        # Chạy đồng bộ tiến trình phân tích để web hiển thị ngay video ra
        result = process_video_offline(save_path)
        
        if result and result.get("status") == "success":
            if result.get("detected_danger"):
                return render_template('upload.html', 
                                       success=True, 
                                       danger=True,
                                       label=result.get("danger_label"),
                                       conf=result.get("max_conf"),
                                       video_url=result.get("video_url"))
            else:
                return render_template('upload.html', 
                                       success=True, 
                                       danger=False,
                                       message="Video an toàn. Không phát hiện dấu hiệu Bạo lực hoặc Ngã.")
        else:
            return render_template('upload.html', error="Có lỗi xảy ra trong quá trình phân tích video.")

    return render_template('upload.html', error="Định dạng file không được hỗ trợ (chỉ nhận .mp4, .avi, .mov)")

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_prediction')
def api_current_prediction(): return jsonify(last_prediction)

@app.route('/api/logs')
def get_logs():
    db_logs = get_logs_from_db()
    if db_logs is not None:
        return jsonify(db_logs)
        
    print("[DB ERROR] Mất kết nối DB, không thể lấy lịch sử.")
    return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
