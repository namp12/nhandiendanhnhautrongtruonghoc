import sys
import os
import cv2
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.detector import ViolenceDetector

def scan_and_save(input_path, output_dir, model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = ViolenceDetector(model_path=model_path)
    
    # List of videos to process
    videos = []
    if os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    videos.append(os.path.join(root, file))
    else:
        videos.append(input_path)
        
    print(f"Scanning {len(videos)} videos...")
    
    for video_path in videos:
        print(f"Analyzing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        is_violence = False
        
        # Simple Logic: Check first few clips or moving window?
        # For simplicity, check first 16-32 frames (if short video) or sample periodically
        
        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= 16:
                break # Just check the beginning for now (Improve later for full scan)
        
        cap.release()
        
        if len(frames) >= 16:
            label, conf = detector.predict(frames[:16])
            print(f"  -> Result: {label} ({conf:.2f})")
            
            if label == 'danh_nhau' and conf > 0.7:
                print(f"  [!] Violence Detected! Saving to {output_dir}")
                # Save the detected video
                filename = os.path.basename(video_path)
                save_path = os.path.join(output_dir, f"VIOLENCE_{filename}")
                shutil.copy2(video_path, save_path)
        else:
            print("  -> Video too short, skipping.")

def main():
    INPUT_PATH = r"e:\Violence_Detection_System\data\test_videos" # Change this to user's input location
    OUTPUT_DIR = r"e:\Violence_Detection_System\results"
    MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
    
    # Check args
    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
        
    scan_and_save(INPUT_PATH, OUTPUT_DIR, MODEL_PATH)

if __name__ == "__main__":
    main()
