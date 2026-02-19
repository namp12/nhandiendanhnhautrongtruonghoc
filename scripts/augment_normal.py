import cv2
import os
import numpy as np
import random

def augment_video(input_path, output_dir, file_prefix):
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        return

    height, width, _ = frames[0].shape
    fps = 30.0

    # Augmentation functions
    def save_video(frames_list, suffix):
        out_path = os.path.join(output_dir, f"{file_prefix}_{suffix}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in frames_list:
            out.write(f)
        out.release()

    # 1. Original
    # save_video(frames, "orig") # Already exists

    # 2. Flip
    frames_flip = [cv2.flip(f, 1) for f in frames]
    save_video(frames_flip, "flip")

    # 3. Brightness Changes (0.7x, 1.3x)
    for beta in [-50, 50]: # Brightness bias
        frames_bright = [cv2.convertScaleAbs(f, alpha=1.0, beta=beta) for f in frames]
        save_video(frames_bright, f"bright_{beta}")

    # 4. Contrast Changes (0.8x, 1.2x)
    for alpha in [0.8, 1.2]:
        frames_contrast = [cv2.convertScaleAbs(f, alpha=alpha, beta=0) for f in frames]
        save_video(frames_contrast, f"contrast_{alpha}")
        
    # 5. Speed Changes (Sample frames)
    # Speed 1.5x (Skip every 3rd frame)
    frames_fast = frames[::2]
    if len(frames_fast) > 16:
        save_video(frames_fast, "speed_1.5")
    
    # Speed 0.5x (Duplicate frames)
    frames_slow = [val for val in frames for _ in (0, 1)]
    save_video(frames_slow, "speed_0.5")

    # 6. Combined Flip + Brightness
    frames_flip_bright = [cv2.convertScaleAbs(f, alpha=1.1, beta=30) for f in frames_flip]
    save_video(frames_flip_bright, f"flip_bright")

def main():
    TRAIN_DIR = r"e:\Violence_Detection_System\dataset_split\train\normal"
    VAL_DIR = r"e:\Violence_Detection_System\dataset_split\val\normal"
    
    print("Augmenting Normal class...")
    
    for d in [TRAIN_DIR, VAL_DIR]:
        if not os.path.exists(d):
            continue
            
        files = [f for f in os.listdir(d) if f.endswith('.mp4') and 'aug' not in f] # Avoid re-augmenting
        print(f"Processing {len(files)} videos in {d}...")
        
        for file in files:
            path = os.path.join(d, file)
            prefix = os.path.splitext(file)[0] + "_aug"
            augment_video(path, d, prefix)
            
    print("Done! Check file counts now.")

if __name__ == "__main__":
    main()
