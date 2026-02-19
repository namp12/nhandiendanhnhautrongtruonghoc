import cv2
import os
import time

def main():
    OUTPUT_DIR = r"e:\Violence_Detection_System\dataset\train\normal"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("--- DATA COLLECTION TOOL ---")
    print(f"Saving to: {OUTPUT_DIR}")
    print("Instructions:")
    print("1. Press 'r' to start recording a 5-second clip.")
    print("2. Perform normal actions:")
    print("   - Sitting still")
    print("   - Standing up / Sitting down")
    print("   - Walking into frame")
    print("   - Waving hand / Drinking water")
    print("3. Press 'q' to quit.")
    
    clip_count = len(os.listdir(OUTPUT_DIR))
    recording = False
    start_time = 0
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        if recording:
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"RECORDING: {elapsed:.1f}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(display_frame, (750, 50), 20, (0, 0, 255), -1)
            
            frames.append(frame)
            
            if elapsed >= 5.0: # 5 seconds clip
                recording = False
                clip_name = f"normal_{clip_count}_{int(time.time())}.mp4"
                save_path = os.path.join(OUTPUT_DIR, clip_name)
                
                # Save video
                height, width, _ = frames[0].shape
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
                for f in frames:
                    out.write(f)
                out.release()
                
                print(f"Saved: {clip_name}")
                clip_count += 1
                frames = []
                cv2.putText(display_frame, "SAVED!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Data Collector', display_frame)
                cv2.waitKey(500) # Show saved message briefly
                
        else:
            cv2.putText(display_frame, "Press 'r' to Record", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Saved Clips: {clip_count}", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Data Collector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not recording:
            recording = True
            start_time = time.time()
            frames = []
            print("Recording started...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
