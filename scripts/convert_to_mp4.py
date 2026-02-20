import os
import cv2

RESULTS_DIR = r"e:\Violence_Detection_System\results"

avi_files = [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith('.avi')]

if not avi_files:
    print("No AVI files found.")
else:
    print(f"Found {len(avi_files)} AVI file(s) to convert.")
    for filename in avi_files:
        src = os.path.join(RESULTS_DIR, filename)
        dst = os.path.join(RESULTS_DIR, filename.replace('.avi', '.mp4'))

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[SKIP] Cannot open: {filename}")
            continue

        fps    = cap.get(cv2.CAP_PROP_FPS) or 20
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(dst, fourcc, fps, (width, height))

        print(f"[CONVERTING] {filename} -> {os.path.basename(dst)} ({total} frames)")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        if frame_idx > 0:
            os.remove(src)  # Xóa file AVI gốc sau khi convert xong
            print(f"[DONE] {filename} -> {os.path.basename(dst)} ({frame_idx} frames written, original deleted)")
        else:
            os.remove(dst)
            print(f"[FAILED] {filename} - 0 frames written, skipped.")

print("\nAll done!")
