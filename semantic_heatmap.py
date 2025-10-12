import cv2
import numpy as np
from detect_heads import detect_heads
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best.pt"
INPUT_VIDEO = "videos/Many_people.mp4"
OUTPUT_VIDEO = "videos/semantic_heatmap_output.mp4"
FRAME_SKIP = 5  # process every Nth frame

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {INPUT_VIDEO}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

frame_id = 0

# Open density log
log_file = open("density_log.txt", "w")

print("[INFO] Processing video and generating semantic heatmap...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Optional: skip frames for speed
    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    # Detect heads
    detections = detect_heads(frame)

    # Log crowd density
    crowd_score = len(detections)
    log_file.write(f"Frame {frame_id}: {crowd_score} heads detected\n")
    print(f"Frame {frame_id}: {crowd_score} heads detected")

    # Step 1: Create blank heatmap canvas
    heatmap = np.zeros((480, 640), dtype=np.float32)

    # Step 2: Plot head centers
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cv2.circle(heatmap, (cx, cy), radius=20, color=1, thickness=-1)

    # Step 3: Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), sigmaX=10)

    # Step 4: Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Step 5: Apply color map
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Step 6: Resize to match original frame
    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

    # Step 7: Blend with original frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Step 8: Add crowd density alert
    if crowd_score > 80:
        cv2.putText(overlay, "High Crowd Density", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif crowd_score > 50:
        cv2.putText(overlay, "Moderate Crowd", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(overlay, "Safe Zone", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to output video
    overlay_resized = cv2.resize(overlay, (frame_width, frame_height))
    out_video.write(overlay_resized)

    frame_id += 1

# Release everything
cap.release()
out_video.release()
log_file.close()
print(f"[INFO] Done! Output video saved as {OUTPUT_VIDEO}")
print("[INFO] Crowd density log saved as density_log.txt")
