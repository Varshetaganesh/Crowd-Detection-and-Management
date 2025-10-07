import cv2
import numpy as np
from detect_heads import detect_heads
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Load video from Colab file system
cap = cv2.VideoCapture("/content/Many_people.mp4")

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read frame — check video file")
        break

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Optional: skip frames for performance
    if frame_id % 5 != 0:
        frame_id += 1
        continue

    # Run head detection
    detections = detect_heads(frame)

    # Log crowd density for current frame
    crowd_score = len(detections)
    with open("density_log.txt", "a") as log_file:
        log_file.write(f"Frame {frame_id}: {crowd_score} heads detected\n")

    # Step 1: Create blank heatmap canvas
    heatmap = np.zeros((480, 640), dtype=np.float32)

    # Step 2: Plot head centers with additive intensity
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cv2.circle(heatmap, (cx, cy), radius=20, color=1, thickness=-1)

    # Step 3: Smooth to simulate crowd spread
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), sigmaX=10)

    # Step 4: Normalize to 0–255 scale
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Step 5: Apply semantic color mapping
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Step 6: Resize heatmap to match frame
    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

    # Step 7: Blend heatmap with original frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Step 8: Add crowd density alert
    crowd_score = len(detections)
    if crowd_score > 80:
        cv2.putText(overlay, "High Crowd Density", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif crowd_score > 50:
        cv2.putText(overlay, "Moderate Crowd", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(overlay, "Safe Zone", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display result
    cv2_imshow(overlay)

    # Optional: save annotated frame
    cv2.imwrite(f"annotated_frame_{frame_id}.jpg", overlay)

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
