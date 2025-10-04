%%writefile semantic_heatmap.py
import cv2
import numpy as np
from detect_heads import detect_heads
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

model = YOLO("best.pt")
cap = cv2.VideoCapture("/content/crowd10.mp4")

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read frame — check video file")
        break

    frame = cv2.resize(frame, (640, 480))

    if frame_id % 5 != 0:
        frame_id += 1
        continue

    detections = detect_heads(frame)

    heatmap = np.zeros((480, 640), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cv2.circle(heatmap, (cx, cy), radius=20, color=1, thickness=-1)

    heatmap = cv2.GaussianBlur(heatmap, (31, 31), sigmaX=10)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    cv2_imshow(overlay)
    cv2.imwrite(f"annotated_frame_{frame_id}.jpg", overlay)

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
