import cv2
from PIL import Image
import numpy as np
from detect_heads import detect_heads
import json

VIDEO_PATH = "videos/crowd10.mp4"
OUTPUT_JSON = "head_detections.json"

cap = cv2.VideoCapture(VIDEO_PATH)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

all_frames_data = []
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break

    # --- Preprocessing for display/analysis ---
    resized_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Background subtraction (for visualization only)
    foreground_mask = fgbg.apply(resized_frame)

    # Gaussian blur (for visualization only)
    blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)

    # Skip frames for performance
    if frame_index % 5 != 0:
        frame_index += 1
        continue

    # --- Detection (feed original resized RGB frame to YOLO) ---
    detections = detect_heads(resized_frame)  # Pass BGR or RGB as your detect_heads expects

    # Draw boxes + IDs for visualization
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"ID {det['id']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show original frame with boxes
    cv2.imshow("Heads", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted")
        break

    # Store frame JSON
    all_frames_data.append({
        "frame": frame_index,
        "heads": detections
    })
    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Save all detections
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_frames_data, f, indent=2)

print(f"All frame detections saved to {OUTPUT_JSON}")
