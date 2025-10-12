import cv2
import json
from ultralytics import YOLO

# === CONFIG ===
VIDEO_PATH = r"videos/test_video.mp4"
OUTPUT_JSON = "head_detections.json"
MODEL_PATH = "best.pt"  # your trained YOLOv8 head detection model
CONF_THRESH = 0.35
IMG_SIZE = 640
HEAD_CLASS_INDEX = 0  # None = all classes, 0 = first class

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot read video.")

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

RESIZED_W, RESIZED_H = 640, 480

frame_index = 0
all_frames_data = []

print("[INFO] Starting head detection (Press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # --- Preprocessing ---
    resized_frame = cv2.resize(frame, (RESIZED_W, RESIZED_H))
    fgmask = fgbg.apply(resized_frame)  # background subtraction
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # --- Run detection ---
    results = model(frame_rgb, conf=CONF_THRESH, imgsz=IMG_SIZE)
    detections = []

    if results and results[0].boxes is not None:
        for box, cls, score in zip(results[0].boxes.xyxy,
                                   results[0].boxes.cls,
                                   results[0].boxes.conf):
            if HEAD_CLASS_INDEX is None or int(cls) == HEAD_CLASS_INDEX:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(score.cpu().numpy())

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "center": [cx, cy]
                })

                # Draw detections on frame
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(resized_frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(resized_frame, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Store detections for JSON
    all_frames_data.append({
        "frame": frame_index,
        "heads": detections
    })

    # Display frame
    cv2.imshow("YOLO Head Detection (Preprocessed)", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_frames_data, f, indent=2)

print(f" Detection complete. JSON saved at {OUTPUT_JSON}")
