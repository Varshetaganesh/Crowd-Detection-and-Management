import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "best.pt"  # path to your trained YOLO head detection model
model = YOLO(MODEL_PATH)

NEXT_ID = 0  # global counter for unique IDs


def detect_heads(frame, head_class_index=0, conf=0.25, imgsz=640, min_heads_threshold=0):
    """
    Detect heads in a frame.
    Returns: [{'box':[x1,y1,x2,y2],'confidence':0.92,'id':0,'center':[cx,cy]}, ...]
    Skips frame only if fewer than 'min_heads_threshold' detections.
    """
    global NEXT_ID
    detections = []

    results = model(frame, conf=conf, imgsz=imgsz)

    if results and results[0].boxes is not None:
        for box, cls, score in zip(results[0].boxes.xyxy,
                                   results[0].boxes.cls,
                                   results[0].boxes.conf):
            # If you trained YOLO with a single "head" class, index = 0
            if head_class_index is None or int(cls) == head_class_index:
                x1, y1, x2, y2 = map(float, box.cpu().numpy())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(score.cpu().numpy()),
                    "id": NEXT_ID,
                    "center": [cx, cy]
                })
                NEXT_ID += 1

    # Allow detections even if few (for visualization)
    if len(detections) < min_heads_threshold:
        return []

    return detections
