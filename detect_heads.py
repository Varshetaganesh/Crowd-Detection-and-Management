import cv2
from ultralytics import YOLO

# Load model globally for efficiency
MODEL_PATH = "best.pt"  # path to your head detection model
model = YOLO(MODEL_PATH)

# Optional: you can add a tracker here if you want consistent IDs across frames
# For now, we leave it without tracking
NEXT_ID = 0  # global counter if you want unique IDs per head


def detect_heads(frame, head_class_index=0, conf=0.35, imgsz=640):
    """
    Detect heads in a single frame.
    Returns a list of dicts for Person B to use:
    [{'box':[x1,y1,x2,y2],'confidence':0.92,'id':0}, ...]
    """
    global NEXT_ID
    detections = []

    # Run detection on the frame
    results = model(frame, conf=conf, imgsz=imgsz)

    if results[0].boxes is not None:
        for box, cls, score in zip(results[0].boxes.xyxy,
                                    results[0].boxes.cls,
                                    results[0].boxes.conf):
            if int(cls) == head_class_index:  # Only heads
                x1, y1, x2, y2 = map(float, box.cpu().numpy())
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": float(score.cpu().numpy()),
                    "id": NEXT_ID
                }
                NEXT_ID += 1  # increment ID for each detected head
                detections.append(detection)

    return detections


# Example usage per frame
if __name__ == "__main__":
    VIDEO = "videos/crowd10.mp4"
    cap = cv2.VideoCapture(VIDEO)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        head_detections = detect_heads(frame)
        print(head_detections)  # This is the JSON-like output for Person B

        # Optional: visualize bounding boxes
        for det in head_detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Head Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
