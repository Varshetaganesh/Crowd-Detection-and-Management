# ========================================
# deepsort_track.py (Fixed: includes frame 0)
# Professional DeepSORT tracking for head detections
# ========================================

import cv2
import json
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = r"videos/test_video.mp4"       
DETECTIONS_JSON = "head_detections.json"    
OUTPUT_CSV = "tracking_output.csv"        
OUTPUT_VIDEO = "tracked_video.mp4"         
CONF_THRESHOLD = 0.5                        

# ==============================
# Initialize DeepSORT
# ==============================
tracker = DeepSort(max_age=30)  # full DeepSORT with appearance embeddings

# ==============================
# Load YOLO detections JSON
# ==============================
with open(DETECTIONS_JSON, "r") as f:
    video_frames = json.load(f)

# Convert list to dict for fast lookup by frame number
video_frames_dict = {f["frame"]: f for f in video_frames}

# ==============================
# Open video
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# ==============================
# Prepare output storage
# ==============================
tracking_data = []

print("[INFO] Starting professional DeepSORT tracking...")

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # --- get detections for this frame ---
    frame_data = video_frames_dict.get(frame_index)
    detections = []

    if frame_data:
        for h in frame_data["heads"]:
            if h["confidence"] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = h["box"]
                detections.append(([float(x1), float(y1), float(x2), float(y2)], float(h["confidence"])))

    # --- update DeepSORT ---
    tracks = tracker.update_tracks(detections, frame=frame if detections else None)

    # --- save all tracks (remove is_confirmed filter) ---
    if tracks:
        for t in tracks:
            x1, y1, x2, y2 = t.to_ltrb()
            track_id = t.track_id
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            tracking_data.append([frame_index, track_id, x_center, y_center, width, height])

            # --- draw bounding boxes + IDs ---
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- write frame to output video ---
    out_video.write(frame)
    frame_index += 1

# ==============================
# Save CSV
# ==============================
df = pd.DataFrame(tracking_data, columns=['frame_id', 'person_id', 'x_center', 'y_center', 'width', 'height'])
df.to_csv(OUTPUT_CSV, index=False)

# ==============================
# Release resources
# ==============================
cap.release()
out_video.release()
cv2.destroyAllWindows()

print(f"[INFO] Tracking complete! CSV saved: {OUTPUT_CSV}, Video saved: {OUTPUT_VIDEO}")
