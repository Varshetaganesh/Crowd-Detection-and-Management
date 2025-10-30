import cv2
import tracker_utils
from ultralytics import YOLO
import numpy as np # Needed for calculation
# --- YOUR IMPORTS ---
from prediction_logic import display_global_prediction
from tracker_utils import CentroidTracker 
# --------------------

# === CONFIG (Friend's settings retained) ===
# NOTE: Check this path!
VIDEO_PATH = r"C:\Users\Dhanya\Crowd-Detection-and-Management\videos\vedio4.mp4" 
MODEL_PATH = "best.pt"
CONF_THRESH = 0.35
IMG_SIZE = 640

# --- GLOBAL TRACKER INITIALIZATION ---
# NOTE: This global variable is needed for the tracker utility to work correctly.
RESIZED_W, RESIZED_H = 640, 480 
# Declare RESIZED_W globally so tracker_utils can use it for normalization
tracker_utils.RESIZED_W = RESIZED_W 

tracker = CentroidTracker(maxDisappeared=15) # Initialize the tracker

# === LOAD MODEL (Necessary to keep detection loop working) ===
model = YOLO(MODEL_PATH)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot read video. Check VIDEO_PATH.")

frame_index = 0

print("[INFO] Starting predictive analysis with REAL-TIME VELOCITY (Press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    resized_frame = cv2.resize(frame, (RESIZED_W, RESIZED_H))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # --- Run Detection (YOLO) ---
    results = model(frame_rgb, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
    
    # 1. Prepare Detections for the Tracker
    rects = []
    
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            rects.append([x1, y1, x2, y2])
            
    # ---------------------------------------------
    
    # --- CORE TRACKING & VELOCITY CALCULATION (NEW) ---
    
    # 2. Update the Tracker: Passes current detections and returns tracked objects
    tracked_output = tracker.update(rects)
    
    head_count = len(tracked_output)
    total_velocities = []

    # 3. Process Tracked Objects and Calculate Total Velocity
    for obj in tracked_output:
        x1, y1, x2, y2 = obj['bbox']
        obj_id = obj['id']
        obj_velocity = obj['velocity'] # This is the real-time calculated speed
        
        total_velocities.append(obj_velocity)
        
        # Draw Bounding Boxes and ID (Update Friend's Drawing)
        color = (0, 255, 0) # Green for active track
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(resized_frame, f"ID:{obj_id} | V:{obj_velocity:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 4. Calculate Global Metric
    if total_velocities:
        avg_velocity = np.mean(total_velocities)
    else:
        avg_velocity = 0.0

    # --- RUN PREDICTIVE LOGIC ---
    
    # Call your function with the REAL calculated metrics
    resized_frame = display_global_prediction(
        resized_frame,          
        head_count,             
        avg_velocity      
    )
    # ----------------------------

    # Display frame
    cv2.imshow("Crowd Anomaly Prediction (Movement-Based Real-Time)", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print(f"Prediction complete. Total frames processed: {frame_index}")