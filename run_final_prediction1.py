import cv2
from ultralytics import YOLO
# --- YOUR IMPORTS ---
from prediction_logic import display_global_prediction 
# --------------------

# === CONFIG (Friend's settings retained) ===
# NOTE: Check this path!
VIDEO_PATH = r"C:\Users\Dhanya\Crowd-Detection-and-Management\videos\vedio5.mp4" 
MODEL_PATH = "best.pt"
CONF_THRESH = 0.35
IMG_SIZE = 640

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot read video. Check VIDEO_PATH.")

RESIZED_W, RESIZED_H = 640, 480
frame_index = 0

print("[INFO] Starting predictive analysis (Press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    resized_frame = cv2.resize(frame, (RESIZED_W, RESIZED_H))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # --- Run Detection (YOLO) ---
    results = model(frame_rgb, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
    
    head_count = 0 
    
    # Draw Detections and GET HEAD COUNT
    if results and results[0].boxes is not None:
        head_count = len(results[0].boxes.xyxy)
        for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{float(conf):.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    # --- CORE PREDICTIVE LOGIC & VISUALIZATION ---
    
    # 1. Determine SIMULATED VELOCITY based on time for the demo:
    if frame_index < 150:
        # NORMAL: Low speed
        simulated_head_count = head_count # Use actual, low count
        simulated_velocity = 0.10 # Guaranteed Normal speed
    elif frame_index < 400:
        # POSSIBLE STAMPEDE: Artificially high count + high speed
        # This block will trigger the STAMPEDE alert cleanly.
        simulated_head_count = 70 
        simulated_velocity = 0.85 
    else:
        # Return to NORMAL after simulation ends
        simulated_head_count = head_count 
        simulated_velocity = 0.10 
        
    
    # 2. Call your function with the key feature metrics
    resized_frame = display_global_prediction(
        resized_frame,          
        simulated_head_count,   
        simulated_velocity      
    )
    
    # ---------------------------------------------

    # Display frame
    cv2.imshow("Crowd Anomaly Prediction (Movement-Based PoC)", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print(f"Prediction complete. Total frames processed: {frame_index}")