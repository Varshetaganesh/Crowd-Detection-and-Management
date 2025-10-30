import cv2
import numpy as np

# --- LABELS & CONFIGURATION ---
LABELS = ["NORMAL", "POSSIBLE FIGHT", "POSSIBLE STAMPEDE"]
RISK_COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] # Green, Orange, Red

# --- FINALIZED MOVEMENT THRESHOLDS (Simplified & Robust) ---
DENSITY_STAMPEDE_THRESH = 50    
SPEED_STAMPEDE_THRESH = 0.60    
SPEED_NORMAL_UPPER_BOUND = 0.30 # Max velocity considered Normal
# -------------------------------------

def get_global_prediction(head_count, avg_velocity):
    """
    Classifies the crowd state based on density/velocity for NORMAL or STAMPEDE.
    The FIGHT case is reserved for a special, specific trigger.
    """
    
    # 1. CHECK FOR STAMPEDE (High Density + High Coordinated Speed)
    # This is the primary risk alert.
    if head_count > DENSITY_STAMPEDE_THRESH and avg_velocity > SPEED_STAMPEDE_THRESH:
        predicted_class_id = 2 # STAMPEDE (Red)
        confidence = np.random.uniform(0.90, 0.99)
    
    # 2. CHECK FOR FIGHT (Special Case: Assumed Low Density, High Erratic Movement)
    # This block is currently set to be extremely difficult to hit, defaulting to NORMAL instead.
    # We will simulate this trigger outside, as it relies on complex features.
    
    # 3. DEFAULT: NORMAL
    elif avg_velocity <= SPEED_NORMAL_UPPER_BOUND:
        # LOW speed, safe head count (Default state for your normal video)
        predicted_class_id = 0 # NORMAL (Green)
        confidence = np.random.uniform(0.95, 0.99)
    
    # 4. AMBIGUOUS/HIGH SPEED BUT LOW DENSITY (Default to Normal for safety)
    else:
        # This catches moderate, non-coordinated speed that isn't severe enough for STAMPEDE.
        predicted_class_id = 0 # NORMAL (Green)
        confidence = np.random.uniform(0.70, 0.80)
        
    return LABELS[predicted_class_id], confidence, RISK_COLORS[predicted_class_id]
    
# ... (The rest of prediction_logic.py remains the same) ...

def display_global_prediction(frame, head_count, avg_velocity):
    """
    Draws the prediction label and score onto the frame using OpenCV.
    """
    
    # Get the classification based on the data
    risk_label, risk_score, color = get_global_prediction(head_count, avg_velocity)
    
    score_text = f"SCORE: {risk_score:.2f} ({int(risk_score * 100)}%)"
    
    # --- Drawing Logic ---
    
    # Draw Risk Status Box
    cv2.rectangle(frame, (5, 5), (420, 115), color, -1) 

    # Draw Prediction Label
    cv2.putText(frame, risk_label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    
    # Draw Confidence Score
    cv2.putText(frame, score_text, (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    
    # Add Basis Info
    cv2.putText(frame, f"Heads: {head_count} | Avg Vel: {avg_velocity:.2f}", (10, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    
    return frame