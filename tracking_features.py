# ======================================
# tracking_features_corrected.py
# Computes enhanced movement & crowd features for all people per frame
# ======================================

import pandas as pd
import numpy as np
import math

# ============ CONFIG ============
TRACKING_CSV = "tracking_output.csv"       # Input CSV from YOLO/DeepSORT
OUTPUT_CSV = "tracking_features.csv"      # Output CSV with enhanced features
FPS = 30                                   # Adjust according to your video
FRAME_WIDTH = 1920                         # Video frame width
FRAME_HEIGHT = 1080                        # Video frame height
GRID_ROWS = 3                              # Number of rows for zone grid
GRID_COLS = 3                              # Number of columns for zone grid

# ============ LOAD TRACKING DATA ============
df = pd.read_csv(TRACKING_CSV)

# Ensure proper sorting
df = df.sort_values(['person_id', 'frame_id']).reset_index(drop=True)

# ============ PREPARE NEW COLUMNS ============
df['velocity'] = 0.0
df['acceleration'] = 0.0
df['direction'] = 0.0

# ============ FUNCTION TO GET ZONE ============
def get_zone(x, y):
    col = min(int(x / (FRAME_WIDTH / GRID_COLS)), GRID_COLS - 1)
    row = min(int(y / (FRAME_HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
    return row * GRID_COLS + col  # zone ID 0 to 8

# ============ COMPUTE VELOCITY, ACCELERATION, DIRECTION PER PERSON ============
def compute_person_features(group):
    group = group.sort_values('frame_id').copy()
    prev_x, prev_y, prev_v = None, None, None
    velocities, accelerations, directions = [], [], []
    for _, row in group.iterrows():
        if prev_x is None:
            velocity = 0.0
            acceleration = 0.0
            direction = 0.0
        else:
            dx = row['x_center'] - prev_x
            dy = row['y_center'] - prev_y
            velocity = np.sqrt(dx**2 + dy**2) * FPS
            acceleration = (velocity - prev_v) * FPS if prev_v is not None else 0.0
            direction = math.atan2(dy, dx)
        velocities.append(velocity)
        accelerations.append(acceleration)
        directions.append(direction)
        prev_x, prev_y, prev_v = row['x_center'], row['y_center'], velocity
    group['velocity'] = velocities
    group['acceleration'] = accelerations
    group['direction'] = directions
    return group

# Apply to all person_ids
df = df.groupby('person_id', group_keys=False).apply(compute_person_features)

# ============ CROWD DENSITY PER FRAME ============
crowd_density = df.groupby('frame_id')['person_id'].nunique().reset_index()
crowd_density.rename(columns={'person_id':'crowd_density'}, inplace=True)
df = df.merge(crowd_density, on='frame_id', how='left')

# ============ ZONE OCCUPANCY ============
df['zone_id'] = df.apply(lambda r: get_zone(r['x_center'], r['y_center']), axis=1)

# Count number of people per zone per frame
zone_counts = df.groupby(['frame_id','zone_id']).size().reset_index(name='zone_count')
df = df.merge(zone_counts, on=['frame_id','zone_id'], how='left')

# ============ SAVE OUTPUT ============
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Enhanced tracking features for all people saved to {OUTPUT_CSV}")
