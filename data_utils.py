import pandas as pd
import numpy as np

# --- CONFIGURATION (Confirmed Settings) ---
SEQUENCE_LENGTH = 10 
FEATURE_COLUMNS = [
    'x_center', 'y_center', 'width', 'height', 
    'velocity', 'acceleration', 'direction', 
    'crowd_density', 'zone_id', 'zone_count'
]
INPUT_SIZE = len(FEATURE_COLUMNS) # 10
NUM_CLASSES = 3 # For Normal, Fight, Stampede
# ------------------------------------------

def create_sequences(df, sequence_length=SEQUENCE_LENGTH):
    """Converts a DataFrame of tracked movements into time sequences."""
    sequences = []
    # Note: For supervised learning, you'd load a separate 'label' column here as well.
    
    for person_id, group in df.groupby('person_id'):
        data = group[FEATURE_COLUMNS].values.astype(np.float32) 
        if len(data) >= sequence_length:
            for i in range(len(data) - sequence_length + 1):
                sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def load_data(file_path):
    """Loads CSV, normalizes, and prepares sequential data."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file '{file_path}' was not found.")
        return np.array([])
    
    # Clean and Normalize
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-8)

    return create_sequences(df)