# data_utils.py (FINALIZED)

import pandas as pd
import numpy as np
import torch

# --- CONFIGURATION (Locked down based on your CSV columns) ---
SEQUENCE_LENGTH = 10 # 10 past frames for sequence analysis (fixed for PoC)
FEATURE_COLUMNS = [
    'x_center', 'y_center', 'width', 'height', 
    'velocity', 'acceleration', 'direction', 
    'crowd_density', 'zone_id', 'zone_count'
]
INPUT_SIZE = len(FEATURE_COLUMNS) # Locked to 10
# ---------------------------------------------------------------------

def create_sequences(df, sequence_length=SEQUENCE_LENGTH):
    """Converts a DataFrame of tracked movements into time sequences."""
    sequences = []
    
    # Group data by each unique person
    for person_id, group in df.groupby('person_id'):
        # Ensure only numeric data is used
        data = group[FEATURE_COLUMNS].values.astype(np.float32) 
        
        # Slide a window of size SEQUENCE_LENGTH over the data
        if len(data) >= sequence_length:
            for i in range(len(data) - sequence_length + 1):
                sequences.append(data[i:i + sequence_length])
    
    # Convert list of sequences into a single NumPy array
    return np.array(sequences)

def load_data(file_path):
    """Loads the CSV and prepares the sequential data for model input."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # We rely on your friend's script to generate this file
        print(f"FATAL ERROR: The required input file '{file_path}' was not found.")
        return np.array([])
    
    # Drop rows with NaN/Inf values, as they break training
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)

    return create_sequences(df)