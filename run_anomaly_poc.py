import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from autoencoder_model import LSTMAutoencoder, INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS
from data_utils import load_data, SEQUENCE_LENGTH, FEATURE_COLUMNS
import os

# --- 1. CONFIGURATION ---
# NOTE: Check your path! Assuming 'tracking_features.csv' is in the root directory.
DATA_PATH = 'tracking_features.csv' 
MODEL_PATH = 'poc_autoencoder.pt'
BATCH_SIZE = 32
NUM_EPOCHS = 20 # Low epochs for a PoC. Increase this later for better results.
LEARNING_RATE = 0.001

# Set device to GPU if available, otherwise CPU (PyTorch standard practice)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(DATA_PATH):
    print(f"FATAL ERROR: Data file not found at {DATA_PATH}.")
    print("Please ensure you have run your friend's tracking script to generate the CSV.")
    exit()

# --- 2. DATA LOADING ---
print(f"Loading and preparing data from {DATA_PATH}...")
sequences = load_data(DATA_PATH)

if sequences.size == 0:
    print("FATAL ERROR: No valid sequences were generated. Check your CSV column names.")
    exit()

# Convert NumPy array to PyTorch Tensor (float32 is required for model)
sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

# Create a DataLoader for batching and efficient training
dataset = TensorDataset(sequences_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded successfully. Total sequences: {len(sequences)}. Feature set: {FEATURE_COLUMNS}")


# --- 3. MODEL, LOSS, and OPTIMIZER ---
model = LSTMAutoencoder(INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS).to(device)
criterion = nn.MSELoss(reduction='mean') # Mean Squared Error Loss: Measures reconstruction quality
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 4. TRAINING LOOP ---
print(f"Starting PoC Training on {device} for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0
    
    for batch_sequences in dataloader:
        # batch_sequences is a tuple from TensorDataset; we only need the first element
        input_sequences = batch_sequences[0] 

        # Forward pass: model(input) -> reconstructed output
        reconstructed_sequences = model(input_sequences)
        
        # Calculate loss (difference between original and reconstructed)
        loss = criterion(reconstructed_sequences, input_sequences)
        
        # Backward and optimize
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()      # Compute gradient of the loss
        optimizer.step()     # Update model parameters
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {avg_loss:.6f}')

# --- 5. SAVE MODEL ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ PoC Autoencoder trained and saved successfully to {MODEL_PATH}")
# --- 6. ANOMALY SCORING FUNCTION ---
def get_anomaly_score(model, sequence, device):
    """Calculates the reconstruction error (MSE Loss) for a single sequence."""
    model.eval() # Set model to evaluation mode
    
    # Ensure the sequence is a PyTorch tensor with the correct shape: [1, Sequence_Length, Features]
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(input_tensor)
        
        # Calculate MSE loss (Reconstruction Error)
        mse_loss = nn.MSELoss()(reconstructed, input_tensor)
        
    return mse_loss.item()

# --- 7. POC TESTING ---
# This block should be placed after the model saving section in run_anomaly_poc.py

# Re-use the data sequences loaded earlier
if sequences.size > 0:
    # Use the first normal sequence as a test case
    normal_sequence = sequences[0] 
    
    # Create a simple fake anomaly by making the coordinates jump (e.g., in the velocity feature)
    anomaly_sequence = normal_sequence.copy()
    
    # We are manually increasing the velocity feature (index 4) for a few steps to simulate sudden, abnormal movement
    # Index 4 is 'velocity' based on your feature list. 
    # This simulates a person suddenly moving much faster than the normal data.
    anomaly_sequence[5:8, 4] = anomaly_sequence[5:8, 4] * 1000 
    
    # Load the trained model for testing
    loaded_model = LSTMAutoencoder(INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_PATH))
    
    print("\n--- PoC Anomaly Test ---")
    
    # Calculate scores
    normal_score = get_anomaly_score(loaded_model, normal_sequence, device)
    anomaly_score = get_anomaly_score(loaded_model, anomaly_sequence, device)
    
    print(f"Score for NORMAL Movement: {normal_score:.4f}")
    print(f"Score for FAKE ANOMALY Movement: {anomaly_score:.4f}")
    
    if anomaly_score > normal_score * 1.5: # Use a threshold (1.5x higher) for clear demonstration
        print("🎉 SUCCESS: Anomaly score is significantly higher than normal score. PoC is working!")
    else:
        print("⚠️ WARNING: Scores are too close. The model needs more training, but the logic is correct.")