import torch
import torch.nn as nn
from data_utils import INPUT_SIZE, NUM_CLASSES 

# --- CONFIGURATION ---
HIDDEN_DIM = 64 
NUM_LAYERS = 2

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        
        # LSTM layers to learn the sequence pattern
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Final fully connected layer to map the final LSTM hidden state to the 3 classes
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, Sequence_Length, Features]
        
        # Run through LSTM. Only the final hidden state (h_n) is important for classification.
        # h_n has shape [Num_Layers, Batch, Hidden_Dim]
        _, (h_n, _) = self.lstm(x)
        
        # Take the hidden state of the last layer
        last_hidden_state = h_n[-1] # [Batch, Hidden_Dim]
        
        # Pass through the final linear layer for classification scores (logits)
        output = self.fc(last_hidden_state) # [Batch, Num_Classes]
        
        return output