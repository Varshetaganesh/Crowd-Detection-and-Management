import torch
import torch.nn as nn
from data_utils import INPUT_SIZE # Import the feature size defined in data_utils

# --- CONFIGURATION ---
# These parameters define the size and complexity of your model.
# We keep them manageable for quick PoC training.
HIDDEN_DIM = 64 # Size of the LSTM's internal memory
NUM_LAYERS = 2

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        
        # --- Encoder (LSTM) ---
        # The encoder reads the input sequence and compresses it into a final state (hidden_state)
        self.encoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True # Input/Output tensors have shape [Batch, Sequence, Features/Hidden]
        )
        
        # --- Decoder (LSTM) ---
        # The decoder takes the final state from the encoder and tries to recreate the original sequence
        self.decoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Output layer maps the decoder's hidden size back to the original feature size (4 features)
        self.output_layer = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, Features]
        
        # 1. ENCODE
        # _ is the full output sequence (we ignore it); (h_n, c_n) is the final hidden and cell state
        _, (hidden_state, cell_state) = self.encoder(x)
        
        # 2. DECODE
        # Prepare for decoding: The decoder needs an input at each step.
        # We use a simple method called 'teacher forcing' or just feeding the 
        # previous output back in. Here, we start with the first element of the input.
        
        decoder_outputs = []
        
        # The first input to the decoder is the first element of the original sequence
        decoder_input = x[:, 0, :].unsqueeze(1) # [Batch, 1, Features]
        
        # Loop through the sequence length to generate the full reconstructed sequence
        for i in range(x.size(1)): # x.size(1) is the SEQUENCE_LENGTH
            # Pass through LSTM decoder, maintaining the hidden and cell state across steps
            decoder_output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state))
            
            # Map the hidden output to the feature space (4 features)
            reconstructed_step = self.output_layer(decoder_output) # [Batch, 1, Features]
            
            decoder_outputs.append(reconstructed_step)
            
            # The next input to the decoder is the step we just reconstructed
            decoder_input = reconstructed_step
            
        # Concatenate all the single steps to form the final reconstructed sequence
        return torch.cat(decoder_outputs, dim=1) # [Batch, Sequence_Length, Features]