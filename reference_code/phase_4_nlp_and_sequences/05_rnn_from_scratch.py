# Source: AI_Learning_Cursor lines 11574-11825
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: RNN from Scratch
# Title: RECURRENT NEURAL NETWORKS FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
RECURRENT NEURAL NETWORKS FROM SCRATCH
=======================================
Understanding RNNs by building one manually.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. RNN THEORY
# =================================

print("=" * 60)
print("1. RNN THEORY")
print("=" * 60)

"""
RECURRENT NEURAL NETWORK (RNN)

The key idea: Hidden state carries information across time steps.

At each time step t:
    h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
    y_t = W_hy * h_t + b_y

Where:
- x_t: Input at time t
- h_t: Hidden state at time t (the "memory")
- h_{t-1}: Previous hidden state
- y_t: Output at time t
- W_hh, W_xh, W_hy: Weight matrices
- b_h, b_y: Biases

Visualization:

    x_0     x_1     x_2     x_3
     |       |       |       |
     v       v       v       v
   [RNN] → [RNN] → [RNN] → [RNN]
     |       |       |       |
     v       v       v       v
    y_0     y_1     y_2     y_3
    
The arrows between RNN cells represent the hidden state being passed.
"""

# =================================
# 2. RNN FROM SCRATCH
# =================================

print("\n" + "=" * 60)
print("2. RNN FROM SCRATCH")
print("=" * 60)

class RNNFromScratch(nn.Module):
    """
    A simple RNN implemented from scratch.
    
    This shows exactly what happens inside an RNN cell.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Linear(input_size, hidden_size)   # Input to hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)  # Hidden to hidden
        self.W_hy = nn.Linear(hidden_size, output_size)  # Hidden to output
        
        # Activation
        self.tanh = nn.Tanh()
    
    def forward(self, x, h_prev=None):
        """
        Forward pass for one time step.
        
        Args:
            x: Input at current time step (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            
        Returns:
            output: Output at current time step
            h_new: New hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # RNN computation
        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
        h_new = self.tanh(self.W_xh(x) + self.W_hh(h_prev))
        
        # Output
        output = self.W_hy(h_new)
        
        return output, h_new
    
    def forward_sequence(self, x_sequence):
        """
        Process an entire sequence.
        
        Args:
            x_sequence: (batch_size, seq_length, input_size)
            
        Returns:
            outputs: (batch_size, seq_length, output_size)
            h_final: Final hidden state
        """
        batch_size, seq_length, _ = x_sequence.shape
        
        h = None
        outputs = []
        
        for t in range(seq_length):
            x_t = x_sequence[:, t, :]
            output, h = self.forward(x_t, h)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h


# Test RNN
input_size = 10
hidden_size = 20
output_size = 5

rnn = RNNFromScratch(input_size, hidden_size, output_size).to(device)

# Create sample sequence
batch_size = 3
seq_length = 7
x = torch.randn(batch_size, seq_length, input_size, device=device)

# Forward pass
outputs, h_final = rnn.forward_sequence(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Final hidden state shape: {h_final.shape}")

# =================================
# 3. PyTorch RNN
# =================================

print("\n" + "=" * 60)
print("3. PyTorch RNN")
print("=" * 60)

# PyTorch's built-in RNN (much faster, same logic)
pytorch_rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True,  # (batch, seq, features) format
    nonlinearity='tanh'
).to(device)

# Forward pass
outputs_pt, h_final_pt = pytorch_rnn(x)

print(f"PyTorch RNN output shape: {outputs_pt.shape}")
print(f"PyTorch RNN hidden shape: {h_final_pt.shape}")

# =================================
# 4. UNDERSTANDING HIDDEN STATE
# =================================

print("\n" + "=" * 60)
print("4. UNDERSTANDING HIDDEN STATE")
print("=" * 60)

"""
The hidden state is the "memory" of the RNN.

Let's visualize how it evolves over a sequence.
"""

# Simple example: Count 1s in a binary sequence
class CountingRNN(nn.Module):
    """RNN that learns to count 1s in a binary sequence."""
    
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_length, 1) - binary values
        output, h = self.rnn(x)
        # Use final hidden state to predict count
        count = self.fc(h.squeeze(0))
        return count


# Generate training data
def generate_counting_data(n_samples, seq_length):
    X = torch.randint(0, 2, (n_samples, seq_length, 1)).float()
    y = X.sum(dim=1)  # Count of 1s
    return X, y


# Train
seq_length = 10
X_train, y_train = generate_counting_data(1000, seq_length)
X_test, y_test = generate_counting_data(200, seq_length)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

model = CountingRNN(hidden_size=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training RNN to count 1s in binary sequences...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

# Test
model.eval()
print("\nTest predictions:")
for i in range(5):
    seq = X_test[i].squeeze().cpu().numpy().astype(int)
    true_count = y_test[i].item()
    pred_count = model(X_test[i:i+1]).item()
    print(f"  Sequence: {seq.tolist()} -> True: {true_count:.0f}, Predicted: {pred_count:.1f}")
