# Source: AI_Learning_Cursor lines 12409-12745
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: LSTM from Scratch
# Title: LSTM (LONG SHORT-TERM MEMORY) FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
LSTM (LONG SHORT-TERM MEMORY) FROM SCRATCH
==========================================
Understanding the gating mechanisms that solve vanishing gradients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# =================================
# 1. LSTM THEORY
# =================================

print("=" * 60)
print("1. LSTM THEORY")
print("=" * 60)

"""
LSTM: Long Short-Term Memory

The key insight: Add a "cell state" that flows unchanged through time.
Gates control what information to add/remove from cell state.

Three gates:
1. FORGET GATE (f_t): What to forget from cell state
   f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)

2. INPUT GATE (i_t): What new information to add
   i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
   c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate values

3. OUTPUT GATE (o_t): What to output from cell state
   o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)

Cell state update:
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Hidden state update:
   h_t = o_t ⊙ tanh(c_t)

The cell state c_t allows gradients to flow unchanged (via f_t ≈ 1)
This solves the vanishing gradient problem!

Salesforce Analogy:
Think of LSTM like a Case with a running summary field:
- Forget gate: Remove outdated info ("Issue no longer relevant")
- Input gate: Add new info ("Customer reported new symptom")
- Output gate: Decide what to include in response ("Summarize for agent")
"""

# =================================
# 2. LSTM FROM SCRATCH
# =================================

print("\n" + "=" * 60)
print("2. LSTM FROM SCRATCH")
print("=" * 60)

class LSTMCellFromScratch(nn.Module):
    """
    Single LSTM cell implemented from scratch.
    
    This shows exactly what happens inside an LSTM.
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrices for efficiency
        # [h_{t-1}, x_t] -> [i_t, f_t, g_t, o_t]
        combined_size = input_size + hidden_size
        
        # Forget gate weights
        self.W_f = nn.Linear(combined_size, hidden_size)
        
        # Input gate weights
        self.W_i = nn.Linear(combined_size, hidden_size)
        
        # Candidate cell state weights
        self.W_c = nn.Linear(combined_size, hidden_size)
        
        # Output gate weights
        self.W_o = nn.Linear(combined_size, hidden_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass for one time step.
        
        Args:
            x: Input (batch_size, input_size)
            hidden: Tuple of (h, c) previous states
            
        Returns:
            h_new: New hidden state
            c_new: New cell state
        """
        batch_size = x.size(0)
        
        # Initialize if needed
        if hidden is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = hidden
        
        # Concatenate h and x
        combined = torch.cat([h_prev, x], dim=1)
        
        # Compute gates
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        c_tilde = torch.tanh(self.W_c(combined)) # Candidate cell
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        
        # Update cell state
        c_new = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


class LSTMFromScratch(nn.Module):
    """Complete LSTM for sequence processing."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cell = LSTMCellFromScratch(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, x, hidden=None):
        """
        Process entire sequence.
        
        Args:
            x: (batch_size, seq_length, input_size)
        """
        batch_size, seq_length, _ = x.shape
        
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hidden
        
        outputs = []
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq, hidden)
        
        # Output for each time step
        output = self.fc(outputs)
        
        return output, (h, c)


# Test our LSTM
lstm_scratch = LSTMFromScratch(10, 20, 5).to(device)
x = torch.randn(3, 7, 10, device=device)
output, (h, c) = lstm_scratch(x)

print("LSTM from scratch:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Hidden state shape: {h.shape}")
print(f"  Cell state shape: {c.shape}")

# =================================
# 3. VISUALIZE GATE ACTIVATIONS
# =================================

print("\n" + "=" * 60)
print("3. VISUALIZING GATE ACTIVATIONS")
print("=" * 60)

class LSTMWithGateLogging(nn.Module):
    """LSTM that logs gate activations for visualization."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        
    def forward(self, x):
        return self.lstm(x)


# Create model and hook to capture internal states
# Note: PyTorch LSTM doesn't expose gates directly, so we'll simulate

def visualize_gates():
    """Visualize how gates respond to input."""
    # Create synthetic input: signal at position 0, noise elsewhere
    seq_length = 20
    x = torch.randn(1, seq_length, 1) * 0.1
    x[0, 0, 0] = 1.0  # Signal at position 0
    x[0, 10, 0] = -1.0  # Different signal at position 10
    
    # Process through LSTM
    lstm = nn.LSTM(1, 16, batch_first=True)
    output, (h, c) = lstm(x)
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Input
    axes[0].plot(x.squeeze().numpy(), 'b-', linewidth=2)
    axes[0].set_title('Input Sequence')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Output (first few dimensions)
    for i in range(min(4, output.size(2))):
        axes[1].plot(output[0, :, i].detach().numpy(), label=f'Dim {i}')
    axes[1].set_title('LSTM Output (Hidden State)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Activation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Hidden state evolution (norm)
    hidden_norms = []
    h, c = None, None
    for t in range(seq_length):
        out, (h, c) = lstm(x[:, t:t+1, :], (h, c) if h is not None else None)
        hidden_norms.append(h.norm().item())
    
    axes[2].plot(hidden_norms, 'r-', linewidth=2)
    axes[2].set_title('Hidden State Magnitude Over Time')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('||h||')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_gates.png', dpi=100)
    print("Saved lstm_gates.png")
    plt.show()


visualize_gates()

# =================================
# 4. GRU (GATED RECURRENT UNIT)
# =================================

print("\n" + "=" * 60)
print("4. GRU (GATED RECURRENT UNIT)")
print("=" * 60)

"""
GRU: Simplified version of LSTM

Two gates instead of three:
1. RESET GATE (r_t): How much of past to forget
2. UPDATE GATE (z_t): How much of new state to use

Equations:
   z_t = sigmoid(W_z · [h_{t-1}, x_t])
   r_t = sigmoid(W_r · [h_{t-1}, x_t])
   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Advantages:
- Fewer parameters than LSTM
- Often performs just as well
- Faster to train
"""

class GRUCellFromScratch(nn.Module):
    """GRU cell implemented from scratch."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size
        
        # Reset gate
        self.W_r = nn.Linear(combined_size, hidden_size)
        
        # Update gate
        self.W_z = nn.Linear(combined_size, hidden_size)
        
        # Candidate hidden state
        self.W_h = nn.Linear(combined_size, hidden_size)
    
    def forward(self, x, h_prev=None):
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        combined = torch.cat([h_prev, x], dim=1)
        
        # Gates
        r_t = torch.sigmoid(self.W_r(combined))  # Reset gate
        z_t = torch.sigmoid(self.W_z(combined))  # Update gate
        
        # Candidate with reset gate applied
        combined_reset = torch.cat([r_t * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        
        # New hidden state
        h_new = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_new


# Compare parameter counts
input_size, hidden_size = 128, 256

lstm = nn.LSTM(input_size, hidden_size)
gru = nn.GRU(input_size, hidden_size)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f"Parameter comparison (input={input_size}, hidden={hidden_size}):")
print(f"  LSTM parameters: {lstm_params:,}")
print(f"  GRU parameters:  {gru_params:,}")
print(f"  GRU is {(1 - gru_params/lstm_params)*100:.1f}% smaller")
