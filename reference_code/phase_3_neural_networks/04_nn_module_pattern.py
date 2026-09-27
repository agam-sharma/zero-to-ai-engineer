# Source: AI_Learning_Cursor lines 5559-5759
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Neural Network with nn.Module
# Title: NEURAL NETWORKS WITH nn.Module
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
NEURAL NETWORKS WITH nn.Module
==============================
This file covers:
1. nn.Module basics
2. Building layers
3. Forward pass
4. Loss functions and optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =================================
# 1. YOUR FIRST nn.Module
# =================================

print("=" * 60)
print("1. BASIC nn.Module NETWORK")
print("=" * 60)

class SimpleNetwork(nn.Module):
    """
    A simple neural network using nn.Module.
    
    This is the standard pattern used in ALL PyTorch code:
    1. Inherit from nn.Module
    2. Define layers in __init__
    3. Define forward pass in forward()
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the network layers.
        
        Note: super().__init__() is REQUIRED - it sets up
        PyTorch's internal bookkeeping.
        """
        super().__init__()
        
        # Define layers
        # nn.Linear is a fully connected layer: output = input @ W + b
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Define the forward pass.
        
        PyTorch automatically tracks operations for backprop!
        """
        # Layer 1 + ReLU
        x = self.layer1(x)
        x = self.relu(x)
        
        # Layer 2 + ReLU
        x = self.layer2(x)
        x = self.relu(x)
        
        # Layer 3 + Sigmoid (for binary classification)
        x = self.layer3(x)
        x = self.sigmoid(x)
        
        return x


# Create network
model = SimpleNetwork(input_size=2, hidden_size=8, output_size=1)
print(f"Network architecture:\n{model}")

# See all parameters
print(f"\nNetwork parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape = {param.shape}, total = {param.numel()}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# Move to GPU
model = model.to(device)
print(f"\nModel device: {next(model.parameters()).device}")

# =================================
# 2. LOSS FUNCTIONS
# =================================

print("\n" + "=" * 60)
print("2. LOSS FUNCTIONS")
print("=" * 60)

# Binary Cross Entropy (for binary classification)
bce_loss = nn.BCELoss()

# Cross Entropy (for multi-class classification)
ce_loss = nn.CrossEntropyLoss()

# Mean Squared Error (for regression)
mse_loss = nn.MSELoss()

# Example: BCE Loss
pred = torch.tensor([0.9, 0.2, 0.8])
target = torch.tensor([1.0, 0.0, 1.0])
loss = bce_loss(pred, target)
print(f"BCE Loss example: pred={pred.tolist()}, target={target.tolist()}")
print(f"  Loss = {loss.item():.4f}")

# =================================
# 3. OPTIMIZERS
# =================================

print("\n" + "=" * 60)
print("3. OPTIMIZERS")
print("=" * 60)

# SGD (Stochastic Gradient Descent) - what we implemented in Phase 1
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam - adaptive learning rate, usually works better
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW - Adam with weight decay (regularization)
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print("Available optimizers:")
print("  - SGD: Simple, but requires tuning")
print("  - Adam: Adaptive, good default choice")
print("  - AdamW: Adam + weight decay, best for transformers")

# =================================
# 4. COMPLETE TRAINING LOOP
# =================================

print("\n" + "=" * 60)
print("4. COMPLETE TRAINING LOOP")
print("=" * 60)

# Generate XOR dataset
X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
y_xor = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

# Create fresh model
model = SimpleNetwork(input_size=2, hidden_size=8, output_size=1).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
losses = []
print("Training on XOR problem...")

for epoch in range(1000):
    # 1. Forward pass
    predictions = model(X_xor)
    
    # 2. Compute loss
    loss = criterion(predictions, y_xor)
    losses.append(loss.item())
    
    # 3. Backward pass
    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Compute gradients
    
    # 4. Update weights
    optimizer.step()        # Apply gradients
    
    if epoch % 200 == 0:
        accuracy = ((predictions >= 0.5) == y_xor).float().mean()
        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.2%}")

# Final results
print("\nFinal predictions:")
with torch.no_grad():  # Don't compute gradients for inference
    final_pred = model(X_xor)
    for x, y, pred in zip(X_xor, y_xor, final_pred):
        print(f"  {x.tolist()} -> {pred.item():.4f} (true: {y.item()})")

# Plot training curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training with PyTorch')
plt.grid(True, alpha=0.3)
plt.savefig('pytorch_xor_training.png', dpi=100)
print("\nSaved pytorch_xor_training.png")
plt.show()
