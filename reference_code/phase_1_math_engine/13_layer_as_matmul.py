# Source: AI_Learning_Cursor lines 1104-1147
# Original transcript phase: None - None
# Nearest header: #### Neural Network Connection
# Title: In neural networks, each layer performs: output = activation(input @ weights + bias)
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
In neural networks, each layer performs: output = activation(input @ weights + bias)

Let's see how this transforms data!
"""
import numpy as np

# INPUT: 4 samples, each with 3 features
# Think of this as 4 Salesforce records with 3 fields each
inputs = np.array([
    [1.0, 0.5, 0.2],  # Sample 1
    [0.3, 0.8, 0.1],  # Sample 2
    [0.9, 0.1, 0.7],  # Sample 3
    [0.2, 0.6, 0.3],  # Sample 4
])
print(f"Input shape: {inputs.shape}")  # (4, 3) = 4 samples × 3 features

# WEIGHTS: Transform 3 features → 2 features
# The neural network LEARNS these weights during training
weights = np.array([
    [0.5, -0.3],  # How feature 1 maps to output 1 and 2
    [0.2, 0.8],   # How feature 2 maps
    [-0.1, 0.4],  # How feature 3 maps
])
print(f"Weights shape: {weights.shape}")  # (3, 2)

# BIAS: Added to each output
bias = np.array([0.1, -0.1])

# FORWARD PASS: The transformation
# Shape: (4, 3) @ (3, 2) = (4, 2)
outputs = inputs @ weights + bias

print(f"Output shape: {outputs.shape}")  # (4, 2)
print("\nTransformed data:")
for i, (inp, out) in enumerate(zip(inputs, outputs)):
    print(f"  Sample {i+1}: {inp} → {out}")

# WHAT JUST HAPPENED?
# We transformed 4 samples from 3-dimensional space to 2-dimensional space
# Each sample now has 2 "learned" features instead of 3 original features
# The weights determine WHAT those new features represent
