# Source: AI_Learning_Cursor lines 431-459
# Original transcript phase: None - None
# Nearest header: #### Core NumPy Concepts You MUST Master
# Title: Core NumPy Concepts You MUST Master
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# Matrix multiplication = processing MULTIPLE inputs at once

# Inputs: 4 samples, each with 3 features
# (Like 4 Salesforce records, each with 3 fields)
inputs = np.array([
    [1.0, 2.0, 3.0],   # Sample 1
    [4.0, 5.0, 6.0],   # Sample 2
    [7.0, 8.0, 9.0],   # Sample 3
    [1.5, 2.5, 3.5],   # Sample 4
])
print(f"Inputs shape: {inputs.shape}")  # (4, 3)

# Weights: Transform 3 features into 2 outputs
# (Like a formula field that computes 2 new values from 3 inputs)
weights = np.array([
    [0.1, 0.2],  # How feature 1 contributes to output 1 and 2
    [0.3, 0.4],  # How feature 2 contributes
    [0.5, 0.6],  # How feature 3 contributes
])
print(f"Weights shape: {weights.shape}")  # (3, 2)

# Matrix multiplication: (4, 3) @ (3, 2) = (4, 2)
outputs = np.matmul(inputs, weights)  # or: inputs @ weights
print(f"Outputs shape: {outputs.shape}")  # (4, 2)
print("Outputs:\n", outputs)

# EACH ROW of outputs is one sample processed through the "neuron layer"
