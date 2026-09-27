# Source: AI_Learning_Cursor lines 391-409
# Original transcript phase: None - None
# Nearest header: #### Core NumPy Concepts You MUST Master
# Title: Core NumPy Concepts You MUST Master
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np

# Creating arrays
a = np.array([1, 2, 3, 4, 5])          # 1D array (vector)
b = np.array([[1, 2, 3], [4, 5, 6]])   # 2D array (matrix)

# Understanding shapes - CRITICAL for AI
print(f"Shape of a: {a.shape}")  # (5,) - 5 elements
print(f"Shape of b: {b.shape}")  # (2, 3) - 2 rows, 3 columns

# Common initialization patterns in AI
zeros = np.zeros((3, 4))       # 3x4 matrix of zeros (used for padding)
ones = np.ones((2, 2))         # 2x2 matrix of ones
random = np.random.randn(3, 3) # 3x3 matrix of random numbers (weight initialization!)

print("Random matrix (like neural network weights):")
print(random)
