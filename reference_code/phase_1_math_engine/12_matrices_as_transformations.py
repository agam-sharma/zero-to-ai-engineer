# Source: AI_Learning_Cursor lines 1060-1100
# Original transcript phase: None - None
# Nearest header: #### Core Concepts
# Title: Core Concepts
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np
import matplotlib.pyplot as plt

# A matrix is a 2D array of numbers
# It represents a TRANSFORMATION of space

# Example: A 2x2 matrix transforms 2D vectors
transformation = np.array([
    [2, 0],   # Row 1: How x-component is transformed
    [0, 2],   # Row 2: How y-component is transformed
])

# This particular matrix SCALES everything by 2
input_vector = np.array([1, 1])
output_vector = transformation @ input_vector  # @ is matrix multiplication
print(f"Input: {input_vector}")
print(f"Output: {output_vector}")  # [2, 2] - scaled by 2!

# Different matrices do different things:
# 1. SCALING
scale_matrix = np.array([[2, 0], [0, 2]])  # Uniform scale

# 2. ROTATION (90 degrees counterclockwise)
rotate_matrix = np.array([[0, -1], [1, 0]])

# 3. SHEARING (slant)
shear_matrix = np.array([[1, 1], [0, 1]])

# 4. REFLECTION (flip over x-axis)
reflect_matrix = np.array([[1, 0], [0, -1]])

# Let's see them in action
v = np.array([1, 0])  # Unit vector pointing right

print("\nTransformations of [1, 0]:")
print(f"Scaled: {scale_matrix @ v}")      # [2, 0]
print(f"Rotated: {rotate_matrix @ v}")    # [0, 1] - now points up!
print(f"Sheared: {shear_matrix @ v}")     # [1, 0]
print(f"Reflected: {reflect_matrix @ v}") # [1, 0]
