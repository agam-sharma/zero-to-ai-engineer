# Source: AI_Learning_Cursor lines 834-876
# Original transcript phase: None - None
# Nearest header: #### Key Vector Operations
# Title: Key Vector Operations
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np

# 1. VECTOR ADDITION - Combining features
v1 = np.array([1, 2])
v2 = np.array([3, 1])
v_sum = v1 + v2
print(f"Vector addition: {v1} + {v2} = {v_sum}")
# Interpretation: Combining effects of two transformations

# 2. SCALAR MULTIPLICATION - Scaling
v = np.array([2, 3])
scaled = 2 * v
print(f"Scalar multiplication: 2 * {v} = {scaled}")
# Interpretation: Doubling the magnitude (strength) of a signal

# 3. DOT PRODUCT - Similarity measure!
a = np.array([1, 0])  # Points right
b = np.array([0, 1])  # Points up
c = np.array([1, 0])  # Points right (same as a)

print(f"\nDot products (similarity):")
print(f"a·b = {np.dot(a, b)}")  # 0 - orthogonal (unrelated)
print(f"a·c = {np.dot(a, c)}")  # 1 - same direction (similar)

# WHY THIS MATTERS IN AI:
# The dot product measures how "aligned" two vectors are
# - High dot product = similar direction = similar meaning
# - Zero dot product = perpendicular = unrelated
# - Negative dot product = opposite = opposite meaning

# 4. VECTOR NORM (MAGNITUDE) - Length of vector
v = np.array([3, 4])
magnitude = np.linalg.norm(v)  # sqrt(3² + 4²) = 5
print(f"\nMagnitude of {v}: {magnitude}")

# 5. UNIT VECTOR - Direction without magnitude
unit_v = v / magnitude
print(f"Unit vector: {unit_v}")  # [0.6, 0.8] - length is 1
print(f"Magnitude of unit vector: {np.linalg.norm(unit_v)}")

# Normalizing vectors is CRUCIAL in AI - it makes comparisons fair
