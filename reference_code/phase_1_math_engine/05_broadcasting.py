# Source: AI_Learning_Cursor lines 462-481
# Original transcript phase: None - None
# Nearest header: #### Core NumPy Concepts You MUST Master
# Title: Core NumPy Concepts You MUST Master
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# Broadcasting lets you add/multiply arrays of different shapes
# This is used for adding BIASES in neural networks

# 4 samples, 2 features each
data = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
])

# One bias value per feature (gets added to ALL samples)
bias = np.array([10, 100])

# Broadcasting: (4, 2) + (2,) → bias is "stretched" to (4, 2)
result = data + bias
print("After adding bias:\n", result)
# [[11, 102], [13, 104], [15, 106], [17, 108]]
