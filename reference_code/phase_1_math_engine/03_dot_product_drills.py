# Source: AI_Learning_Cursor lines 412-428
# Original transcript phase: None - None
# Nearest header: #### Core NumPy Concepts You MUST Master
# Title: Core NumPy Concepts You MUST Master
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# The dot product is THE fundamental operation in neural networks
# Every neuron computes: output = dot(inputs, weights) + bias

# Vector dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Manual calculation: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_result = np.dot(a, b)
print(f"Dot product: {dot_result}")  # 32

# WHY THIS MATTERS IN AI:
# Imagine 'a' is [pixel1, pixel2, pixel3] from an image
# And 'b' is [weight1, weight2, weight3] of a neuron
# The dot product gives ONE number representing "how much this neuron fires"
