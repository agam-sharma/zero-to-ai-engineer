# Source: AI_Learning_Cursor lines 808-830
# Original transcript phase: None - None
# Nearest header: #### Theory: What is a Vector?
# Title: Theory: What is a Vector?
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np
import matplotlib.pyplot as plt

# A vector is just an ordered list of numbers
# But geometrically, it represents a POINT or a DIRECTION

# Example: Customer as a vector
# Dimensions: [Monthly Revenue ($K), Number of Users, Support Tickets]
customer_a = np.array([50, 100, 5])
customer_b = np.array([200, 500, 20])
customer_c = np.array([30, 50, 2])

print("Customers represented as vectors:")
print(f"Customer A: {customer_a}")
print(f"Customer B: {customer_b}")
print(f"Customer C: {customer_c}")

# In AI, we process these vectors to:
# 1. Find similar customers (compare vectors)
# 2. Predict outcomes (transform vectors)
# 3. Cluster customers (group similar vectors)
