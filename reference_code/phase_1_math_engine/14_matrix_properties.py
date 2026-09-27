# Source: AI_Learning_Cursor lines 1151-1195
# Original transcript phase: None - None
# Nearest header: #### Matrix Properties Important for AI
# Title: Matrix Properties Important for AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np

# 1. TRANSPOSE - Swap rows and columns
# Used constantly in neural networks (gradient computation)
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(f"A shape: {A.shape}")           # (2, 3)
print(f"A.T shape: {A.T.shape}")       # (3, 2)
print(f"A:\n{A}")
print(f"A.T:\n{A.T}")

# 2. MATRIX MULTIPLICATION RULES
# (m × n) @ (n × p) = (m × p)
# The inner dimensions MUST match!

A = np.random.randn(3, 4)  # 3×4
B = np.random.randn(4, 2)  # 4×2
C = A @ B                   # 3×2 ✅ Works!
print(f"\n(3×4) @ (4×2) = {C.shape}")  # (3, 2)

try:
    D = A @ np.random.randn(5, 2)  # 3×4 @ 5×2 = ❌ Error!
except ValueError as e:
    print(f"Error: {e}")

# 3. IDENTITY MATRIX - Does nothing (multiplying by 1)
# Used for initialization sometimes
I = np.eye(3)  # 3×3 identity
print(f"\nIdentity matrix:\n{I}")

v = np.array([5, 10, 15])
print(f"I @ v = {I @ v}")  # Same as v!

# 4. MATRIX INVERSE - Undo a transformation
# If A @ x = b, then x = A⁻¹ @ b
# Used in some optimization techniques
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print(f"\nA:\n{A}")
print(f"A inverse:\n{A_inv}")
print(f"A @ A_inv:\n{A @ A_inv}")  # Should be identity (with floating point error)
