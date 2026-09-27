# Source: AI_Learning_Cursor lines 517-624
# Original transcript phase: None - None
# Nearest header: #### 🏗️ Build Assignment: Matrix Operations Library
# Title: Your first AI-related code: Implement matrix operations from scratch!
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
Your first AI-related code: Implement matrix operations from scratch!
Then compare with NumPy to understand what NumPy does for you.
"""
import numpy as np
import time

def manual_dot_product(a: list, b: list) -> float:
    """
    Implement dot product WITHOUT using NumPy.
    Formula: sum of (a[i] * b[i]) for all i
    """
    # YOUR CODE HERE
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def manual_matrix_multiply(A: list, B: list) -> list:
    """
    Implement matrix multiplication WITHOUT using NumPy.
    A is (m x n), B is (n x p), result is (m x p)
    result[i][j] = dot product of row i of A and column j of B
    """
    # YOUR CODE HERE
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # Verify dimensions
    if n != len(B):
        raise ValueError(f"Cannot multiply: A has {n} columns, B has {len(B)} rows")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(p)] for _ in range(m)]
    
    # Compute each element
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def compare_performance():
    """
    Compare your implementation with NumPy.
    This will show you WHY we use NumPy!
    """
    # Create test data
    size = 200
    A_list = [[float(i + j) for j in range(size)] for i in range(size)]
    B_list = [[float(i * j) for j in range(size)] for i in range(size)]
    
    A_np = np.array(A_list)
    B_np = np.array(B_list)
    
    # Time manual implementation
    start = time.time()
    result_manual = manual_matrix_multiply(A_list, B_list)
    manual_time = time.time() - start
    
    # Time NumPy
    start = time.time()
    result_numpy = np.matmul(A_np, B_np)
    numpy_time = time.time() - start
    
    print(f"Matrix size: {size}x{size}")
    print(f"Manual implementation: {manual_time:.4f} seconds")
    print(f"NumPy implementation: {numpy_time:.6f} seconds")
    print(f"NumPy is {manual_time/numpy_time:.0f}x faster!")
    
    # Verify correctness
    result_manual_np = np.array(result_manual)
    if np.allclose(result_manual_np, result_numpy):
        print("✅ Your implementation is correct!")
    else:
        print("❌ Results don't match. Debug your code.")


if __name__ == "__main__":
    # Test dot product
    print("Testing dot product...")
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = manual_dot_product(a, b)
    expected = 32  # 1*4 + 2*5 + 3*6
    print(f"manual_dot_product({a}, {b}) = {result}")
    print(f"Expected: {expected}, {'✅ Correct!' if result == expected else '❌ Wrong!'}")
    
    print("\nTesting matrix multiplication...")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = manual_matrix_multiply(A, B)
    expected = [[19, 22], [43, 50]]
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print('✅ Correct!' if result == expected else '❌ Wrong!')
    
    print("\nPerformance comparison...")
    compare_performance()
