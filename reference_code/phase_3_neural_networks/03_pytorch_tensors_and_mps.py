# Source: AI_Learning_Cursor lines 5311-5545
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: PyTorch Basics and MPS Setup
# Title: PYTORCH FUNDAMENTALS
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
PYTORCH FUNDAMENTALS
====================
This file covers:
1. Tensor creation and operations
2. GPU (MPS) acceleration on Mac M3 Pro
3. Comparison with NumPy
4. Automatic differentiation (autograd)
"""

import torch
import numpy as np
import time

print("=" * 60)
print("PYTORCH FUNDAMENTALS")
print("=" * 60)

# =================================
# 1. TENSOR BASICS
# =================================

print("\n1. TENSOR CREATION")
print("-" * 40)

# Creating tensors (like NumPy arrays, but with GPU support)
# From Python list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"From list: {tensor_from_list}")

# From NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"From NumPy:\n{tensor_from_numpy}")

# Common initialization patterns
zeros = torch.zeros(3, 4)           # 3x4 matrix of zeros
ones = torch.ones(2, 2)             # 2x2 matrix of ones
random = torch.randn(3, 3)          # Random normal distribution
eye = torch.eye(3)                  # Identity matrix

print(f"Zeros shape: {zeros.shape}")
print(f"Random tensor:\n{random}")

# Tensor properties
print(f"\nTensor properties:")
print(f"  Shape: {random.shape}")
print(f"  Data type: {random.dtype}")
print(f"  Device: {random.device}")  # CPU by default

# =================================
# 2. MPS (GPU) SETUP FOR MAC M3 PRO
# =================================

print("\n2. MPS (GPU) CONFIGURATION")
print("-" * 40)

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU (MPS not available)")

# Move tensors to GPU
cpu_tensor = torch.randn(1000, 1000)
gpu_tensor = cpu_tensor.to(device)
print(f"CPU tensor device: {cpu_tensor.device}")
print(f"GPU tensor device: {gpu_tensor.device}")

# Create tensor directly on GPU
direct_gpu = torch.randn(1000, 1000, device=device)
print(f"Direct GPU tensor device: {direct_gpu.device}")

# =================================
# 3. GPU VS CPU PERFORMANCE
# =================================

print("\n3. PERFORMANCE COMPARISON")
print("-" * 40)

def benchmark_matmul(size, device, num_iterations=100):
    """Benchmark matrix multiplication on a device."""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    for _ in range(10):
        c = torch.matmul(a, b)
    
    # Synchronize before timing
    if device.type == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    for _ in range(num_iterations):
        c = torch.matmul(a, b)
    
    # Synchronize after operations
    if device.type == "mps":
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    return elapsed

# Compare CPU vs MPS
size = 2000
print(f"Matrix multiplication: {size}x{size} matrices, 100 iterations")

cpu_time = benchmark_matmul(size, torch.device("cpu"))
print(f"  CPU time: {cpu_time:.3f} seconds")

if torch.backends.mps.is_available():
    mps_time = benchmark_matmul(size, torch.device("mps"))
    print(f"  MPS time: {mps_time:.3f} seconds")
    print(f"  Speedup: {cpu_time / mps_time:.1f}x faster on GPU!")

# =================================
# 4. TENSOR OPERATIONS
# =================================

print("\n4. TENSOR OPERATIONS")
print("-" * 40)

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise operations
print(f"a + b:\n{a + b}")
print(f"a * b (element-wise):\n{a * b}")

# Matrix multiplication (three equivalent ways)
print(f"Matrix multiply (torch.matmul):\n{torch.matmul(a, b)}")
print(f"Matrix multiply (@ operator):\n{a @ b}")
print(f"Matrix multiply (torch.mm):\n{torch.mm(a, b)}")

# Common operations
x = torch.tensor([1., 2., 3., 4., 5.])
print(f"\nVector: {x}")
print(f"  Sum: {x.sum()}")
print(f"  Mean: {x.mean()}")
print(f"  Max: {x.max()}")
print(f"  Argmax: {x.argmax()}")  # Index of max value

# Reshaping
batch = torch.randn(32, 28, 28)  # 32 images, 28x28 pixels
flattened = batch.view(32, -1)   # Flatten to 32 x 784
print(f"\nReshape: {batch.shape} -> {flattened.shape}")

# Alternative reshape methods
reshaped = batch.reshape(32, 784)
flattened2 = batch.flatten(start_dim=1)
print(f"reshape: {reshaped.shape}")
print(f"flatten: {flattened2.shape}")

# =================================
# 5. AUTOGRAD - AUTOMATIC DIFFERENTIATION
# =================================

print("\n5. AUTOGRAD (AUTOMATIC DIFFERENTIATION)")
print("-" * 40)

# This is the MAGIC of PyTorch - it computes gradients automatically!

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}")

# Perform operations
y = x ** 2          # y = x^2
z = y.sum()         # z = sum(y) = x1^2 + x2^2
print(f"y = x^2 = {y}")
print(f"z = sum(y) = {z}")

# Compute gradients (backpropagation!)
z.backward()

# dz/dx = 2x
print(f"Gradient dz/dx = {x.grad}")  # [4.0, 6.0] = [2*2, 2*3]

# WHY THIS MATTERS:
# In Phase 1, you computed gradients manually (chain rule, etc.)
# PyTorch does this AUTOMATICALLY for ANY computation!
# No matter how complex your neural network, one call to .backward()
# computes ALL the gradients you need.

print("\n" + "=" * 60)
print("AUTOGRAD EXAMPLE: Linear Regression")
print("=" * 60)

# Let's redo linear regression with autograd
torch.manual_seed(42)

# Generate data: y = 3x + 2 + noise
X = torch.randn(100, 1)
y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5

# Parameters to learn (with gradient tracking)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1

print("Training linear regression with autograd...")
for epoch in range(100):
    # Forward pass
    y_pred = X * w + b
    
    # Compute loss
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward pass (computes gradients automatically!)
    loss.backward()
    
    # Update parameters (gradient descent)
    with torch.no_grad():  # Don't track these operations
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\nLearned: y = {w.item():.3f}x + {b.item():.3f}")
print(f"True:    y = 3.000x + 2.000")
