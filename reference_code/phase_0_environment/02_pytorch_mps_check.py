# Source: AI_Learning_Cursor lines 298-331
# Original transcript phase: None - None
# Nearest header: #### Step-by-Step Setup Guide
# Title: Step-by-Step Setup Guide
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import torch

# Check if MPS is available
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    # Create a tensor on MPS device
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform matrix multiplication on GPU
    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.mps.synchronize()  # Wait for GPU to finish
    print(f"100 matrix multiplications on MPS: {time.time() - start:.3f} seconds")
    
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start = time.time()
    for _ in range(100):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    print(f"100 matrix multiplications on CPU: {time.time() - start:.3f} seconds")
    
    print("\n✅ Your M3 Pro is ready for AI development!")
else:
    print("❌ MPS not available. Check your PyTorch installation.")
