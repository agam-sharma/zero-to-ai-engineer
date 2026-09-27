# Source: AI_Learning_Cursor lines 736-770
# Original transcript phase: None - None
# Nearest header: #### Key Python Patterns Used in AI
# Title: Key Python Patterns Used in AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np

# Training AI models often requires managing resources
# Example: Tracking gradients (simplified)

class GradientTracker:
    """
    Context manager to track computations for backpropagation.
    
    PyTorch uses this pattern with torch.no_grad()
    """
    _tracking = False
    
    def __enter__(self):
        GradientTracker._tracking = True
        print("📊 Started tracking gradients")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        GradientTracker._tracking = False
        print("📊 Stopped tracking gradients")
        return False


# Usage
with GradientTracker():
    # All computations here would be tracked for backprop
    x = np.array([1, 2, 3])
    y = x ** 2
    print(f"Computed: {y}")

# Outside the context - no tracking
print(f"Tracking active: {GradientTracker._tracking}")
