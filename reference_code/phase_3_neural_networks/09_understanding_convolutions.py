# Source: AI_Learning_Cursor lines 6864-7066
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Understanding Convolutions
# Title: UNDERSTANDING CONVOLUTIONS
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
UNDERSTANDING CONVOLUTIONS
==========================
Visualizing what convolution operations do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# =================================
# 1. WHAT IS A CONVOLUTION?
# =================================

print("=" * 60)
print("1. WHAT IS A CONVOLUTION?")
print("=" * 60)

"""
A convolution slides a small "kernel" (filter) over the image.
At each position, it computes the dot product between the kernel
and the image patch beneath it.

Different kernels detect different features:
- Edge detectors find boundaries
- Blur kernels smooth the image
- Sharpen kernels enhance details

The key insight: THE SAME KERNEL IS USED EVERYWHERE.
This means:
1. Far fewer parameters than fully connected layers
2. Translation invariance (detects features anywhere in image)
"""

# Load a sample image
mnist = datasets.MNIST('./data', train=True, download=True)
sample_image = mnist[0][0]  # Get first image
img_array = np.array(sample_image, dtype=np.float32)

print(f"Image shape: {img_array.shape}")  # (28, 28)

# Convert to tensor for PyTorch (add batch and channel dimensions)
img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
print(f"Tensor shape: {img_tensor.shape}")  # (1, 1, 28, 28)

# =================================
# 2. MANUAL CONVOLUTION KERNELS
# =================================

print("\n" + "=" * 60)
print("2. HAND-CRAFTED KERNELS")
print("=" * 60)

# Define some classic kernels
kernels = {
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    
    'edge_horizontal': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    
    'edge_vertical': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    
    'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    
    'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    
    'blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0,
    
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}

# Apply kernels and visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Original image
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

# Apply each kernel
for i, (name, kernel) in enumerate(kernels.items()):
    if i >= 7:
        break
    
    # Convert kernel to tensor
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32)
    kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    
    # Apply convolution
    output = F.conv2d(img_tensor, kernel_tensor, padding=1)
    output_np = output.squeeze().numpy()
    
    axes[i + 1].imshow(output_np, cmap='gray')
    axes[i + 1].set_title(f'{name}')
    axes[i + 1].axis('off')

plt.suptitle('Convolution with Different Kernels', fontsize=14)
plt.tight_layout()
plt.savefig('convolution_kernels.png', dpi=100)
print("Saved convolution_kernels.png")
plt.show()

# =================================
# 3. PyTorch Conv2d LAYER
# =================================

print("\n" + "=" * 60)
print("3. PyTorch Conv2d LAYER")
print("=" * 60)

# Create a convolutional layer
conv_layer = nn.Conv2d(
    in_channels=1,     # Input has 1 channel (grayscale)
    out_channels=16,   # Learn 16 different filters
    kernel_size=3,     # 3x3 kernels
    stride=1,          # Move 1 pixel at a time
    padding=1          # Add 1 pixel border to keep size
)

print("Conv2d layer:")
print(f"  Input channels: 1")
print(f"  Output channels (filters): 16")
print(f"  Kernel size: 3x3")
print(f"  Weight shape: {conv_layer.weight.shape}")  # (16, 1, 3, 3)
print(f"  Bias shape: {conv_layer.bias.shape}")      # (16,)
print(f"  Total parameters: {conv_layer.weight.numel() + conv_layer.bias.numel()}")

# Apply to image
output = conv_layer(img_tensor)
print(f"\nInput shape: {img_tensor.shape}")   # (1, 1, 28, 28)
print(f"Output shape: {output.shape}")         # (1, 16, 28, 28)

# Visualize learned filters
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flatten()):
    if i < 16:
        filter_output = output[0, i].detach().numpy()
        ax.imshow(filter_output, cmap='gray')
        ax.set_title(f'Filter {i}')
    ax.axis('off')

plt.suptitle('Output of 16 Learned Convolutional Filters')
plt.tight_layout()
plt.savefig('conv_layer_outputs.png', dpi=100)
print("Saved conv_layer_outputs.png")
plt.show()

# =================================
# 4. POOLING LAYERS
# =================================

print("\n" + "=" * 60)
print("4. POOLING LAYERS")
print("=" * 60)

"""
Pooling reduces spatial dimensions while keeping important features.
- Max Pooling: Takes the maximum value in each window
- Avg Pooling: Takes the average value in each window

This helps:
1. Reduce computation
2. Provide translation invariance
3. Prevent overfitting
"""

# Create pooling layers
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Apply to conv output
pooled_max = max_pool(output)
pooled_avg = avg_pool(output)

print(f"Before pooling: {output.shape}")      # (1, 16, 28, 28)
print(f"After max pool: {pooled_max.shape}")  # (1, 16, 14, 14)
print(f"After avg pool: {pooled_avg.shape}")  # (1, 16, 14, 14)

# Visualize pooling effect
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(output[0, 0].detach().numpy(), cmap='gray')
axes[0].set_title(f'Original: {output[0, 0].shape}')

axes[1].imshow(pooled_max[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title(f'Max Pool: {pooled_max[0, 0].shape}')

axes[2].imshow(pooled_avg[0, 0].detach().numpy(), cmap='gray')
axes[2].set_title(f'Avg Pool: {pooled_avg[0, 0].shape}')

for ax in axes:
    ax.axis('off')

plt.suptitle('Effect of Pooling')
plt.tight_layout()
plt.savefig('pooling_effect.png', dpi=100)
print("Saved pooling_effect.png")
plt.show()
