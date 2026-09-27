# Source: AI_Learning_Cursor lines 6129-6223
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: MNIST Data Loading and Exploration
# Title: MNIST DATASET LOADING
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MNIST DATASET LOADING
=====================
Loading and exploring the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =================================
# 1. DATA LOADING
# =================================

print("=" * 60)
print("1. LOADING MNIST DATASET")
print("=" * 60)

# Define transforms (preprocessing pipeline)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor (0-255 -> 0-1)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# =================================
# 2. EXPLORING THE DATA
# =================================

print("\n" + "=" * 60)
print("2. EXPLORING THE DATA")
print("=" * 60)

# Get a batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Labels shape: {labels.shape}")  # [64]
print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
print(f"Labels in batch: {labels[:10].tolist()}")

# Visualize some samples
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()  # Remove channel dimension
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels[i].item()}')
    ax.axis('off')

plt.suptitle('MNIST Samples')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=100)
print("Saved mnist_samples.png")
plt.show()

# Class distribution
all_labels = [label for _, label in train_dataset]
unique, counts = np.unique(all_labels, return_counts=True)
print("\nClass distribution:")
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} samples ({count/len(train_dataset)*100:.1f}%)")
