# Source: AI_Learning_Cursor lines 7074-7363
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Complete CNN Implementation
# Title: CNN FOR MNIST CLASSIFICATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
CNN FOR MNIST CLASSIFICATION
============================
Building a convolutional neural network for 99%+ accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. DATA PREPARATION
# =================================

print("=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training: {len(train_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# =================================
# 2. CNN MODEL
# =================================

print("\n" + "=" * 60)
print("2. CNN MODEL DEFINITION")
print("=" * 60)

class MNISTConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST.
    
    Architecture:
    - Conv layers extract features (edges, shapes, patterns)
    - Pooling layers reduce spatial dimensions
    - Fully connected layers make final classification
    """
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1: 1 input channel -> 32 filters
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1, 28, 28) -> (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 28, 28) -> (32, 14, 14)
            
            # Layer 2: 32 -> 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32, 14, 14) -> (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 14, 14) -> (64, 7, 7)
            
            # Layer 3: 64 -> 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (64, 7, 7) -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = MNISTConvNet().to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# =================================
# 3. TRAINING
# =================================

print("\n" + "=" * 60)
print("3. TRAINING")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, correct / total


# Training loop
n_epochs = 15
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print("Starting training...")
start_time = time.time()

for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1:2d}/{n_epochs} | "
          f"Train: {train_loss:.4f}, {train_acc:.2%} | "
          f"Test: {test_loss:.4f}, {test_acc:.2%}")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f} seconds")
print(f"Best test accuracy: {max(test_accs):.2%}")

# =================================
# 4. VISUALIZE RESULTS
# =================================

print("\n" + "=" * 60)
print("4. RESULTS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses, label='Test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train')
axes[1].plot(test_accs, label='Test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('CNN Training on MNIST')
plt.tight_layout()
plt.savefig('cnn_training.png', dpi=100)
print("Saved cnn_training.png")
plt.show()

# =================================
# 5. VISUALIZE FEATURE MAPS
# =================================

print("\n" + "=" * 60)
print("5. VISUALIZING FEATURE MAPS")
print("=" * 60)

# Get a sample image
sample_img, sample_label = next(iter(test_loader))
sample_img = sample_img[0:1].to(device)  # Single image

# Hook to capture intermediate outputs
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.conv_layers[0].register_forward_hook(get_activation('conv1'))
model.conv_layers[4].register_forward_hook(get_activation('conv2'))
model.conv_layers[8].register_forward_hook(get_activation('conv3'))

# Forward pass
model.eval()
with torch.no_grad():
    output = model(sample_img)

# Visualize
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

for i in range(8):
    # Conv layer 1
    axes[0, i].imshow(activations['conv1'][0, i].cpu().numpy(), cmap='viridis')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Conv1', fontsize=12)
    
    # Conv layer 2
    axes[1, i].imshow(activations['conv2'][0, i].cpu().numpy(), cmap='viridis')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel('Conv2', fontsize=12)
    
    # Conv layer 3
    axes[2, i].imshow(activations['conv3'][0, i].cpu().numpy(), cmap='viridis')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_ylabel('Conv3', fontsize=12)

plt.suptitle('Feature Maps at Different Layers')
plt.tight_layout()
plt.savefig('feature_maps.png', dpi=100)
print("Saved feature_maps.png")
plt.show()

# Save model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\nModel saved to mnist_cnn.pth")
