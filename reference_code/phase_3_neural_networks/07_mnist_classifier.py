# Source: AI_Learning_Cursor lines 6231-6553
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Complete MNIST Classifier
# Title: MNIST DIGIT CLASSIFIER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MNIST DIGIT CLASSIFIER
======================
Complete training pipeline for MNIST classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# Set device and seed
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. DATA PREPARATION
# =================================

print("=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")

# =================================
# 2. MODEL DEFINITION
# =================================

print("\n" + "=" * 60)
print("2. MODEL DEFINITION")
print("=" * 60)

class MNISTClassifier(nn.Module):
    """
    Multi-layer perceptron for MNIST classification.
    
    Input: 28x28 = 784 pixels
    Output: 10 classes (digits 0-9)
    """
    
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()  # 28x28 -> 784
        
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 output classes
        )
        # Note: No softmax here because CrossEntropyLoss includes it
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


model = MNISTClassifier().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# =================================
# 3. TRAINING SETUP
# =================================

print("\n" + "=" * 60)
print("3. TRAINING SETUP")
print("=" * 60)

# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (reduce LR when progress stalls)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=2,
    verbose=True
)

print(f"Criterion: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Scheduler: ReduceLROnPlateau")

# =================================
# 4. TRAINING LOOP
# =================================

print("\n" + "=" * 60)
print("4. TRAINING")
print("=" * 60)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, correct / total


# Training
n_epochs = 15
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print("Starting training...")
start_time = time.time()

for epoch in range(n_epochs):
    epoch_start = time.time()
    
    # Train and evaluate
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(test_loss)
    
    # Record metrics
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1:2d}/{n_epochs} ({epoch_time:.1f}s) | "
          f"Train: {train_loss:.4f}, {train_acc:.2%} | "
          f"Test: {test_loss:.4f}, {test_acc:.2%} | "
          f"LR: {current_lr:.6f}")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f} seconds")
print(f"Best test accuracy: {max(test_accs):.2%}")

# =================================
# 5. VISUALIZATION
# =================================

print("\n" + "=" * 60)
print("5. RESULTS VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curves
axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses, label='Test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(train_accs, label='Train')
axes[1].plot(test_accs, label='Test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Sample predictions
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    _, predictions = outputs.max(1)

# Show some predictions
images_np = images.cpu()
labels_np = labels.numpy()
predictions_np = predictions.cpu().numpy()

axes[2].axis('off')
axes[2].set_title('Sample Predictions')

# Create mini grid
mini_fig, mini_axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(mini_axes.flat):
    img = images_np[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    color = 'green' if predictions_np[i] == labels_np[i] else 'red'
    ax.set_title(f'Pred: {predictions_np[i]}, True: {labels_np[i]}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=100)
print("Saved mnist_predictions.png")
plt.show()

# Save the training curves figure
fig.savefig('mnist_training_curves.png', dpi=100)
print("Saved mnist_training_curves.png")

# =================================
# 6. CONFUSION MATRIX
# =================================

print("\n" + "=" * 60)
print("6. CONFUSION MATRIX")
print("=" * 60)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get all predictions
all_predictions = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('mnist_confusion_matrix.png', dpi=100)
print("Saved mnist_confusion_matrix.png")
plt.show()

# Per-class accuracy
print("\nPer-class accuracy:")
for digit in range(10):
    mask = np.array(all_labels) == digit
    acc = (np.array(all_predictions)[mask] == digit).mean()
    print(f"  Digit {digit}: {acc:.2%}")

# Save model
torch.save(model.state_dict(), 'mnist_classifier.pth')
print("\nModel saved to mnist_classifier.pth")
