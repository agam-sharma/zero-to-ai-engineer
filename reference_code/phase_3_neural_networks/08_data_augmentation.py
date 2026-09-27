# Source: AI_Learning_Cursor lines 6561-6812
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Improved Training with Augmentation
# Title: DATA AUGMENTATION AND REGULARIZATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
DATA AUGMENTATION AND REGULARIZATION
====================================
Techniques to improve model generalization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# =================================
# 1. DATA AUGMENTATION
# =================================

print("=" * 60)
print("1. DATA AUGMENTATION")
print("=" * 60)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),              # Rotate +/- 10 degrees
    transforms.RandomAffine(                    # Random affine transforms
        degrees=0,
        translate=(0.1, 0.1),                   # Shift up to 10%
        scale=(0.9, 1.1)                        # Scale 90-110%
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST('./data', train=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

# Visualize augmentations
print("Visualizing data augmentation...")
original_img, label = datasets.MNIST('./data', train=True)[0]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes[0, 0].imshow(original_img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, ax in enumerate(axes.flat[1:]):
    augmented, _ = train_dataset[0]  # Same image with random augmentation
    ax.imshow(augmented.squeeze(), cmap='gray')
    ax.set_title(f'Augmented {i+1}')
    ax.axis('off')

plt.suptitle('Data Augmentation Examples')
plt.tight_layout()
plt.savefig('data_augmentation.png', dpi=100)
print("Saved data_augmentation.png")
plt.show()

# =================================
# 2. MODEL WITH DROPOUT
# =================================

print("\n" + "=" * 60)
print("2. MODEL WITH DROPOUT")
print("=" * 60)

class MNISTClassifierWithDropout(nn.Module):
    """
    MLP with dropout regularization.
    
    Dropout randomly "turns off" neurons during training,
    forcing the network to not rely too heavily on any single neuron.
    This prevents overfitting.
    """
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 30% of neurons randomly dropped
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = MNISTClassifierWithDropout(dropout_rate=0.3).to(device)
print(model)

# =================================
# 3. TRAINING WITH EARLY STOPPING
# =================================

print("\n" + "=" * 60)
print("3. TRAINING WITH EARLY STOPPING")
print("=" * 60)

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
early_stopping = EarlyStopping(patience=5)

# Training functions (reuse from before)
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


# Training loop with early stopping
n_epochs = 50
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print("Training with dropout and early stopping...")
for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1:2d}: "
          f"Train: {train_loss:.4f}, {train_acc:.2%} | "
          f"Test: {test_loss:.4f}, {test_acc:.2%}")
    
    # Check early stopping
    if early_stopping(test_loss, model):
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        model.load_state_dict(early_stopping.best_model)
        break

print(f"\nBest test accuracy: {max(test_accs):.2%}")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss with Dropout + Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy with Dropout + Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_with_regularization.png', dpi=100)
print("Saved training_with_regularization.png")
plt.show()
