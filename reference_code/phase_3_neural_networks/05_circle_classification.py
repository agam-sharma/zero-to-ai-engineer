# Source: AI_Learning_Cursor lines 5767-6080
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Circle Classification with PyTorch
# Title: COMPLETE CLASSIFIER EXAMPLE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
COMPLETE CLASSIFIER EXAMPLE
===========================
Building a full classifier with:
- Data loading
- Model definition
- Training loop
- Evaluation
- Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set device and seed
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

# =================================
# 1. CUSTOM DATASET
# =================================

print("=" * 60)
print("1. CREATING DATASET")
print("=" * 60)

class CircleDataset(Dataset):
    """
    Custom dataset for circle classification.
    
    PyTorch Dataset pattern:
    1. Inherit from Dataset
    2. Implement __len__ (return dataset size)
    3. Implement __getitem__ (return one sample)
    """
    
    def __init__(self, n_samples=1000):
        """Generate circular dataset."""
        # Inner circle (class 0)
        n_inner = n_samples // 2
        theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
        r_inner = np.random.uniform(0, 1, n_inner)
        X_inner = np.column_stack([
            r_inner * np.cos(theta_inner),
            r_inner * np.sin(theta_inner)
        ])
        
        # Outer ring (class 1)
        n_outer = n_samples - n_inner
        theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
        r_outer = np.random.uniform(1.5, 2.5, n_outer)
        X_outer = np.column_stack([
            r_outer * np.cos(theta_outer),
            r_outer * np.sin(theta_outer)
        ])
        
        # Combine
        self.X = np.vstack([X_inner, X_outer]).astype(np.float32)
        self.y = np.array([0] * n_inner + [1] * n_outer, dtype=np.float32)
        
        # Shuffle
        idx = np.random.permutation(n_samples)
        self.X = self.X[idx]
        self.y = self.y[idx]
        
        # Convert to tensors
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create datasets
train_dataset = CircleDataset(n_samples=1000)
test_dataset = CircleDataset(n_samples=200)

# Create DataLoaders (handles batching, shuffling, etc.)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: 32")
print(f"Training batches: {len(train_loader)}")

# =================================
# 2. MODEL DEFINITION
# =================================

print("\n" + "=" * 60)
print("2. MODEL DEFINITION")
print("=" * 60)

class CircleClassifier(nn.Module):
    """
    Neural network for circle classification.
    
    Using nn.Sequential for cleaner code.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define network using Sequential
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


model = CircleClassifier().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# =================================
# 3. TRAINING FUNCTION
# =================================

print("\n" + "=" * 60)
print("3. TRAINING")
print("=" * 60)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Set to training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        # Move to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch_X.size(0)
        predicted_labels = (predictions >= 0.5).float()
        correct += (predicted_labels == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed for evaluation
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            predicted_labels = (predictions >= 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
n_epochs = 50

# Track metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# Training loop
print("Starting training...")
for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%} | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2%}")

print("\nTraining complete!")

# =================================
# 4. VISUALIZATION
# =================================

print("\n" + "=" * 60)
print("4. VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves
axes[0, 0].plot(train_losses, label='Train')
axes[0, 0].plot(test_losses, label='Test')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[0, 1].plot(train_accs, label='Train')
axes[0, 1].plot(test_accs, label='Test')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Data distribution
X_np = test_dataset.X.numpy()
y_np = test_dataset.y.numpy().flatten()
axes[1, 0].scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='blue', label='Class 0', alpha=0.6)
axes[1, 0].scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='red', label='Class 1', alpha=0.6)
axes[1, 0].set_xlabel('X1')
axes[1, 0].set_ylabel('X2')
axes[1, 0].set_title('Test Data')
axes[1, 0].legend()
axes[1, 0].axis('equal')

# Plot 4: Decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid = np.column_stack([xx.ravel(), yy.ravel()])
grid_tensor = torch.from_numpy(grid.astype(np.float32)).to(device)

model.eval()
with torch.no_grad():
    Z = model(grid_tensor).cpu().numpy()
Z = Z.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
axes[1, 1].scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='blue', edgecolor='white', label='Class 0')
axes[1, 1].scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='red', edgecolor='white', label='Class 1')
axes[1, 1].set_xlabel('X1')
axes[1, 1].set_ylabel('X2')
axes[1, 1].set_title('Decision Boundary')
axes[1, 1].axis('equal')

plt.tight_layout()
plt.savefig('pytorch_circle_classifier.png', dpi=100)
print("Saved pytorch_circle_classifier.png")
plt.show()

# =================================
# 5. SAVE AND LOAD MODEL
# =================================

print("\n" + "=" * 60)
print("5. SAVING AND LOADING MODEL")
print("=" * 60)

# Save model
torch.save(model.state_dict(), 'circle_classifier.pth')
print("Model saved to circle_classifier.pth")

# Load model (example)
loaded_model = CircleClassifier().to(device)
loaded_model.load_state_dict(torch.load('circle_classifier.pth'))
loaded_model.eval()
print("Model loaded successfully!")

# Verify loaded model works
test_loss, test_acc = evaluate(loaded_model, test_loader, criterion, device)
print(f"Loaded model accuracy: {test_acc:.2%}")
