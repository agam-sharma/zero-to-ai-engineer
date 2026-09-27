# Source: AI_Learning_Cursor lines 7406-7882
# Original transcript phase: 2 - DEEP LEARNING WITH PYTORCH
# Nearest header: #### CODE: Modern Training Techniques
# Title: MODERN TRAINING TECHNIQUES
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MODERN TRAINING TECHNIQUES
==========================
Batch Normalization, optimizers, and more.
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

# =================================
# 1. BATCH NORMALIZATION
# =================================

print("=" * 60)
print("1. BATCH NORMALIZATION")
print("=" * 60)

"""
WHAT IS BATCH NORMALIZATION?

During training, each layer's input distribution changes as previous
layers' weights update. This is called "Internal Covariate Shift."

Batch Normalization:
1. Normalizes each batch to have mean=0, std=1
2. Adds learnable scale (gamma) and shift (beta)
3. Stabilizes training and allows higher learning rates

Formula:
    x_norm = (x - mean) / sqrt(var + epsilon)
    output = gamma * x_norm + beta
"""

class ModelWithoutBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)


class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),  # Normalize after linear, before activation
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)


# Compare the two models
print("Model without BatchNorm:")
model_no_bn = ModelWithoutBatchNorm()
print(f"  Parameters: {sum(p.numel() for p in model_no_bn.parameters()):,}")

print("\nModel with BatchNorm:")
model_bn = ModelWithBatchNorm()
print(f"  Parameters: {sum(p.numel() for p in model_bn.parameters()):,}")

# =================================
# 2. WEIGHT INITIALIZATION
# =================================

print("\n" + "=" * 60)
print("2. WEIGHT INITIALIZATION")
print("=" * 60)

"""
PROPER INITIALIZATION IS CRUCIAL

Poor initialization can lead to:
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Slow or no convergence

Common strategies:
- Xavier/Glorot: Good for tanh, sigmoid
- He/Kaiming: Good for ReLU
"""

def init_weights_xavier(m):
    """Xavier initialization for sigmoid/tanh activations."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_kaiming(m):
    """Kaiming/He initialization for ReLU activations."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Apply initialization
model_with_init = ModelWithBatchNorm()
model_with_init.apply(init_weights_kaiming)
print("Applied Kaiming initialization to model")

# =================================
# 3. OPTIMIZER COMPARISON
# =================================

print("\n" + "=" * 60)
print("3. OPTIMIZER COMPARISON")
print("=" * 60)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


def train_with_optimizer(optimizer_name, model_class, lr, n_epochs=10):
    """Train a model with a specific optimizer."""
    model = model_class().to(device)
    model.apply(init_weights_kaiming)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = outputs.max(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        test_accs.append(correct / total)
    
    return train_losses, test_accs


# Compare optimizers
print("Comparing optimizers (10 epochs each)...")

results = {}
optimizers_to_test = [
    ('SGD', 0.01),
    ('Adam', 0.001),
    ('AdamW', 0.001)
]

for opt_name, lr in optimizers_to_test:
    print(f"  Training with {opt_name}...")
    losses, accs = train_with_optimizer(opt_name, ModelWithBatchNorm, lr)
    results[opt_name] = {'losses': losses, 'accs': accs}
    print(f"    Final accuracy: {accs[-1]:.2%}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for opt_name in results:
    axes[0].plot(results[opt_name]['losses'], label=opt_name)
    axes[1].plot(results[opt_name]['accs'], label=opt_name)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Optimizer Comparison')
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=100)
print("Saved optimizer_comparison.png")
plt.show()

# =================================
# 4. LEARNING RATE SCHEDULES
# =================================

print("\n" + "=" * 60)
print("4. LEARNING RATE SCHEDULES")
print("=" * 60)

"""
LEARNING RATE SCHEDULES

Starting with a high LR and decreasing over time often works better:
- High LR at start: Explore the loss landscape quickly
- Low LR at end: Fine-tune to find the minimum

Common schedules:
- StepLR: Multiply by gamma every N epochs
- CosineAnnealing: Smooth cosine decay
- OneCycleLR: Increase then decrease (super effective!)
- ReduceLROnPlateau: Reduce when metric stops improving
"""

# Visualize different schedules
model = ModelWithBatchNorm().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create different schedulers
schedulers = {
    'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
    'CosineAnnealing': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),
    'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
}

# Plot LR over epochs
plt.figure(figsize=(10, 6))

for name, scheduler in schedulers.items():
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    if name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif name == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    lrs = []
    for epoch in range(50):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.plot(lrs, label=name)

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedules.png', dpi=100)
print("Saved lr_schedules.png")
plt.show()

# =================================
# 5. GRADIENT CLIPPING
# =================================

print("\n" + "=" * 60)
print("5. GRADIENT CLIPPING")
print("=" * 60)

"""
GRADIENT CLIPPING

Prevents exploding gradients by limiting their magnitude.
Essential for RNNs and Transformers!

Two types:
- clip_grad_norm_: Limit total gradient norm
- clip_grad_value_: Limit each gradient element
"""

# Example training loop with gradient clipping
def train_with_gradient_clipping(max_norm=1.0):
    model = ModelWithBatchNorm().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        total_norm = 0
        count = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            total_norm += grad_norm.item()
            count += 1
            
            optimizer.step()
        
        avg_norm = total_norm / count
        print(f"  Epoch {epoch+1}: Average gradient norm = {avg_norm:.4f}")
    
    return model


print("Training with gradient clipping (max_norm=1.0)...")
model = train_with_gradient_clipping(max_norm=1.0)

# =================================
# 6. COMPLETE BEST PRACTICES MODEL
# =================================

print("\n" + "=" * 60)
print("6. COMPLETE BEST PRACTICES MODEL")
print("=" * 60)

class BestPracticesCNN(nn.Module):
    """
    A CNN that uses all modern best practices:
    - Batch Normalization
    - Proper initialization (Kaiming)
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        
        # Apply proper initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Train the best practices model
model = BestPracticesCNN().to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("\nTraining best practices model...")
for epoch in range(20):
    model.train()
    train_loss = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    lr = optimizer.param_groups[0]['lr']
    
    if epoch % 5 == 0 or epoch == 19:
        print(f"Epoch {epoch+1:2d}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.2%}, LR={lr:.6f}")

print("\nTraining complete!")

# Save model
torch.save(model.state_dict(), 'best_practices_cnn.pth')
print("Model saved to best_practices_cnn.pth")
