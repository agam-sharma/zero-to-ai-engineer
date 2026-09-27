# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: NEURAL NETWORKS DEEP DIVE (Weeks 10–15)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
#
# ⚠️  PHASE RENUMBERING NOTE:
# This phase was previously "Phase 2" in an earlier version of the plan.
# Under the current 9-phase / 36-week structure it is Phase 3 (after
# Classical ML). The week numbers in the body (5.1, 5.2, …, 9.1) are
# NOT calendar weeks — they are preserved topic IDs for git-history
# stability. Map them to the NEW calendar weeks via the table below.
# ═══════════════════════════════════════════════════════════════════════
#
# TOPIC-ID  →  CALENDAR-WEEK MAPPING (NEW)
#   "Week 5" topics  → Calendar Week 10
#   "Week 6" topics  → Calendar Weeks 11–12 (MLP + BatchNorm)
#   "Week 7" topics  → Calendar Week 13 (CNNs + diagnostics)
#   "Week 8" topics  → Calendar Week 14 (hyperparam tuning + W&B)
#   "Week 9" topics  → Calendar Week 15 (CIFAR-10 capstone)
# See 01_ZERO_TO_GPT_6_Month_Masterclass.md for the full calendar.
# ═══════════════════════════════════════════════════════════════════════

---

## 🔗 The Salesforce Analogy

> **Neural Networks are to AI what Lightning Web Components are to Salesforce.**
>
> In Salesforce, you moved from Visualforce (manual, page-by-page, fragile) to Lightning Web Components (composable, reactive, scalable). That transition required you to learn new primitives: properties, decorators, event handling, lifecycle hooks.
>
> This phase is that exact same transition for AI. You already know the primitives (vectors, matrices, derivatives, backprop from Phase 1). Now you'll compose them into real, working neural networks — layer by layer, activation by activation, optimizer by optimizer.
>
> And just like LWC has a standard set of base components (`lightning-button`, `lightning-datatable`), neural networks have standard building blocks: Linear layers, ReLU, Softmax, BatchNorm, Dropout. **Phase 2 teaches you all the building blocks.**

---

# WEEK 5: The Perceptron, Neurons & Activation Functions

## 📖 THEORY: From Biology to Math

### The Single Neuron (Perceptron)

A single neuron does exactly THREE things:

```
Step 1: WEIGHTED SUM  →  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Step 2: ACTIVATION    →  a = σ(z)
Step 3: OUTPUT        →  pass 'a' to the next layer
```

Where:
- **x** = inputs (the data)
- **w** = weights (learned during training)
- **b** = bias (a learnable offset)
- **σ** = activation function (introduces non-linearity)

In code:
```python
output = activation(dot(weights, inputs) + bias)
```

That's it. A "deep neural network" is just thousands of these stacked together.

### Why Activation Functions?

Without activations, stacking layers is useless:
```
Layer 1: y = W₁x + b₁
Layer 2: z = W₂y + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```
This is just **another** linear function! No matter how many layers you stack, it collapses to a single linear transformation.

Activation functions break this linearity, allowing networks to learn curved, complex decision boundaries.

---

## 💻 CODE: Activation Functions Visualized

```python
"""
Week 5: Activation Functions — The Complete Guide
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

x = np.linspace(-5, 5, 1000)

# ═══════════════════════════════════════════════
# THE BIG 5 ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════

activations = {
    'Sigmoid': {
        'fn': lambda x: 1 / (1 + np.exp(-x)),
        'formula': 'σ(x) = 1 / (1 + e^(-x))',
        'range': '(0, 1)',
        'used_in': 'Binary classification output, gates in LSTMs',
        'problem': 'Vanishing gradient for large |x|'
    },
    'Tanh': {
        'fn': lambda x: np.tanh(x),
        'formula': 'tanh(x) = (e^x - e^-x) / (e^x + e^-x)',
        'range': '(-1, 1)',
        'used_in': 'Hidden layers (legacy), RNNs',
        'problem': 'Vanishing gradient for large |x|'
    },
    'ReLU': {
        'fn': lambda x: np.maximum(0, x),
        'formula': 'ReLU(x) = max(0, x)',
        'range': '[0, ∞)',
        'used_in': 'Most hidden layers in modern networks',
        'problem': 'Dead neurons (output=0 for negative inputs)'
    },
    'GELU': {
        'fn': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
        'formula': 'GELU(x) = x · Φ(x)',
        'range': '≈(-0.17, ∞)',
        'used_in': '🔥 GPT, BERT, most Transformers',
        'problem': 'Slightly more expensive to compute'
    },
    'SiLU/Swish': {
        'fn': lambda x: x / (1 + np.exp(-x)),
        'formula': 'SiLU(x) = x · σ(x)',
        'range': '≈(-0.28, ∞)',
        'used_in': 'LLaMA, modern networks',
        'problem': 'Slightly more expensive'
    }
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, info) in enumerate(activations.items()):
    ax = axes[idx]
    y = info['fn'](x)
    ax.plot(x, y, linewidth=2, color='#2196F3')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(f"{name}\n{info['formula']}", fontsize=11)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"Range: {info['range']}\nUsed in: {info['used_in']}", 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved activation_functions.png")

# KEY TAKEAWAY:
print("""
╔═══════════════════════════════════════════════════════════╗
║  Which activation to use:                                  ║
║                                                            ║
║  Hidden layers:  ReLU (default) or GELU (for Transformers) ║
║  Output layer:   Depends on your task:                     ║
║    - Binary classification → Sigmoid                       ║
║    - Multi-class → Softmax                                 ║
║    - Regression → No activation (linear)                   ║
║    - Language model → Softmax over vocabulary               ║
║                                                            ║
║  GPT specifically uses GELU in all its feed-forward layers ║
╚═══════════════════════════════════════════════════════════╝
""")
```

---

# WEEK 6: Loss Functions, Optimizers & the Training Loop

## 📖 THEORY: The Training Loop — Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                 THE TRAINING LOOP                            │
│                                                              │
│  for each epoch:                                             │
│    for each batch:                                           │
│                                                              │
│      ┌──────────────┐                                        │
│      │ FORWARD PASS │  prediction = model(input)             │
│      └──────┬───────┘                                        │
│             ▼                                                │
│      ┌──────────────┐                                        │
│      │  COMPUTE LOSS │  loss = criterion(prediction, target) │
│      └──────┬───────┘                                        │
│             ▼                                                │
│      ┌──────────────┐                                        │
│      │ BACKWARD PASS │  loss.backward()                      │
│      └──────┬───────┘                                        │
│             ▼                                                │
│      ┌──────────────┐                                        │
│      │ UPDATE WEIGHTS│  optimizer.step()                     │
│      └──────┬───────┘                                        │
│             ▼                                                │
│      ┌──────────────┐                                        │
│      │ ZERO GRADIENTS│  optimizer.zero_grad()                │
│      └──────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Loss Functions

The loss function measures **how wrong** the model's prediction is. Lower loss = better model.

| Loss Function | When to Use | Formula |
|---------------|-------------|---------|
| **MSE (Mean Squared Error)** | Regression | $\frac{1}{n}\sum(y - \hat{y})^2$ |
| **Binary Cross-Entropy** | Binary classification | $-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ |
| **Cross-Entropy** | Multi-class / Language models | $-\sum y_i \log(\hat{y}_i)$ |

**For GPT:** Cross-entropy loss. The model predicts a probability distribution over the vocabulary, and cross-entropy measures how far that distribution is from the correct next token.

### Optimizers

The optimizer decides **how** to update weights. Think of it as the strategy for walking downhill.

| Optimizer | Idea | Used In |
|-----------|------|---------|
| **SGD** | Simple: step in gradient direction | Classical |
| **SGD + Momentum** | Add "inertia" — keep moving in the same direction | Better than SGD |
| **Adam** | Adaptive learning rates per parameter + momentum | Most common default |
| **AdamW** | Adam with weight decay (regularization) | 🔥 GPT and all modern LLMs |

---

## 💻 CODE: The Complete PyTorch Training Loop

```python
"""
Week 6: Your First PyTorch Training Loop (on M3 Pro MPS!)
We'll classify MNIST digits — the "Hello World" of neural networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ═══════════════════════════════════════════════
# 1. DEVICE SETUP — Use your M3 Pro's GPU!
# ═══════════════════════════════════════════════

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════

transform = transforms.Compose([
    transforms.ToTensor(),                     # Convert to tensor [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Input shape: {train_data[0][0].shape}")  # (1, 28, 28)


# ═══════════════════════════════════════════════
# 3. MODEL DEFINITION
# ═══════════════════════════════════════════════

class MNISTClassifier(nn.Module):
    """
    A simple MLP for MNIST digit classification.
    
    Architecture: 784 → 256 → 128 → 10
    
    Salesforce analogy: Think of each layer as a step in a Flow.
    Each step transforms the data, extracting increasingly abstract features.
    Layer 1: raw pixels → edges
    Layer 2: edges → shapes  
    Layer 3: shapes → digit identity (0-9)
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),          # (1, 28, 28) → (784,)
            nn.Linear(784, 256),   # Weight matrix: 784×256
            nn.ReLU(),             # Non-linearity
            nn.Dropout(0.2),       # Regularization: randomly zero 20% of neurons
            nn.Linear(256, 128),   # Weight matrix: 256×128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),    # Weight matrix: 128×10 (10 digit classes)
            # No activation — CrossEntropyLoss includes softmax
        )
    
    def forward(self, x):
        return self.network(x)


model = MNISTClassifier().to(device)
print(f"\nModel architecture:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ═══════════════════════════════════════════════
# 4. LOSS & OPTIMIZER
# ═══════════════════════════════════════════════

criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ═══════════════════════════════════════════════
# 5. TRAINING LOOP — THE CORE PATTERN
# ═══════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Enable dropout, batchnorm training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # THE FOUR SACRED STEPS:
        optimizer.zero_grad()          # 1. Zero gradients from previous step
        output = model(data)           # 2. Forward pass
        loss = criterion(output, target)  # 3. Compute loss
        loss.backward()                # 4a. Backward pass (compute gradients)
        optimizer.step()               # 4b. Update weights
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()  # Disable dropout, use batchnorm running stats
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


# ═══════════════════════════════════════════════
# 6. TRAIN!
# ═══════════════════════════════════════════════

print("\n" + "=" * 60)
print("🧠 Training MNIST Classifier on M3 Pro MPS")
print("=" * 60)

epochs = 10
for epoch in range(1, epochs + 1):
    start = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    elapsed = time.time() - start
    print(f"Epoch {epoch:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
          f"Time: {elapsed:.1f}s")

print(f"\n🎉 Final Test Accuracy: {test_acc:.2f}%")
print("(Should be ~97-98% — not bad for a simple MLP!)")
```

---

# WEEK 7: MLPs, Initialization & BatchNorm

## 📖 THEORY: Why Networks Fail and How to Fix Them

### The Initialization Problem

When you create a neural network, the weights start as random numbers. **How** you generate those random numbers matters enormously:

```
Too small → signals shrink to zero as they pass through layers → "vanishing gradients"
Too large → signals explode to infinity → "exploding gradients"
Just right → signals maintain a healthy magnitude → stable training
```

### Key Initialization Strategies

| Strategy | Formula | Used With |
|----------|---------|-----------|
| **Xavier/Glorot** | $W \sim \mathcal{U}(-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})})$ | Sigmoid, Tanh |
| **Kaiming/He** | $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$ | ReLU, GELU |

PyTorch uses appropriate defaults, but understanding them helps you debug.

### BatchNorm: The Stabilizer

```python
# Without BatchNorm: activations drift, training is unstable
# With BatchNorm: activations are normalized to mean=0, std=1 at each layer

class BatchNorm1d:
    """Simplified BatchNorm for understanding."""
    def forward(self, x):
        # During training:
        mean = x.mean(dim=0)  # Mean across the batch
        std = x.std(dim=0)    # Std across the batch
        x_norm = (x - mean) / (std + 1e-5)  # Normalize
        return self.gamma * x_norm + self.beta  # Scale and shift (learned)
```

**Why BatchNorm works:** It keeps the inputs to each layer in a "healthy" range, preventing the vanishing/exploding gradient problem.

---

## 💻 CODE: Karpathy's makemore MLP (Week 7 Core Exercise)

**Watch first:** [Karpathy: Building makemore Part 2](https://youtu.be/TCH_1BHY58I) (1h15m)

```python
"""
Week 7: MLP Character-Level Language Model (Karpathy's makemore)

This bridges the gap between simple classifiers and language models.
We're predicting the NEXT CHARACTER in a name — exactly like GPT
predicts the next TOKEN in text, just at a smaller scale.

Based on: "A Neural Probabilistic Language Model" (Bengio et al., 2003)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════
# 1. DATA: Load and Preprocess Names
# ═══════════════════════════════════════════════

# Download names.txt from Karpathy's makemore repo
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
urllib.request.urlretrieve(url, "names.txt")

words = open('names.txt', 'r').read().splitlines()
print(f"Loaded {len(words)} names")
print(f"First 10: {words[:10]}")

# Build character vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}  # char → index
stoi['.'] = 0  # Special: start/end token
itos = {i: s for s, i in stoi.items()}  # index → char
vocab_size = len(stoi)
print(f"Vocabulary: {vocab_size} characters")
print(f"Mapping: {stoi}")


# ═══════════════════════════════════════════════
# 2. BUILD DATASET: Context → Next Character
# ═══════════════════════════════════════════════

block_size = 3  # Context length: how many chars we look at to predict the next

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # Start with '...'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # Slide window
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# Split: 80% train, 10% val, 10% test
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xval, Yval = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"Train: {Xtr.shape}, Val: {Xval.shape}, Test: {Xte.shape}")


# ═══════════════════════════════════════════════
# 3. THE MLP MODEL
# ═══════════════════════════════════════════════

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
n_embd = 10       # Embedding dimension (each char → 10D vector)
n_hidden = 200    # Hidden layer size

g = torch.Generator().manual_seed(2147483647)

# Parameters
C = torch.randn((vocab_size, n_embd), generator=g, device=device)  # Embedding table
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g, device=device) * (5/3) / (n_embd * block_size)**0.5
b1 = torch.randn(n_hidden, generator=g, device=device) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g, device=device) * 0.01
b2 = torch.randn(vocab_size, generator=g, device=device) * 0

# BatchNorm parameters
bngain = torch.ones((1, n_hidden), device=device)
bnbias = torch.zeros((1, n_hidden), device=device)
bnmean_running = torch.zeros((1, n_hidden), device=device)
bnstd_running = torch.ones((1, n_hidden), device=device)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

print(f"Total parameters: {sum(p.nelement() for p in parameters):,}")


# ═══════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════

print(f"\nTraining on {device}...")
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # Minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix].to(device), Ytr[ix].to(device)
    
    # Forward pass
    emb = C[Xb]                          # (B, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # (B, block_size * n_embd) — concatenate
    
    # Hidden layer with BatchNorm
    hpreact = embcat @ W1 + b1           # Pre-activation
    
    # BatchNorm
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / (bnstdi + 1e-5) + bnbias
    
    # Update running stats for inference
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    
    h = torch.tanh(hpreact)              # Activation
    logits = h @ W2 + b2                 # Output layer
    loss = F.cross_entropy(logits, Yb)   # Loss
    
    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # Update — learning rate decay
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    lossi.append(loss.log10().item())
    
    if i % 50000 == 0:
        print(f"  Step {i:6d}: loss = {loss.item():.4f}")

print(f"  Final loss: {loss.item():.4f}")


# ═══════════════════════════════════════════════
# 5. GENERATE NAMES!
# ═══════════════════════════════════════════════

print("\n🎉 Generated Names:")
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context]).to(device)]
        embcat = emb.view(1, -1)
        hpreact = embcat @ W1 + b1
        hpreact = bngain * (hpreact - bnmean_running) / (bnstd_running + 1e-5) + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(f"  {''.join(out[:-1])}")

print("""
What you just built:
  - A character-level language model
  - It learned to generate plausible English names
  - The EXACT same architecture as GPT, just:
    • Characters instead of tokens
    • Fixed context window instead of attention
    • One hidden layer instead of many Transformer blocks
  
  In Phase 4, you'll replace this MLP with a Transformer.
""")
```

---

# WEEK 8: Convolutional Neural Networks (CNNs) & Computer Vision

## 📖 THEORY: Why CNNs?

For Week 5-7, we flattened images into 1D vectors (784 pixels). This throws away **spatial structure** — the fact that nearby pixels are related.

CNNs preserve spatial structure by sliding a small filter across the image:

```
Image (5×5):          Filter (3×3):        Output:
┌───────────────┐     ┌─────────┐          
│ 1  0  1  0  1 │     │ 1  0  1 │     The filter slides across
│ 0  1  0  1  0 │  ⊛  │ 0  1  0 │  =  the image, computing a
│ 1  0  1  0  1 │     │ 1  0  1 │     dot product at each position
│ 0  1  0  1  0 │     └─────────┘     
│ 1  0  1  0  1 │                     Result: a "feature map"
└───────────────┘                     that detects a pattern
```

**Key idea:** A filter learns to detect a specific pattern (edge, corner, texture). Multiple filters detect multiple patterns. Stack CNN layers → detect increasingly complex patterns.

```
Layer 1 filters detect: edges, corners
Layer 2 filters detect: textures, shapes (using edges from layer 1)
Layer 3 filters detect: objects, parts (using shapes from layer 2)
```

---

## 💻 CODE: CNN for CIFAR-10

```python
"""
Week 8: CNN on CIFAR-10 — Classify 10 Categories of Images
Running on M3 Pro MPS for GPU acceleration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══════════════════════════════════════════════
# DATA: CIFAR-10 — 60K 32×32 color images in 10 classes
# ═══════════════════════════════════════════════

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Data augmentation
    transforms.RandomHorizontalFlip(),         # Random flip
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ═══════════════════════════════════════════════
# MODEL: A proper CNN
# ═══════════════════════════════════════════════

class CIFAR10CNN(nn.Module):
    """
    Architecture:
    Conv → BN → ReLU → Conv → BN → ReLU → Pool →
    Conv → BN → ReLU → Conv → BN → ReLU → Pool →
    Flatten → Linear → ReLU → Dropout → Linear
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(3, 32, 3, padding=1),   # (3, 32, 32) → (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (32, 32, 32) → (32, 16, 16)
            nn.Dropout2d(0.25),
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, 3, padding=1),   # (32, 16, 16) → (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (64, 16, 16) → (64, 8, 8)
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (64, 8, 8) → (4096,)
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CIFAR10CNN().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training (abbreviated — use the same train_epoch/evaluate pattern from Week 6)
print(f"\nTraining on {device}...")
for epoch in range(1, 31):
    model.train()
    start = time.time()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
    
    scheduler.step()
    
    # Evaluate
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_correct += output.argmax(1).eq(target).sum().item()
            test_total += target.size(0)
    
    elapsed = time.time() - start
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d} | Train Acc: {100*correct/total:.1f}% | "
              f"Test Acc: {100*test_correct/test_total:.1f}% | {elapsed:.1f}s")

print("Target: ~85-90% test accuracy")
```

---

# WEEK 9: Experiment Tracking & Hyperparameter Tuning

## 📖 THEORY: Weights & Biases (W&B) — Your MLOps Dashboard

```
Salesforce analogy:
  Debug Logs → W&B Logs
  Salesforce Inspector → W&B Dashboard
  Setup > Monitoring → W&B Experiment Tracking
  
You would never deploy Salesforce code without checking debug logs.
Same here — never train a model without experiment tracking.
```

### Key Hyperparameters to Track

| Hyperparameter | What it Controls | Typical Range |
|----------------|-----------------|---------------|
| Learning rate | Step size for weight updates | 1e-5 to 1e-2 |
| Batch size | Samples per gradient update | 16 to 512 |
| Hidden size | Width of hidden layers | 64 to 4096 |
| Number of layers | Depth of network | 1 to 100+ |
| Dropout rate | Regularization strength | 0.0 to 0.5 |
| Weight decay | L2 regularization | 1e-6 to 1e-2 |

---

## 💻 CODE: W&B Integration

```python
"""
Week 9: Experiment Tracking with Weights & Biases
"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# ═══════════════════════════════════════════════
# W&B SETUP
# ═══════════════════════════════════════════════

# First time: run `wandb login` in terminal and paste your API key
# Get a free account at https://wandb.ai

def train_with_wandb(config=None):
    """Train with experiment tracking."""
    
    with wandb.init(config=config):
        config = wandb.config
        
        # Build model with config hyperparameters
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 10),
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with W&B logging
        for epoch in range(config.epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    val_correct += output.argmax(1).eq(target).sum().item()
                    val_total += target.size(0)
            
            # LOG TO W&B — this is the magic
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_loss": val_loss / len(test_loader),
                "val_accuracy": 100. * val_correct / val_total,
            })
        
        return 100. * val_correct / val_total

# ═══════════════════════════════════════════════
# HYPERPARAMETER SWEEP
# ═══════════════════════════════════════════════

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-2},
        'hidden_size': {'values': [128, 256, 512]},
        'dropout': {'min': 0.0, 'max': 0.5},
        'weight_decay': {'min': 1e-6, 'max': 1e-2},
        'epochs': {'value': 10},
    }
}

# Uncomment to run:
# sweep_id = wandb.sweep(sweep_config, project="mnist-sweep")
# wandb.agent(sweep_id, train_with_wandb, count=20)

print("""
After running the sweep, go to wandb.ai to see:
  - Interactive loss curves for all experiments
  - Parallel coordinates plot showing hyperparameter importance  
  - Best model configuration automatically identified
  
This is your MLOps foundation. In Phase 5, you'll use W&B
to track your GPT training runs.
""")
```

---

## 📚 PHASE 3 COMPLETE RESOURCE LIST

### 🎥 Videos (Watch in Order)

| # | Video | Duration | Week | Purpose |
|---|-------|----------|------|---------|
| 1 | [3B1B: What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | 19 min | Week 5 | Best visual introduction |
| 2 | [3B1B: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | 21 min | Week 5 | How training actually works |
| 3 | [StatQuest: Neural Networks Pt 1](https://www.youtube.com/watch?v=CqOfi41LfDw) | 19 min | Week 5 | Step-by-step neuron math |
| 4 | [StatQuest: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) | 14 min | Week 6 | The loss function GPT uses |
| 5 | [StatQuest: Adam Optimizer](https://www.youtube.com/watch?v=MD2fYip6QsQ) | 17 min | Week 6 | The optimizer GPT uses |
| 6 | **[Karpathy: makemore Part 2 (MLP)](https://youtu.be/TCH_1BHY58I)** | **1h15m** | **Week 7** | **Build the character MLP** |
| 7 | **[Karpathy: makemore Part 3 (Activations & BatchNorm)](https://youtu.be/P6sfmUTpUmc)** | **1h55m** | **Week 7** | **Initialization, BatchNorm, diagnostics** |
| 8 | **[Karpathy: makemore Part 4 (Becoming a Backprop Ninja)](https://youtu.be/q8SA3rM6ckI)** | **1h55m** | **Week 7** | **Manual backprop through entire MLP** |
| 9 | [Stanford CS231n: CNNs](https://www.youtube.com/watch?v=bNb2fEVKeEo) | 1h15m | Week 8 | Convolutional networks deep dive |
| 10 | [W&B: PyTorch Integration Tutorial](https://www.youtube.com/watch?v=G7GH0SeNBMA) | 20 min | Week 9 | Experiment tracking setup |

### 📖 Reading & Tutorials

| # | Resource | Week | Purpose |
|---|----------|------|---------|
| 1 | [PyTorch Official Tutorials: 60-Min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | Week 5-6 | Official PyTorch introduction |
| 2 | [PyTorch Official: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) | Week 8 | CIFAR-10 CNN tutorial |
| 3 | [Karpathy: "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) | Week 9 | Practical training advice from the best |
| 4 | [Understanding Batch Normalization](https://arxiv.org/abs/1502.03167) | Week 7 | Original BatchNorm paper |
| 5 | [An Overview of Optimization in Deep Learning](https://ruder.io/optimizing-gradient-descent/) | Week 6 | Comprehensive optimizer comparison |
| 6 | [W&B Documentation](https://docs.wandb.ai/) | Week 9 | Official experiment tracking docs |

---

## ✅ PHASE 3 COMPLETION CHECKLIST

- [ ] Can explain what a neuron computes: `activation(weights · input + bias)`
- [ ] Know when to use ReLU, GELU, Sigmoid, Softmax
- [ ] Can write the PyTorch training loop from memory (the 4 sacred steps)
- [ ] Trained MNIST MLP to >97% accuracy on M3 Pro MPS
- [ ] Built the makemore MLP character-level language model
- [ ] Understand BatchNorm: why it helps, what it does at train vs eval time
- [ ] Manually backpropagated through an MLP (Karpathy's "backprop ninja")
- [ ] Built and trained a CNN for CIFAR-10 to >85% accuracy
- [ ] Set up W&B experiment tracking
- [ ] Ran at least one hyperparameter sweep
- [ ] Read Karpathy's "Recipe for Training Neural Networks"
- [ ] Can diagnose common training issues: learning rate too high/low, overfitting, underfitting

**🧪 Phase 3 Project:** CIFAR-10 classifier achieving ≥85% test accuracy, tracked with W&B, plus the "loss-curve pathology zoo" (project #7) showing 6 distinct training failure modes.

**Next:** [Phase 4: NLP & Sequence Models (Weeks 16–20)](./Phase_4_NLP_and_Sequences.md)
