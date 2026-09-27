# Source: AI_Learning_Cursor lines 17573-17801
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Complete Attention Implementation with Visualization
# Title: ATTENTION MECHANICS DEEP DIVE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
ATTENTION MECHANICS DEEP DIVE
=============================
Understanding every component of attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =================================
# 1. STEP-BY-STEP ATTENTION WALKTHROUGH
# =================================

print("=" * 60)
print("1. STEP-BY-STEP ATTENTION")
print("=" * 60)

# Create a simple example we can trace manually
torch.manual_seed(42)

# Simple 3-token sequence with 4-dimensional embeddings
tokens = ['I', 'love', 'AI']
seq_len = 3
d_model = 4

# Input embeddings (pretend these are word embeddings)
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # "I"
    [0.0, 1.0, 0.0, 1.0],   # "love"
    [1.0, 1.0, 0.0, 0.0],   # "AI"
])

print("Input embeddings X:")
for i, (token, emb) in enumerate(zip(tokens, X)):
    print(f"  {token}: {emb.numpy()}")

# Simple projection matrices (normally learned)
W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.0, 0.0],
])

W_K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.5],
    [0.5, 0.0],
])

W_V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
])

d_k = 2  # Key dimension

# Step 1: Compute Q, K, V
Q = X @ W_Q  # (3, 4) @ (4, 2) = (3, 2)
K = X @ W_K
V = X @ W_V

print("\nStep 1: Linear projections")
print("Queries Q:")
for i, (token, q) in enumerate(zip(tokens, Q)):
    print(f"  {token}: {q.numpy()}")
    
print("Keys K:")
for i, (token, k) in enumerate(zip(tokens, K)):
    print(f"  {token}: {k.numpy()}")

print("Values V:")
for i, (token, v) in enumerate(zip(tokens, V)):
    print(f"  {token}: {v.numpy()}")

# Step 2: Compute attention scores
scores = Q @ K.T  # (3, 2) @ (2, 3) = (3, 3)
print(f"\nStep 2: Attention scores (Q @ K^T):")
print(scores.numpy())

# Step 3: Scale
scaled_scores = scores / np.sqrt(d_k)
print(f"\nStep 3: Scaled scores (/ sqrt(d_k={d_k})):")
print(scaled_scores.numpy())

# Step 4: Softmax
attention_weights = F.softmax(scaled_scores, dim=-1)
print(f"\nStep 4: Attention weights (softmax):")
print(attention_weights.numpy().round(4))

# Step 5: Weighted sum of values
output = attention_weights @ V  # (3, 3) @ (3, 2) = (3, 2)
print(f"\nStep 5: Output (weights @ V):")
for i, (token, out) in enumerate(zip(tokens, output)):
    print(f"  {token}: {out.numpy()}")

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
Each output token is now a WEIGHTED COMBINATION of all input values.

For example, the output for "love" is:
  weight['I'] * value['I'] + weight['love'] * value['love'] + weight['AI'] * value['AI']
  
The weights are determined by how well each query matches each key.
This is how self-attention builds context-aware representations!
""")

# =================================
# 2. WHY SCALING MATTERS
# =================================

print("\n" + "=" * 60)
print("2. WHY SCALING MATTERS")
print("=" * 60)

"""
Without scaling, dot products can become very large for high dimensions.
Large values -> softmax becomes too "sharp" (one position dominates)
-> Gradients become tiny (vanishing gradient)
"""

def show_softmax_scaling():
    """Demonstrate the effect of scaling on softmax."""
    d_k_values = [4, 64, 512, 2048]
    
    fig, axes = plt.subplots(1, len(d_k_values), figsize=(16, 4))
    
    for ax, d_k in zip(axes, d_k_values):
        # Random query and keys
        Q = torch.randn(1, d_k)
        K = torch.randn(5, d_k)
        
        # Unscaled scores
        scores_unscaled = (Q @ K.T).squeeze()
        weights_unscaled = F.softmax(scores_unscaled, dim=-1)
        
        # Scaled scores
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        weights_scaled = F.softmax(scores_scaled, dim=-1)
        
        x = np.arange(5)
        width = 0.35
        
        ax.bar(x - width/2, weights_unscaled.detach().numpy(), width, label='Unscaled', color='red', alpha=0.7)
        ax.bar(x + width/2, weights_scaled.detach().numpy(), width, label='Scaled', color='blue', alpha=0.7)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'd_k = {d_k}')
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.suptitle('Effect of Scaling on Attention Weights', fontsize=14)
    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=100)
    print("Saved scaling_effect.png")
    plt.show()
    
    print("Observation: Without scaling, higher d_k leads to sharper (more concentrated) attention.")
    print("Scaling keeps the distribution more balanced, allowing gradients to flow better.")


show_softmax_scaling()

# =================================
# 3. ATTENTION AS SOFT LOOKUP
# =================================

print("\n" + "=" * 60)
print("3. ATTENTION AS SOFT LOOKUP")
print("=" * 60)

"""
Think of attention as a "soft" dictionary lookup.

Hard lookup (regular dictionary):
    dict["cat"] -> returns value for "cat" exactly
    
Soft lookup (attention):
    query("fluffy animal") -> returns weighted combination of
        0.7 * dict["cat"] + 0.2 * dict["dog"] + 0.1 * dict["rabbit"]
        
The query doesn't need to match a key exactly.
It can match PARTIALLY with multiple keys and blend their values.
"""

# Demonstration: Semantic similarity
print("Semantic similarity example:")

# Pretend embeddings for words
word_embeddings = {
    'cat': torch.tensor([1.0, 0.8, 0.2, 0.0]),
    'dog': torch.tensor([0.9, 0.7, 0.3, 0.1]),
    'car': torch.tensor([0.1, 0.0, 0.9, 0.8]),
    'truck': torch.tensor([0.2, 0.1, 0.85, 0.75]),
}

# Query: "pet"
query_pet = torch.tensor([0.95, 0.75, 0.25, 0.05])

print("\nQuery: 'pet' (an animal-related concept)")
print("Computing similarity to each word...")

similarities = {}
for word, emb in word_embeddings.items():
    sim = torch.dot(query_pet, emb) / (query_pet.norm() * emb.norm())
    similarities[word] = sim.item()
    print(f"  {word}: {sim.item():.4f}")

# Apply softmax to get attention weights
sim_tensor = torch.tensor(list(similarities.values()))
attention = F.softmax(sim_tensor * 5, dim=-1)  # Temperature scaling for sharper distribution

print("\nAttention weights:")
for word, weight in zip(similarities.keys(), attention):
    print(f"  {word}: {weight.item():.4f}")

print("\nResult: 'pet' attends most to 'cat' and 'dog' (animals), not 'car' or 'truck'")
