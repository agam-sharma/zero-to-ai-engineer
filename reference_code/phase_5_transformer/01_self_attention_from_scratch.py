# Source: AI_Learning_Cursor lines 17185-17565
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Understanding Self-Attention
# Title: SELF-ATTENTION FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
SELF-ATTENTION FROM SCRATCH
============================
The core mechanism that powers GPT, BERT, and all modern AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. THE QUERY-KEY-VALUE PARADIGM
# =================================

print("=" * 60)
print("1. THE QUERY-KEY-VALUE PARADIGM")
print("=" * 60)

"""
SELF-ATTENTION: Every token attends to every other token

The key insight: Think of it like a DATABASE LOOKUP

QUERY (Q): "What am I looking for?"
KEY (K):   "What do I contain?"
VALUE (V): "What information do I provide?"

For each token:
1. Create a Query: "I'm looking for relevant context"
2. Compare Query to all Keys: "How relevant is each token?"
3. Use scores to weight Values: "Combine relevant information"

Example sentence: "The cat sat on the mat"

When processing "sat":
- Query for "sat": "I need to know WHO sat and WHERE"
- Keys from all tokens tell what they offer
- "cat" key matches well (WHO)
- "mat" key matches well (WHERE)
- Values from "cat" and "mat" flow into "sat"'s representation

This is SELF-attention because Q, K, V all come from the SAME sequence.
(Cross-attention uses K, V from a different sequence - like encoder-decoder)
"""

# =================================
# 2. SCALED DOT-PRODUCT ATTENTION
# =================================

print("\n" + "=" * 60)
print("2. SCALED DOT-PRODUCT ATTENTION")
print("=" * 60)

"""
FORMULA:

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
- Q: Queries  (seq_len, d_k)
- K: Keys     (seq_len, d_k)  
- V: Values   (seq_len, d_v)
- d_k: Key dimension
- sqrt(d_k): Scaling factor to prevent softmax saturation

Step by step:
1. Q @ K^T         -> Attention scores (seq_len, seq_len)
2. / sqrt(d_k)     -> Scale to prevent large values
3. softmax(...)    -> Convert to probabilities (rows sum to 1)
4. @ V             -> Weighted combination of values
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Queries (batch, seq_len, d_k)
        K: Keys (batch, seq_len, d_k)
        V: Values (batch, seq_len, d_v)
        mask: Optional mask (batch, seq_len, seq_len)
        
    Returns:
        output: Attention output (batch, seq_len, d_v)
        attention_weights: Attention weights (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute attention scores
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        # Mask positions with -infinity so softmax gives 0
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# Test with simple example
print("Testing scaled dot-product attention...")

batch_size = 1
seq_len = 4
d_k = 8  # Key/Query dimension
d_v = 8  # Value dimension

# Random Q, K, V (in practice, these come from linear projections)
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights (each row sums to 1):")
print(weights[0].detach().numpy().round(3))
print(f"Row sums: {weights[0].sum(dim=-1).detach().numpy()}")

# =================================
# 3. VISUALIZING ATTENTION
# =================================

print("\n" + "=" * 60)
print("3. VISUALIZING ATTENTION")
print("=" * 60)

def visualize_attention_weights(weights, tokens, title="Attention Weights"):
    """Visualize attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    weights_np = weights.squeeze().detach().numpy()
    
    im = ax.imshow(weights_np, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attention Weight', rotation=-90, va='bottom')
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{weights_np[i, j]:.2f}',
                          ha='center', va='center', color='black' if weights_np[i, j] < 0.5 else 'white')
    
    ax.set_title(title)
    ax.set_xlabel('Key (attending to)')
    ax.set_ylabel('Query (from)')
    
    plt.tight_layout()
    plt.savefig('attention_weights_basic.png', dpi=100)
    print(f"Saved attention_weights_basic.png")
    plt.show()


# Create a more meaningful example
tokens = ['The', 'cat', 'sat', 'mat']

# Simulate learned attention (in practice, this is learned)
# Let's create attention that makes sense:
# - "sat" should attend to "cat" (subject)
# - "mat" might attend to "sat" (verb)
simulated_weights = torch.tensor([
    [0.7, 0.1, 0.1, 0.1],  # "The" mostly attends to itself
    [0.2, 0.5, 0.2, 0.1],  # "cat" attends to itself and context
    [0.1, 0.5, 0.3, 0.1],  # "sat" attends strongly to "cat"
    [0.1, 0.2, 0.4, 0.3],  # "mat" attends to "sat"
]).unsqueeze(0)

visualize_attention_weights(simulated_weights, tokens, "Self-Attention: 'The cat sat mat'")

# =================================
# 4. CREATING Q, K, V FROM EMBEDDINGS
# =================================

print("\n" + "=" * 60)
print("4. CREATING Q, K, V FROM EMBEDDINGS")
print("=" * 60)

"""
In practice, Q, K, V are LINEAR PROJECTIONS of the input embeddings.

Input: X (seq_len, d_model)

Q = X @ W_Q    where W_Q is (d_model, d_k)
K = X @ W_K    where W_K is (d_model, d_k)
V = X @ W_V    where W_V is (d_model, d_v)

These projection matrices are LEARNED during training.
They transform the input into different "views" for querying, being queried, and providing values.
"""

class SelfAttention(nn.Module):
    """
    Self-Attention layer with learned Q, K, V projections.
    """
    
    def __init__(self, d_model, d_k=None, d_v=None):
        """
        Args:
            d_model: Input embedding dimension
            d_k: Key/Query dimension (defaults to d_model)
            d_v: Value dimension (defaults to d_model)
        """
        super().__init__()
        
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        
        # Learnable projection matrices
        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attention output (batch, seq_len, d_v)
            attention_weights: Attention weights
        """
        # Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Apply scaled dot-product attention
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        return output, weights


# Test SelfAttention module
d_model = 64
seq_len = 6
batch_size = 2

attention = SelfAttention(d_model)
x = torch.randn(batch_size, seq_len, d_model)

output, weights = attention(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
print(f"\nParameters:")
print(f"  W_Q: {attention.W_Q.weight.shape}")
print(f"  W_K: {attention.W_K.weight.shape}")
print(f"  W_V: {attention.W_V.weight.shape}")
print(f"  Total: {sum(p.numel() for p in attention.parameters()):,}")

# =================================
# 5. ATTENTION MASKS
# =================================

print("\n" + "=" * 60)
print("5. ATTENTION MASKS")
print("=" * 60)

"""
ATTENTION MASKS: Control which positions can attend to which

Two main types:

1. PADDING MASK
   - Ignore padding tokens in batched sequences
   - Applied to both encoder and decoder

2. CAUSAL (LOOK-AHEAD) MASK
   - Prevent attending to future tokens
   - Used in decoder for autoregressive generation
   - Critical for GPT-style models!

Example causal mask for seq_len=4:
    [1, 0, 0, 0]   <- Position 0 can only see position 0
    [1, 1, 0, 0]   <- Position 1 can see positions 0, 1
    [1, 1, 1, 0]   <- Position 2 can see positions 0, 1, 2
    [1, 1, 1, 1]   <- Position 3 can see all positions
"""

def create_causal_mask(seq_len):
    """
    Create causal mask for decoder self-attention.
    
    Returns lower triangular matrix:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask.
    
    Args:
        seq: Token indices (batch, seq_len)
        pad_idx: Index of padding token
        
    Returns:
        mask: (batch, 1, seq_len) - broadcastable
    """
    mask = (seq != pad_idx).unsqueeze(1)
    return mask


# Visualize causal mask
seq_len = 6
causal_mask = create_causal_mask(seq_len)

plt.figure(figsize=(6, 5))
plt.imshow(causal_mask.numpy(), cmap='Blues')
plt.colorbar(label='Can Attend (1) / Cannot Attend (0)')

for i in range(seq_len):
    for j in range(seq_len):
        plt.text(j, i, int(causal_mask[i, j].item()),
                ha='center', va='center', 
                color='white' if causal_mask[i, j] > 0.5 else 'black')

plt.xlabel('Key Position (attending to)')
plt.ylabel('Query Position (from)')
plt.title('Causal Mask for Autoregressive Generation')
plt.tight_layout()
plt.savefig('causal_mask.png', dpi=100)
print("Saved causal_mask.png")
plt.show()

# Test masked attention
print("\nTesting causal masked attention...")

x = torch.randn(1, seq_len, d_model)
attention = SelfAttention(d_model)

# Without mask (can see everything)
output_no_mask, weights_no_mask = attention(x)
print(f"Without mask - weights[0,3,:] (pos 3 sees): {weights_no_mask[0, 3, :].detach().numpy().round(3)}")

# With causal mask
mask = create_causal_mask(seq_len).unsqueeze(0)
output_masked, weights_masked = attention(x, mask)
print(f"With causal mask - weights[0,3,:] (pos 3 sees): {weights_masked[0, 3, :].detach().numpy().round(3)}")

print("\nNote: With causal mask, position 3 cannot attend to positions 4, 5 (weight = 0)")
