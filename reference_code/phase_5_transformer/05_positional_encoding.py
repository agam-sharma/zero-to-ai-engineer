# Source: AI_Learning_Cursor lines 18393-18642
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Positional Encoding Implementation
# Title: POSITIONAL ENCODING
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
POSITIONAL ENCODING
===================
Adding position information to embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =================================
# 1. POSITIONAL ENCODING THEORY
# =================================

print("=" * 60)
print("1. WHY POSITIONAL ENCODING?")
print("=" * 60)

"""
PROBLEM: Self-attention doesn't know position!

For input "The cat sat on the mat":
- Attention sees a SET of token embeddings
- No information about which token comes first, second, etc.
- "The cat sat" and "sat cat The" would give the same attention!

SOLUTION: Add positional encoding to embeddings

Position Encoding adds a unique pattern to each position.
Token embedding + Position encoding = Position-aware embedding

THE SINUSOIDAL ENCODING (from original Transformer):

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: Position in sequence (0, 1, 2, ...)
- i: Dimension index
- d_model: Model dimension

Why sinusoids?
1. Unique pattern for each position
2. Bounded values (between -1 and 1)
3. Can extrapolate to longer sequences
4. Relative positions can be computed (PE(pos+k) is linear function of PE(pos))
"""

# =================================
# 2. SINUSOIDAL POSITIONAL ENCODING
# =================================

print("\n" + "=" * 60)
print("2. SINUSOIDAL POSITIONAL ENCODING")
print("=" * 60)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Register as buffer (not a parameter, but should be saved)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Test
d_model = 64
max_len = 100

pe_layer = SinusoidalPositionalEncoding(d_model, max_len)
x = torch.zeros(1, 50, d_model)  # Dummy input
x_with_pe = pe_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {x_with_pe.shape}")
print(f"PE shape: {pe_layer.pe.shape}")

# =================================
# 3. VISUALIZING POSITIONAL ENCODING
# =================================

print("\n" + "=" * 60)
print("3. VISUALIZING POSITIONAL ENCODING")
print("=" * 60)

def visualize_positional_encoding(d_model=128, max_len=100):
    """Visualize the sinusoidal positional encoding pattern."""
    pe_layer = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_layer.pe.squeeze(0).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full heatmap
    ax = axes[0, 0]
    im = ax.imshow(pe[:50, :64], cmap='RdBu', aspect='auto')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Position')
    ax.set_title('Positional Encoding Heatmap')
    plt.colorbar(im, ax=ax)
    
    # First few dimensions over positions
    ax = axes[0, 1]
    for dim in [0, 1, 2, 3, 4, 5]:
        ax.plot(pe[:50, dim], label=f'Dim {dim}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('First 6 Dimensions Over Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fixed position, all dimensions
    ax = axes[1, 0]
    for pos in [0, 10, 20, 30, 40]:
        ax.plot(pe[pos, :32], label=f'Pos {pos}')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title('First 32 Dimensions at Different Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Similarity between positions
    ax = axes[1, 1]
    # Compute cosine similarity between position pairs
    similarities = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            sim = np.dot(pe[i], pe[j]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[j]))
            similarities[i, j] = sim
    
    im = ax.imshow(similarities, cmap='RdBu')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Cosine Similarity Between Positions')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=100)
    print("Saved positional_encoding.png")
    plt.show()
    
    print("Observations:")
    print("- Each position has a unique pattern")
    print("- Lower dimensions change slowly (capture global position)")
    print("- Higher dimensions change quickly (capture fine position)")
    print("- Nearby positions have similar encodings")


visualize_positional_encoding()

# =================================
# 4. LEARNED POSITIONAL EMBEDDINGS
# =================================

print("\n" + "=" * 60)
print("4. LEARNED POSITIONAL EMBEDDINGS")
print("=" * 60)

"""
ALTERNATIVE: Learned Positional Embeddings

Instead of fixed sinusoids, we can LEARN positional embeddings.
This is what BERT and GPT use!

Pros:
- Can learn task-specific position patterns
- Simpler implementation

Cons:
- Cannot extrapolate to longer sequences than seen in training
- More parameters
"""

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (used in GPT, BERT).
    """
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_embedding = self.embedding(positions)  # (seq_len, d_model)
        
        x = x + pos_embedding.unsqueeze(0)  # Broadcast over batch
        return self.dropout(x)


# Compare
print("Comparing sinusoidal vs learned positional encoding:")

d_model = 64
max_len = 512

sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len)
learned_pe = LearnedPositionalEmbedding(d_model, max_len)

print(f"\nSinusoidal PE parameters: 0 (fixed)")
print(f"Learned PE parameters: {sum(p.numel() for p in learned_pe.parameters()):,}")
