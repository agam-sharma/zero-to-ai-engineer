# Source: AI_Learning_Cursor lines 18650-18923
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Combining Everything
# Title: COMPLETE ATTENTION LAYER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
COMPLETE ATTENTION LAYER
========================
Combining Multi-Head Attention with Positional Encoding
and adding Layer Normalization + Feed-Forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =================================
# 1. LAYER NORMALIZATION
# =================================

print("=" * 60)
print("1. LAYER NORMALIZATION")
print("=" * 60)

"""
LAYER NORMALIZATION

Normalizes across the feature dimension (not batch).
Critical for transformer training stability.

For each sample position:
    x_norm = (x - mean) / sqrt(var + eps)
    output = gamma * x_norm + beta

Where gamma and beta are learned parameters.

Different from BatchNorm:
- BatchNorm: Normalizes across batch (problematic for variable-length sequences)
- LayerNorm: Normalizes across features (works for any batch size)
"""

# PyTorch has LayerNorm built-in
layer_norm = nn.LayerNorm(64)

x = torch.randn(2, 10, 64)
x_norm = layer_norm(x)

print(f"Input mean: {x.mean(dim=-1)[0, :3]}")
print(f"Output mean: {x_norm.mean(dim=-1)[0, :3]}")  # Should be ~0
print(f"Input std: {x.std(dim=-1)[0, :3]}")
print(f"Output std: {x_norm.std(dim=-1)[0, :3]}")   # Should be ~1

# =================================
# 2. FEED-FORWARD NETWORK
# =================================

print("\n" + "=" * 60)
print("2. FEED-FORWARD NETWORK")
print("=" * 60)

"""
POSITION-WISE FEED-FORWARD NETWORK

After attention, we apply a feed-forward network INDEPENDENTLY to each position.

FFN(x) = max(0, x @ W1 + b1) @ W2 + b2

Or: Linear(d_model -> d_ff) -> ReLU -> Linear(d_ff -> d_model)

d_ff is typically 4 * d_model (e.g., d_model=512, d_ff=2048)

This adds:
- Non-linearity (ReLU)
- More parameters and expressivity
- Helps process information at each position
"""

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)         # (batch, seq_len, d_ff)
        x = F.gelu(x)               # GELU activation (used in GPT, BERT)
        x = self.dropout(x)
        x = self.linear2(x)         # (batch, seq_len, d_model)
        return x


# Test
d_model = 64
d_ff = 256

ff = FeedForward(d_model, d_ff)
x = torch.randn(2, 10, d_model)
output = ff(x)

print(f"FFN input shape: {x.shape}")
print(f"FFN output shape: {output.shape}")
print(f"FFN parameters: {sum(p.numel() for p in ff.parameters()):,}")

# =================================
# 3. COMPLETE TRANSFORMER LAYER
# =================================

print("\n" + "=" * 60)
print("3. COMPLETE TRANSFORMER LAYER")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (from earlier)."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_O(context)


class TransformerBlock(nn.Module):
    """
    A complete Transformer block (used in encoder or decoder).
    
    Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


# Test
d_model = 64
num_heads = 8
d_ff = 256
batch_size = 2
seq_len = 20

block = TransformerBlock(d_model, num_heads, d_ff).to(device)
x = torch.randn(batch_size, seq_len, d_model, device=device)

output = block(x)

print(f"TransformerBlock:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Parameters: {sum(p.numel() for p in block.parameters()):,}")

# With causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
output_masked = block(x, causal_mask)
print(f"  With causal mask: {output_masked.shape}")

# =================================
# 4. STACKING TRANSFORMER BLOCKS
# =================================

print("\n" + "=" * 60)
print("4. STACKING TRANSFORMER BLOCKS")
print("=" * 60)

"""
Real transformers stack multiple blocks:

GPT-2 Small:  12 layers
GPT-2 Medium: 24 layers
GPT-2 Large:  36 layers
GPT-2 XL:     48 layers
GPT-3:        96 layers
"""

class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks for encoding.
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Create a small encoder
num_layers = 6
encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff).to(device)

output = encoder(x)

print(f"TransformerEncoder with {num_layers} layers:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
