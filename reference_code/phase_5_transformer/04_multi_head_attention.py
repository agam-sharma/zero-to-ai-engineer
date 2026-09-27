# Source: AI_Learning_Cursor lines 18146-18379
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Multi-Head Attention Implementation
# Title: MULTI-HEAD ATTENTION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MULTI-HEAD ATTENTION
====================
Running multiple attention heads in parallel for richer representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# =================================
# 1. MULTI-HEAD ATTENTION THEORY
# =================================

print("=" * 60)
print("1. MULTI-HEAD ATTENTION THEORY")
print("=" * 60)

"""
MULTI-HEAD ATTENTION

Instead of one attention with d_model dimensions:
    Single head: Q, K, V each (seq, d_model) -> attention(seq, d_model)

We use h heads, each with d_k = d_model / h dimensions:
    Head 1: Q1, K1, V1 each (seq, d_k) -> attention1(seq, d_k)
    Head 2: Q2, K2, V2 each (seq, d_k) -> attention2(seq, d_k)
    ...
    Head h: Qh, Kh, Vh each (seq, d_k) -> attentionh(seq, d_k)

Then CONCATENATE and project:
    Concat: (seq, h * d_k) = (seq, d_model)
    Output projection: (seq, d_model)

WHY MULTIPLE HEADS?
- Each head can learn different relationship types
- One head might learn syntax, another semantics
- One head might learn local patterns, another global
- More parameters, more expressivity

Example with d_model=512, h=8:
- Each head has d_k = 512/8 = 64 dimensions
- We run 8 parallel attention computations
- Total computation is similar to single head
"""

# =================================
# 2. MULTI-HEAD ATTENTION FROM SCRATCH
# =================================

print("\n" + "=" * 60)
print("2. MULTI-HEAD ATTENTION IMPLEMENTATION")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    This is THE core component of transformers.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections (we can do all heads at once!)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask
            return_attention: Return attention weights for visualization
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 2. Reshape for multi-head: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, d_k)
        
        # 3. Compute attention scores
        # (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 4. Apply mask
        if mask is not None:
            # Expand mask for num_heads: (batch, 1, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)
        
        # 7. Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 8. Output projection
        output = self.W_O(context)
        
        if return_attention:
            return output, attention_weights
        return output


# Test
d_model = 64
num_heads = 8
batch_size = 2
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads).to(device)
x = torch.randn(batch_size, seq_len, d_model, device=device)

output, weights = mha(x, return_attention=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"  - batch_size: {weights.shape[0]}")
print(f"  - num_heads: {weights.shape[1]}")
print(f"  - seq_len x seq_len: {weights.shape[2]} x {weights.shape[3]}")

print(f"\nModel parameters: {sum(p.numel() for p in mha.parameters()):,}")
print(f"  - W_Q: {d_model * d_model} = {d_model * d_model:,}")
print(f"  - W_K: {d_model * d_model} = {d_model * d_model:,}")
print(f"  - W_V: {d_model * d_model} = {d_model * d_model:,}")
print(f"  - W_O: {d_model * d_model} = {d_model * d_model:,}")

# =================================
# 3. VISUALIZING MULTIPLE HEADS
# =================================

print("\n" + "=" * 60)
print("3. VISUALIZING ATTENTION HEADS")
print("=" * 60)

def visualize_multi_head_attention(attention_weights, tokens, num_heads_to_show=4):
    """Visualize attention patterns from different heads."""
    weights = attention_weights[0].cpu().detach().numpy()  # First batch
    
    num_heads = min(num_heads_to_show, weights.shape[0])
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    
    for i in range(num_heads):
        ax = axes[i] if num_heads > 1 else axes
        im = ax.imshow(weights[i], cmap='Blues')
        
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    plt.suptitle('Different Attention Heads Learn Different Patterns')
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=100)
    print("Saved multi_head_attention.png")
    plt.show()


# Simulate with a sentence
tokens = ['The', 'quick', 'brown', 'fox', 'jumps']
seq_len = len(tokens)
d_model = 32
num_heads = 4

mha = MultiHeadAttention(d_model, num_heads).to(device)

# Create embeddings (normally from embedding layer)
x = torch.randn(1, seq_len, d_model, device=device)

output, weights = mha(x, return_attention=True)
visualize_multi_head_attention(weights, tokens, num_heads_to_show=4)

print("\nNote: Each head learns different attention patterns!")
print("In a trained model, you might see:")
print("- Head 1: Attends to adjacent words (local patterns)")
print("- Head 2: Attends to sentence start (global context)")
print("- Head 3: Attends to semantically similar words")
print("- Head 4: Attends to syntactically related words")
