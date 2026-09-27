# Source: AI_Learning_Cursor lines 18965-19197
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Complete Encoder Implementation
# Title: TRANSFORMER ENCODER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TRANSFORMER ENCODER
===================
The encoding side of the Transformer (used in BERT, sentence encoders).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# =================================
# 1. TRANSFORMER ENCODER ARCHITECTURE
# =================================

print("=" * 60)
print("1. TRANSFORMER ENCODER")
print("=" * 60)

"""
TRANSFORMER ENCODER

Architecture:
1. Input Embedding
2. Positional Encoding
3. N x Encoder Blocks
   - Multi-Head Self-Attention
   - Add & Norm
   - Feed-Forward
   - Add & Norm

Output: Contextualized representations for each input token

Used in:
- BERT (bidirectional encoder)
- Sentence encoders
- Encoder side of machine translation
"""

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention."""
    
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
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_O(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderBlock(nn.Module):
    """Single Encoder Block."""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention (query, key, value are all the same)
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder.
    
    Includes:
    - Token Embedding
    - Positional Encoding
    - N Encoder Blocks
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices (batch, seq_len)
            mask: Padding mask (batch, seq_len)
            
        Returns:
            (batch, seq_len, d_model) encoded representations
        """
        batch_size, seq_len = x.shape
        
        # Create padding mask if not provided
        if mask is None:
            mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        
        # Token + Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# Test Encoder
vocab_size = 10000
d_model = 256
num_heads = 8
num_layers = 4

encoder = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)

# Create sample input
batch_size = 4
seq_len = 32
x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

output = encoder(x)

print(f"Encoder configuration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  num_layers: {num_layers}")
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
