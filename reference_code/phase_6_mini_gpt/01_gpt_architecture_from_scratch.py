# Source: AI_Learning_Cursor lines 22810-23252
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: GPT Architecture
# Title: GPT ARCHITECTURE FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
GPT ARCHITECTURE FROM SCRATCH
==============================
Building the complete GPT model - decoder-only transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. GPT VS BERT VS TRANSFORMER
# =================================

print("=" * 60)
print("1. GPT VS BERT VS TRANSFORMER")
print("=" * 60)

"""
ARCHITECTURAL COMPARISON:

Original Transformer (2017):
    - Encoder-Decoder architecture
    - For translation: "Hello" -> [Encoder] -> [Decoder] -> "Bonjour"
    - Encoder: Bidirectional (sees all tokens)
    - Decoder: Unidirectional (causal masking)

BERT (2018):
    - Encoder-only
    - Bidirectional (sees all tokens at once)
    - Training: Masked Language Modeling (fill in the blanks)
    - Good for: Classification, NER, QA
    - Cannot generate text naturally

GPT (2018+):
    - Decoder-only
    - Unidirectional (causal masking - only sees past)
    - Training: Next token prediction
    - Good for: Text generation, few-shot learning
    - Powers ChatGPT, Claude, etc.

WHY DECODER-ONLY FOR GENERATION?

Text generation is inherently left-to-right:
    "The cat sat on the" -> predict "mat"
    
We can only use PAST tokens to predict the NEXT token.
Causal masking enforces this constraint.

GPT Architecture:
    Input tokens -> Token Embedding + Position Embedding
                 -> N x Transformer Decoder Blocks
                 -> Layer Norm
                 -> Output Projection (to vocabulary)
                 -> Softmax -> Next token probabilities
"""

# =================================
# 2. GPT COMPONENTS
# =================================

print("\n" + "=" * 60)
print("2. GPT COMPONENTS")
print("=" * 60)

class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for GPT.
    
    Each position can only attend to previous positions.
    This is what makes GPT "autoregressive".
    """
    
    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Combined Q, K, V projection (more efficient)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()  # Batch, Sequence Length, Embedding Dim
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """
    Feed-forward network in GPT.
    
    Uses GELU activation (smoother than ReLU, used in GPT-2/3).
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """
    Single GPT block (decoder layer).
    
    Architecture:
        x -> LayerNorm -> Attention -> + (residual)
          -> LayerNorm -> MLP -> + (residual)
    
    Note: GPT-2 uses "Pre-LN" (LayerNorm before attention/MLP)
    Original Transformer used "Post-LN" (after)
    Pre-LN is more stable for deep networks.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


# =================================
# 3. COMPLETE GPT MODEL
# =================================

print("\n" + "=" * 60)
print("3. COMPLETE GPT MODEL")
print("=" * 60)

class GPTConfig:
    """Configuration for GPT model."""
    
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size  # Maximum sequence length
        self.n_layer = n_layer        # Number of transformer blocks
        self.n_head = n_head          # Number of attention heads
        self.n_embd = n_embd          # Embedding dimension
        self.dropout = dropout


class GPT(nn.Module):
    """
    The complete GPT model.
    
    This is the same architecture used in GPT-2!
    (Just with different hyperparameters)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # Position embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output projection (language model head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between input embeddings and output projection
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Token indices (batch, seq_len)
            targets: Target token indices for loss computation (optional)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
        
        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        
        # Embeddings
        tok_emb = self.wte(idx)  # Token embeddings (B, T, n_embd)
        pos_emb = self.wpe(pos)  # Position embeddings (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            
        Returns:
            idx: Extended sequence with generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# =================================
# 4. TEST GPT MODEL
# =================================

print("\n" + "=" * 60)
print("4. TESTING GPT MODEL")
print("=" * 60)

# Create a small GPT for testing
config = GPTConfig(
    vocab_size=1000,   # Small vocabulary for testing
    block_size=128,    # Max sequence length
    n_layer=4,         # 4 transformer layers
    n_head=4,          # 4 attention heads
    n_embd=128,        # 128-dimensional embeddings
    dropout=0.1,
)

model = GPT(config).to(device)

# Test forward pass
batch_size = 2
seq_len = 32
x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

logits, loss = model(x, targets)

print(f"\nForward pass test:")
print(f"  Input shape: {x.shape}")
print(f"  Logits shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")

# Test generation
print(f"\nGeneration test:")
start_tokens = torch.randint(0, config.vocab_size, (1, 5), device=device)
generated = model.generate(start_tokens, max_new_tokens=20, temperature=1.0, top_k=50)
print(f"  Start tokens: {start_tokens.squeeze().tolist()}")
print(f"  Generated: {generated.squeeze().tolist()}")

# =================================
# 5. GPT SIZE COMPARISON
# =================================

print("\n" + "=" * 60)
print("5. GPT SIZE COMPARISON")
print("=" * 60)

def count_parameters(config):
    """Estimate GPT parameters for a given config."""
    n_embd = config.n_embd
    n_layer = config.n_layer
    vocab_size = config.vocab_size
    block_size = config.block_size
    
    # Embeddings
    emb_params = vocab_size * n_embd + block_size * n_embd
    
    # Per layer
    attn_params = 4 * n_embd * n_embd  # Q, K, V, Output projections
    mlp_params = 8 * n_embd * n_embd   # 2 linear layers with 4x expansion
    ln_params = 4 * n_embd             # 2 LayerNorms per layer
    layer_params = attn_params + mlp_params + ln_params
    
    # Total
    total = emb_params + n_layer * layer_params + 2 * n_embd  # Final LN
    
    return total


configs = {
    "Our Mini-GPT": GPTConfig(vocab_size=10000, block_size=256, n_layer=6, n_head=6, n_embd=384),
    "GPT-2 Small": GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768),
    "GPT-2 Medium": GPTConfig(vocab_size=50257, block_size=1024, n_layer=24, n_head=16, n_embd=1024),
    "GPT-2 Large": GPTConfig(vocab_size=50257, block_size=1024, n_layer=36, n_head=20, n_embd=1280),
    "GPT-2 XL": GPTConfig(vocab_size=50257, block_size=1024, n_layer=48, n_head=25, n_embd=1600),
}

print("Model size comparison:")
for name, cfg in configs.items():
    params = count_parameters(cfg)
    print(f"  {name:20s}: {params/1e6:>8.1f}M parameters")

print("\nNote: GPT-3 has 175B parameters, GPT-4 has ~1T+ parameters")
print("We'll train a ~10-50M parameter model on your M3 Pro!")
