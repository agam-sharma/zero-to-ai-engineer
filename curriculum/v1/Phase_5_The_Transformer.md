# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: THE TRANSFORMER (Weeks 21–25)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
#
# ⚠️  PHASE RENUMBERING NOTE:
# Previously "Phase 4" in an earlier version. Under the current
# 9-phase / 36-week structure this is Phase 5. In-body topic IDs
# "Week 15.x" through "Week 19.x" are preserved for git-history
# stability. Map to NEW calendar weeks as follows:
#   "Week 15" topics → Calendar Week 21 (self-attention + multi-head)
#   "Week 16" topics → Calendar Week 22 (pos encoding, LayerNorm, FFN)
#   "Week 17" topics → Calendar Week 23 (full Transformer block + decoder)
#   "Week 18" topics → Calendar Week 24 (training loop + LR sched)
#   "Week 19" topics → Calendar Week 25 (paper + mech-interp + capstone)
# The Week 25 topic (NEW) adds a mechanistic-interpretability primer
# with TransformerLens; see the master plan.
# See 01_ZERO_TO_GPT_6_Month_Masterclass.md for the full calendar.
# ═══════════════════════════════════════════════════════════════════════

---

## 🔗 The Salesforce Analogy

> **Building GPT is like building your own Salesforce Platform from scratch.**
>
> In Phases 1-3, you built the individual components: the database (vectors/matrices), the triggers (backprop), the UI components (layers), the Flow engine (attention). Now you assemble everything into a complete, working platform.
>
> Just as Salesforce has a specific architecture — multi-tenant, metadata-driven, API-first — GPT has a specific architecture: decoder-only Transformer, autoregressive, next-token prediction. This phase teaches you that architecture, top to bottom.

---

# WEEK 15: The GPT Architecture (Paper Reading + Design)

## 📖 THEORY: GPT's Complete Architecture

GPT is simpler than you think. Here's the ENTIRE model:

```
╔═════════════════════════════════════════════════════╗
║                   GPT ARCHITECTURE                    ║
╠═════════════════════════════════════════════════════╣
║                                                       ║
║  Input tokens:  [The] [cat] [sat] [on]                ║
║       ↓                                               ║
║  Token Embedding:    nn.Embedding(vocab_size, d_model) ║
║  + Position Embedding: nn.Embedding(max_seq_len, d_model)║
║       ↓                                               ║
║  ┌──────────────────────────────────────────────┐     ║
║  │ Transformer Block 1                           │     ║
║  │   LayerNorm → Multi-Head Attention → Residual │     ║
║  │   LayerNorm → Feed-Forward Network → Residual │     ║
║  └──────────────────────────────────────────────┘     ║
║       ↓                                               ║
║  ┌──────────────────────────────────────────────┐     ║
║  │ Transformer Block 2                           │     ║
║  │   (same structure)                            │     ║
║  └──────────────────────────────────────────────┘     ║
║       ↓                                               ║
║  ... (repeat N times) ...                             ║
║       ↓                                               ║
║  ┌──────────────────────────────────────────────┐     ║
║  │ Transformer Block N                           │     ║
║  └──────────────────────────────────────────────┘     ║
║       ↓                                               ║
║  Final LayerNorm                                      ║
║       ↓                                               ║
║  Linear (d_model → vocab_size)   ← "LM Head"         ║
║       ↓                                               ║
║  Output logits: [0.1, 0.3, ..., 0.02, 8.5, ...]      ║
║                                  ↑                    ║
║                            "the" has highest logit    ║
║                                                       ║
║  → softmax → next token: "the"                        ║
╚═════════════════════════════════════════════════════╝
```

### GPT-2 Model Sizes

| Model | Layers | d_model | Heads | d_ff | Parameters |
|-------|--------|---------|-------|------|------------|
| GPT-2 Small | 12 | 768 | 12 | 3072 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 4096 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 5120 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 1.5B |

For your M3 Pro (18GB unified memory), you can comfortably train a model with:
- **~50M parameters** from scratch
- **~120M parameters** with gradient accumulation and reduced batch size

### The Training Objective

GPT's training is remarkably simple: **predict the next token.**

```
Input:  "The cat sat on"
Target: "cat sat on the"

The model sees "The" and tries to predict "cat"
The model sees "The cat" and tries to predict "sat"  
The model sees "The cat sat" and tries to predict "on"
The model sees "The cat sat on" and tries to predict "the"
```

Loss = Cross-Entropy between predicted probabilities and actual next token.

That's it. No labels needed. No human annotation. Just lots and lots of text.

---

## 💻 CODE: GPT from Scratch — The Complete Model

**Watch first:** [Karpathy: Let's Build GPT from Scratch](https://youtu.be/kCc8FmEb1nY) (1h56m)

```python
"""
Week 15-16: Building GPT from Scratch

Based on Karpathy's "Let's Build GPT" video and nanoGPT.
This is THE culmination of everything you've learned.

Every line of code maps to a concept from Phases 1-3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════

class GPTConfig:
    """Configuration for GPT model."""
    # Model architecture
    vocab_size: int = 50257      # GPT-2 vocabulary size
    block_size: int = 256        # Maximum sequence length (context window)
    n_layer: int = 6             # Number of Transformer blocks
    n_head: int = 6              # Number of attention heads
    n_embd: int = 384            # Embedding dimension (d_model)
    dropout: float = 0.2         # Dropout rate
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    
    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'


config = GPTConfig()


# ═══════════════════════════════════════════════
# BUILDING BLOCKS (you built all of these in Phase 3!)
# ═══════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    
    "Causal" means each position can only attend to positions before it.
    This is what makes GPT autoregressive — it generates left to right.
    
    Salesforce analogy: It's like a BEFORE trigger — each record can only
    see records that were processed before it, never after.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, Query, Value projections combined for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head
        
        # Causal mask: lower triangular matrix
        # This ensures each position can only attend to earlier positions
        self.register_buffer("mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Compute Q, K, V in one matrix multiply (efficient!)
        qkv = self.c_attn(x)  # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each: (B, T, n_embd)
        
        # Reshape into heads: (B, T, n_embd) → (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Attention: softmax(QK^T / sqrt(d_k)) @ V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, n_head, T, d_k)
        
        # Re-assemble heads: (B, n_head, T, d_k) → (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This is where the model "thinks" and stores knowledge.
    Each token gets processed independently through 2 linear layers.
    
    GPT-2: 768 → 3072 → 768 (4× expansion with GELU)
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)      # Expand: n_embd → 4*n_embd
        x = self.gelu(x)      # Non-linearity
        x = self.c_proj(x)    # Contract: 4*n_embd → n_embd
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    Pre-norm architecture (GPT-2 style):
      x → LayerNorm → Attention → + (residual)
        → LayerNorm → FFN       → + (residual)
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Attention + residual
        x = x + self.mlp(self.ln_2(x))    # FFN + residual
        return x


# ═══════════════════════════════════════════════
# THE COMPLETE GPT MODEL
# ═══════════════════════════════════════════════

class GPT(nn.Module):
    """
    The complete GPT model.
    
    This is it. This is what ChatGPT, Claude, Gemini, and LLaMA are 
    built on (with modifications and much more training data).
    
    The entire model:
    1. Token embedding + Position embedding
    2. N × Transformer blocks
    3. Final LayerNorm
    4. Linear projection to vocabulary (LM head)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: vocab_size → n_embd
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position embeddings: block_size → n_embd
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout
            drop = nn.Dropout(config.dropout),
            # Transformer blocks
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            # Final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language Model head: project from d_model back to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and LM head
        # This is a common trick that improves performance and reduces parameters
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model initialized: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights using GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        idx: (B, T) tensor of token indices
        targets: (B, T) tensor of target token indices (for training)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Forward through the model
        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)
        
        # Through all Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)          # Final LayerNorm
        logits = self.lm_head(x)              # (B, T, vocab_size)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1)                    # (B*T,)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.
        
        This is the inference loop:
        1. Feed current sequence to model → get logits for next token
        2. Sample from the logits (with temperature and top-k)
        3. Append sampled token to sequence
        4. Repeat
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
                       idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Take logits at the last position
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ═══════════════════════════════════════════════
# INSTANTIATE AND INSPECT
# ═══════════════════════════════════════════════

model = GPT(config)
print(f"\nArchitecture:")
print(f"  Layers:     {config.n_layer}")
print(f"  d_model:    {config.n_embd}")
print(f"  Heads:      {config.n_head}")
print(f"  d_k:        {config.n_embd // config.n_head}")
print(f"  d_ff:       {4 * config.n_embd}")
print(f"  Block size: {config.block_size}")
print(f"  Vocab size: {config.vocab_size}")
```

---

# WEEKS 16-17: Data Pipeline & Training Loop

## 📖 THEORY: Training GPT on Real Data

### Data Loading Strategy

```
Raw text → Tokenize → Create windows → Batch → GPU

"The cat sat on the mat." 
    ↓ tokenize
[464, 3797, 3332, 319, 262, 2603, 13]
    ↓ create input/target pairs
Input:  [464, 3797, 3332, 319, 262, 2603]
Target: [3797, 3332, 319, 262, 2603, 13]
    ↓ batch multiple sequences
(batch_size, block_size) tensor pairs
```

### Learning Rate Scheduling

Modern LLMs use a **warmup + cosine decay** schedule:

```
LR
  │    ╱──────╲
  │   /         ╲
  │  /            ╲
  │ /               ╲
  │/                  ╲___
  └─────────────────────────→ Steps
   warmup   main training  cooldown
```

---

## 💻 CODE: Complete Training Script

```python
"""
Weeks 16-17: Training GPT on Shakespeare

This trains your GPT model on the complete works of Shakespeare.
After training, it will generate Shakespeare-like text.
"""
import torch
import tiktoken
import time
import os

# ═══════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════

# Download Shakespeare
import urllib.request
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_path = 'shakespeare.txt'
if not os.path.exists(data_path):
    urllib.request.urlretrieve(data_url, data_path)

with open(data_path, 'r') as f:
    text = f.read()

print(f"Dataset: {len(text):,} characters")
print(f"First 200 chars:\n{text[:200]}")

# Tokenize with GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
print(f"\nTokenized: {len(tokens):,} tokens")

# Train/val split
n = int(0.9 * len(tokens))
train_data = tokens[:n]
val_data = tokens[n:]

# Data loader
def get_batch(split, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


# ═══════════════════════════════════════════════
# 2. TRAINING WITH LEARNING RATE SCHEDULING
# ═══════════════════════════════════════════════

# Use the GPT class and GPTConfig from Week 15 code above
model = GPT(config).to(config.device)

# AdamW optimizer — the standard for training LLMs
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.learning_rate,
    betas=(0.9, 0.95),     # Standard for LLMs
    weight_decay=0.1        # Regularization
)

# Learning rate scheduler: cosine decay with warmup
def get_lr(it, warmup_iters=100, lr_decay_iters=5000, min_lr=3e-5):
    """Cosine learning rate schedule with warmup."""
    max_lr = config.learning_rate
    # Warmup
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # After decay
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Evaluation function
@torch.no_grad()
def estimate_loss(model, config):
    """Estimate loss on train and val sets."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ═══════════════════════════════════════════════
# 3. THE TRAINING LOOP
# ═══════════════════════════════════════════════

import math

print(f"\nTraining on {config.device}")
print(f"Max iterations: {config.max_iters}")
print(f"Batch size: {config.batch_size}")
print(f"Block size: {config.block_size}")
print("=" * 60)

best_val_loss = float('inf')
start_time = time.time()

for iter_num in range(config.max_iters):
    # Update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate periodically
    if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
        losses = estimate_loss(model, config)
        elapsed = time.time() - start_time
        print(f"Step {iter_num:5d} | Train Loss: {losses['train']:.4f} | "
              f"Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | Time: {elapsed:.0f}s")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'gpt_shakespeare.pt')
    
    # Training step
    xb, yb = get_batch('train', config)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping — prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()


# ═══════════════════════════════════════════════
# 4. GENERATE TEXT!
# ═══════════════════════════════════════════════

print("\n" + "=" * 60)
print("🎭 GENERATING SHAKESPEARE")
print("=" * 60)

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
print(enc.decode(generated[0].tolist()))

print("""
🎉 Your GPT model is generating text!

What you just built is architecturally IDENTICAL to GPT-2.
The only differences from ChatGPT are:
  1. Scale: GPT-4 has ~1.8 trillion params, yours has ~10M
  2. Data: GPT-4 trained on trillions of tokens, yours on ~1M
  3. Fine-tuning: ChatGPT has additional RLHF training
  
But the ARCHITECTURE — the Transformer with causal attention —
is exactly the same.
""")
```

---

# WEEKS 18-19: Advanced Training Techniques

## 📖 THEORY: Scaling Up Training

### Gradient Accumulation

When your GPU can't fit large batches, simulate them:

```python
# Instead of: batch_size=256 (doesn't fit in memory)
# Do: batch_size=64, accumulation_steps=4 (same effective batch)

accumulation_steps = 4
optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    xb, yb = get_batch('train', config)
    logits, loss = model(xb, yb)
    loss = loss / accumulation_steps  # Scale the loss
    loss.backward()  # Accumulate gradients
optimizer.step()  # Single update with accumulated gradients
```

### Mixed Precision Training

Use float16 for forward/backward, float32 for weight updates:

```python
# On your M3 Pro:
scaler = torch.amp.GradScaler('mps')  # or 'cuda' on NVIDIA

with torch.amp.autocast(device_type='mps', dtype=torch.float16):
    logits, loss = model(xb, yb)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Text Generation Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **Greedy** | Always pick highest probability token | Deterministic output |
| **Temperature** | Scale logits by T before softmax | T<1 = more focused, T>1 = more creative |
| **Top-k** | Only sample from top k tokens | Filter out unlikely tokens |
| **Top-p (nucleus)** | Sample from smallest set with cumulative prob ≥ p | Adaptive filtering |

```python
def generate_with_strategies(model, prompt, enc, device):
    """Compare different generation strategies."""
    tokens = torch.tensor([enc.encode(prompt)], device=device)
    
    strategies = [
        ("Greedy (T=0.1)", 0.1, None),
        ("Balanced (T=0.8)", 0.8, 40),
        ("Creative (T=1.2)", 1.2, 100),
    ]
    
    for name, temp, top_k in strategies:
        output = model.generate(tokens, max_new_tokens=100, 
                                temperature=temp, top_k=top_k)
        text = enc.decode(output[0].tolist())
        print(f"\n--- {name} ---")
        print(text[:200])
```

---

## 📚 PHASE 5 COMPLETE RESOURCE LIST

### 🎥 Videos (Watch in Order)

| # | Video | Duration | Week | Purpose |
|---|-------|----------|------|---------|
| 1 | **[Karpathy: Let's Build GPT from Scratch](https://youtu.be/kCc8FmEb1nY)** | **1h56m** | **Week 15** | **THE video. Build GPT step by step.** |
| 2 | [3B1B: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 26 min | Week 15 | Revisit attention with deeper understanding |
| 3 | [3B1B: How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8) | 24 min | Week 16 | Understanding FFN layers |
| 4 | [Yannic Kilcher: GPT-2 Paper](https://www.youtube.com/watch?v=T0I88NhR_9M) | 30 min | Week 15 | Paper reading walkthrough |
| 5 | [Yannic Kilcher: GPT-3 Paper](https://www.youtube.com/watch?v=SY5PvZrJhLE) | 45 min | Week 18 | Scaling laws and in-context learning |
| 6 | [Karpathy: State of GPT (Microsoft Build)](https://www.youtube.com/watch?v=bZQun8Y4L2A) | 42 min | Week 19 | Training pipeline overview |

### 📖 Papers & Reading

| # | Resource | Week | Purpose |
|---|----------|------|---------|
| 1 | [Radford et al.: "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Week 15 | The GPT-2 paper |
| 2 | [Brown et al.: "Language Models are Few-Shot Learners" (GPT-3)](https://arxiv.org/abs/2005.14165) | Week 18 | Scaling laws |
| 3 | [Karpathy's nanoGPT repo](https://github.com/karpathy/nanoGPT) | Week 15-17 | Reference implementation |
| 4 | [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/) | Week 15 | Visual architecture guide |
| 5 | [Kaplan et al.: "Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) | Week 18 | How performance scales with compute |
| 6 | [Hoffmann et al.: "Training Compute-Optimal LLMs" (Chinchilla)](https://arxiv.org/abs/2203.15556) | Week 18 | Optimal model size vs data trade-off |

### 💻 Code Repositories

| # | Repo | Purpose |
|---|------|---------|
| 1 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Clean GPT-2 training code |
| 2 | [karpathy/minGPT](https://github.com/karpathy/minGPT) | Minimal GPT implementation |
| 3 | [karpathy/llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in C (advanced) |

---

## ✅ PHASE 5 COMPLETION CHECKLIST

- [ ] Can draw the GPT architecture from memory
- [ ] Built the complete GPT model class from scratch (all components)
- [ ] Understand weight tying between embedding and LM head
- [ ] Implemented the training loop with AdamW and cosine LR schedule
- [ ] Trained GPT on Shakespeare, generating coherent text
- [ ] Understand gradient accumulation and when to use it
- [ ] Implemented text generation with temperature, top-k, top-p
- [ ] Read the GPT-2 paper (at least abstract + architecture section)
- [ ] Studied nanoGPT repo structure
- [ ] Understand scaling laws: more data + more params = better performance
- [ ] Can explain every component's role: embedding, attention, FFN, LayerNorm, residual
- [ ] Saved model checkpoints and loaded them for inference

**🧪 Phase 4 Project:** A trained GPT model that generates coherent Shakespeare-style text, with a generation script supporting temperature, top-k, and top-p sampling.

**Next:** [Phase 6: Build Your mini-GPT (Weeks 26–29)](./Phase_6_Build_Your_GPT.md)
