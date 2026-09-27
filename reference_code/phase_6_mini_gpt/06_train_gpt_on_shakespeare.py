# Source: AI_Learning_Cursor lines 24360-24802
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Train Your GPT
# Title: TRAINING YOUR GPT ON SHAKESPEARE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TRAINING YOUR GPT ON SHAKESPEARE
================================
The complete training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. CONFIGURATION
# =================================

print("=" * 60)
print("1. MODEL CONFIGURATION")
print("=" * 60)

class GPTConfig:
    """GPT configuration for Shakespeare model."""
    
    # Model architecture
    vocab_size = 65        # Character vocabulary (will be set from data)
    block_size = 256       # Context length
    n_layer = 6            # Number of transformer layers
    n_head = 6             # Number of attention heads
    n_embd = 384           # Embedding dimension
    dropout = 0.2          # Dropout rate
    
    # Training
    batch_size = 64
    max_steps = 5000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 3e-4
    min_lr = 3e-5
    warmup_steps = 100
    weight_decay = 0.1
    grad_clip = 1.0
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f"GPTConfig({self.__dict__})"


config = GPTConfig()
print(config)

# =================================
# 2. COMPLETE GPT MODEL
# =================================

print("\n" + "=" * 60)
print("2. GPT MODEL")
print("=" * 60)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model with {n_params:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx

# =================================
# 3. DATA PREPARATION
# =================================

print("\n" + "=" * 60)
print("3. DATA PREPARATION")
print("=" * 60)

# Download/load Shakespeare
import requests

def get_shakespeare():
    """Get Shakespeare text."""
    if os.path.exists('shakespeare.txt'):
        with open('shakespeare.txt', 'r') as f:
            return f.read()
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        text = requests.get(url).text
        with open('shakespeare.txt', 'w') as f:
            f.write(text)
        return text
    except:
        # Fallback
        return """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 1000


text = get_shakespeare()
print(f"Loaded {len(text):,} characters")

# Create tokenizer
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

config.vocab_size = len(chars)
print(f"Vocabulary size: {config.vocab_size}")

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")


def get_batch(split):
    """Get a random batch of data."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)


# Test batch
x, y = get_batch('train')
print(f"\nBatch shapes: x={x.shape}, y={y.shape}")

# =================================
# 4. TRAINING LOOP
# =================================

print("\n" + "=" * 60)
print("4. TRAINING")
print("=" * 60)

# Create model
model = GPT(config).to(device)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=config.weight_decay
)


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(step):
    """Learning rate schedule with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# Training
train_losses = []
val_losses = []
best_val_loss = float('inf')

print(f"Starting training for {config.max_steps} steps...")
start_time = time.time()

for step in range(config.max_steps):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate periodically
    if step % config.eval_interval == 0 or step == config.max_steps - 1:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        
        elapsed = time.time() - start_time
        
        print(f"Step {step:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'step': step,
                'val_loss': best_val_loss,
            }, 'gpt_shakespeare_best.pth')
        
        # Generate sample
        if step % (config.eval_interval * 2) == 0:
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            print(f"\n--- Generated sample ---")
            print(decode(generated[0].tolist()))
            print(f"--- End sample ---\n")
            model.train()
    
    # Training step
    X, Y = get_batch('train')
    _, loss = model(X, Y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s")
print(f"Best validation loss: {best_val_loss:.4f}")

# =================================
# 5. VISUALIZATION
# =================================

print("\n" + "=" * 60)
print("5. TRAINING CURVES")
print("=" * 60)

plt.figure(figsize=(10, 4))
steps = list(range(0, config.max_steps, config.eval_interval))
plt.plot(steps[:len(train_losses)], train_losses, label='Train')
plt.plot(steps[:len(val_losses)], val_losses, label='Validation')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('GPT Training on Shakespeare')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gpt_training_loss.png', dpi=100)
print("Saved gpt_training_loss.png")
plt.show()

# =================================
# 6. FINAL GENERATION
# =================================

print("\n" + "=" * 60)
print("6. FINAL GENERATION")
print("=" * 60)

# Load best model
checkpoint = torch.load('gpt_shakespeare_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Generating Shakespeare-style text...\n")

prompts = [
    "To be, or not to be",
    "ROMEO:",
    "The king",
    "Alas, poor",
]

for prompt in prompts:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    output = decode(generated[0].tolist())
    print(f"Prompt: '{prompt}'")
    print(f"Generated:\n{output}\n")
    print("-" * 50 + "\n")

print("Your GPT is trained and generating Shakespeare! 🎭")
