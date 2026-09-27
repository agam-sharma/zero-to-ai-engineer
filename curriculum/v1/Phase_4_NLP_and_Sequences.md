# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: NLP & SEQUENCE MODELS (Weeks 16–20)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
#
# ⚠️  PHASE RENUMBERING NOTE:
# Previously "Phase 3" in an earlier version. Under the current
# 9-phase / 36-week structure this is Phase 4. In-body topic IDs
# "Week 10.x" through "Week 14.x" are preserved for git-history
# stability. Map to NEW calendar weeks as follows:
#   "Week 10" topics → Calendar Week 16 (tokenization)
#   "Week 11" topics → Calendar Week 17 (embeddings + retrieval primer)
#   "Week 12" topics → Calendar Week 18 (RNN / LSTM)
#   "Week 13" topics → Calendar Week 19 (seq2seq + Bahdanau attention)
#   "Week 14" topics → Calendar Week 20 (Shakespeare capstone)
# See 01_ZERO_TO_GPT_6_Month_Masterclass.md for the full calendar.
# ═══════════════════════════════════════════════════════════════════════

---

## 🔗 The Salesforce Analogy

> **Sequence models in AI are like Process Builder → Flow → Screen Flow evolution.**
>
> In Salesforce, you evolved from handling one record at a time (Workflow Rules) to handling sequences of actions (Process Builder) to understanding full context and branching (Flow). The same evolution happened in NLP:
>
> - **Bag of Words** → like Workflow Rules: looks at one thing, ignores order
> - **RNNs/LSTMs** → like Process Builder: processes sequentially, limited context
> - **Transformers/Attention** → like Flow: sees the ENTIRE sequence at once, routes information efficiently
>
> Phase 3 takes you through this entire evolution. By the end, you'll have built the **attention mechanism** from scratch — the single most important innovation in modern AI.

---

# WEEK 10: Tokenization — How Text Becomes Numbers

## 📖 THEORY: The Tokenization Pipeline

Before a neural network can process text, we need to convert words into numbers. This is **tokenization** — and it's more nuanced than you might think.

```
"Hello, world!" → [15496, 11, 995, 0] → model → [next_token_id] → "How"
```

### Three Levels of Tokenization

| Level | Example | Vocab Size | Used In |
|-------|---------|-----------|---------|
| **Character** | `"Hello" → ['H','e','l','l','o']` | ~100 | makemore (Phase 2) |
| **Word** | `"Hello world" → ['Hello', 'world']` | ~500,000+ | Legacy NLP |
| **Subword (BPE)** | `"unhappiness" → ['un', 'happi', 'ness']` | 50,000-100,000 | 🔥 GPT, BERT, all modern LLMs |

### Byte Pair Encoding (BPE) — How GPT Tokenizes

BPE is a **compression algorithm** turned tokenizer:

1. Start with individual bytes (256 tokens)
2. Find the most frequent pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat until you reach your desired vocabulary size

```
Starting: "l o w e r" appears 5 times, "l o w" appears 2 times, "n e w e r" appears 6 times

Step 1: Most frequent pair is "e r" → merge into "er"
        Now: "l o w er" "n e w er" ...

Step 2: Most frequent pair is "n e" → merge into "ne"
        Now: "l o w er" "ne w er" ...

Step 3: Most frequent pair is "ne w" → merge into "new"
        Now: "l o w er" "new er" ...

...continue until vocab_size reached
```

**Watch:** [Karpathy: Let's Build the GPT Tokenizer](https://youtu.be/zduSFxRajkE) (2h13m)

---

## 💻 CODE: Build a BPE Tokenizer from Scratch

```python
"""
Week 10: Build a BPE Tokenizer (Simplified version of GPT's tokenizer)

Watch first: https://youtu.be/zduSFxRajkE (Karpathy's GPT Tokenizer video)
"""

class SimpleBPETokenizer:
    """
    A minimal BPE tokenizer — the same algorithm used by GPT-2/3/4.
    
    In GPT-2:
      - Vocabulary: 50,257 tokens
      - Includes all 256 bytes + 50,000 merges + <|endoftext|>
    """
    
    def __init__(self):
        self.merges = {}  # (token_a, token_b) → new_token
        self.vocab = {}   # int → bytes
    
    def _get_pair_counts(self, ids):
        """Count frequency of each adjacent pair."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def train(self, text, vocab_size=276):
        """
        Train the tokenizer on text.
        
        vocab_size: 256 (bytes) + number of merges.
        GPT-2 uses 50,257 total.
        We use 276 for demonstration (256 + 20 merges).
        """
        # Start: convert text to raw bytes
        tokens = list(text.encode("utf-8"))
        print(f"Original text length: {len(text)} characters")
        print(f"Starting token count: {len(tokens)} bytes")
        
        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            # Find the most frequent pair
            counts = self._get_pair_counts(tokens)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            
            # Create new token ID
            new_id = 256 + i
            
            # Replace all occurrences of the pair with the new token
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            
            tokens = new_tokens
            self.merges[best_pair] = new_id
            
            # Build vocab entry
            a_bytes = self.vocab.get(best_pair[0], bytes([best_pair[0]]))
            b_bytes = self.vocab.get(best_pair[1], bytes([best_pair[1]]))
            self.vocab[new_id] = a_bytes + b_bytes
            
            if i < 10:  # Show first 10 merges
                merged_text = (a_bytes + b_bytes).decode("utf-8", errors="replace")
                print(f"  Merge {i}: {best_pair} → {new_id} "
                      f"('{merged_text}', count={counts[best_pair]})")
        
        print(f"Final token count: {len(tokens)} (compression: {len(text)/len(tokens):.1f}x)")
        return tokens
    
    def encode(self, text):
        """Tokenize text using learned merges."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            counts = self._get_pair_counts(tokens)
            # Find the pair with the lowest merge index (earliest learned)
            pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  # No more merges to apply
            new_id = self.merges[pair]
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        raw_bytes = b""
        for id in ids:
            if id < 256:
                raw_bytes += bytes([id])
            else:
                raw_bytes += self.vocab[id]
        return raw_bytes.decode("utf-8", errors="replace")


# ─── Train and Test ───
text = """The quick brown fox jumps over the lazy dog. 
The quick brown fox is faster than the lazy dog.
Neural networks learn patterns from data through training.
The transformer architecture uses self-attention mechanisms.
Self-attention allows each token to attend to all other tokens."""

tokenizer = SimpleBPETokenizer()
tokens = tokenizer.train(text, vocab_size=280)

# Test encoding/decoding roundtrip
test = "The quick brown fox"
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)
print(f"\nEncode/Decode test:")
print(f"  Original: '{test}'")
print(f"  Encoded:  {encoded}")
print(f"  Decoded:  '{decoded}'")
assert test == decoded, "Roundtrip failed!"
print("  ✅ Roundtrip successful!")


# ─── Compare with tiktoken (GPT's actual tokenizer) ───
print("\n\n=== GPT-2's Actual Tokenizer (tiktoken) ===\n")
import tiktoken

enc = tiktoken.get_encoding("gpt2")  # GPT-2's tokenizer

sentences = [
    "Hello, world!",
    "The Transformer architecture revolutionized NLP.",
    "Salesforce is a cloud-based CRM platform.",
    "supercalifragilisticexpialidocious",
    "こんにちは",  # Japanese: "Hello"
]

for s in sentences:
    tokens = enc.encode(s)
    print(f"'{s}'")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded pieces: {[enc.decode([t]) for t in tokens]}")
    print(f"  Token count: {len(tokens)}")
    print()

print(f"GPT-2 vocabulary size: {enc.n_vocab}")  # 50,257
```

---

# WEEK 11: Word Embeddings — Words as Vectors

## 📖 THEORY: The Embedding Layer

In Phase 1, you saw that words can be represented as vectors. But how do we GET those vectors?

**An embedding is a lookup table: token_id → vector**

```python
# Conceptually:
embedding_table = {
    0: [0.12, -0.34, 0.56, ...],    # Token 0's vector (768 numbers)
    1: [-0.23, 0.45, -0.67, ...],   # Token 1's vector
    2: [0.89, 0.12, -0.45, ...],    # Token 2's vector
    ...
    50256: [0.34, -0.78, 0.23, ...] # Token 50256's vector
}
```

In PyTorch:
```python
embedding = nn.Embedding(vocab_size, d_model)  # e.g., (50257, 768)
# embedding.weight is a (50257, 768) matrix — randomly initialized
# During training, these vectors are LEARNED to capture meaning
```

### Why Embeddings Work

The key insight: words that appear in similar **contexts** develop similar vectors.

```
"The ___ sat on the mat"  →  cat, dog, bird — these get similar embeddings
"The ___ is a programming language"  →  Python, Java, C++ — these get similar embeddings
```

This is the **distributional hypothesis**: "You shall know a word by the company it keeps."

---

## 💻 CODE: Exploring Pre-trained Embeddings

```python
"""
Week 11: Word Embeddings — Exploring Learned Representations
"""
import torch
import numpy as np

# ═══════════════════════════════════════════════
# 1. EMBEDDING FROM SCRATCH
# ═══════════════════════════════════════════════

print("=== Building an Embedding Layer ===\n")

vocab_size = 10   # Tiny vocab for demo
d_model = 4       # 4-dimensional embeddings

# Create embedding table (randomly initialized)
embedding = torch.nn.Embedding(vocab_size, d_model)
print(f"Embedding table shape: {embedding.weight.shape}")
print(f"Embedding table:\n{embedding.weight.data}")

# Look up embeddings for token IDs
token_ids = torch.tensor([0, 3, 7, 3])  # 4 tokens
vectors = embedding(token_ids)
print(f"\nToken IDs: {token_ids.tolist()}")
print(f"Looked up vectors:\n{vectors}")
print(f"Note: tokens 3 (index 1) and 3 (index 3) have IDENTICAL vectors")


# ═══════════════════════════════════════════════
# 2. POSITIONAL ENCODING — Position Matters!
# ═══════════════════════════════════════════════

print("\n=== Positional Encoding ===\n")

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    The original Transformer's positional encoding.
    Uses sine and cosine waves of different frequencies.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why this works:
    - Each position gets a unique encoding
    - Nearby positions have similar encodings  
    - The model can learn to compute relative positions from these
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    
    return pe

# Visualize
import matplotlib.pyplot as plt

pe = sinusoidal_positional_encoding(50, 64)
plt.figure(figsize=(12, 5))
plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position in Sequence')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved positional_encoding.png")

# How it's used in the Transformer:
print("""
In the Transformer:
  token_embedding = embedding(token_ids)     # (seq_len, d_model)
  position_embedding = PE[:seq_len]          # (seq_len, d_model)
  input = token_embedding + position_embedding  # Element-wise addition!

Without positional encoding, the model can't tell the difference between:
  "The dog bit the man"  and  "The man bit the dog"
because self-attention is permutation-invariant by default.

GPT-2 uses LEARNED positional embeddings instead of sinusoidal:
  position_embedding = nn.Embedding(max_seq_len, d_model)
  # These are learned during training, just like token embeddings
""")
```

---

# WEEK 12: RNNs & LSTMs — The Sequential Models

## 📖 THEORY: Recurrent Neural Networks

Before Transformers, RNNs were the standard for sequence modeling. Understanding them helps you appreciate WHY Transformers were such a breakthrough.

### How an RNN Works

```
"The" → [RNN] → h₁
         ↓
"cat" → [RNN] → h₂  (uses h₁ as additional input)
         ↓
"sat" → [RNN] → h₃  (uses h₂ as additional input)
         ↓
 ???  → [predict] → "on"
```

At each timestep:
```python
h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
```

**The problem:** As sequences get longer, gradients vanish during backpropagation through time. The RNN "forgets" the beginning of the sequence.

### LSTM: The Fix (Partially)

LSTMs add **gates** that control information flow:
- **Forget gate:** What old information to discard
- **Input gate:** What new information to store
- **Output gate:** What to output from the cell

```
Salesforce analogy: 
  RNN = A Flow with no loop protection — crashes on long sequences
  LSTM = A Flow with proper error handling — works better but still sequential
  Transformer = Parallel Apex batch processing — handles everything at once
```

---

## 💻 CODE: RNN vs LSTM vs Transformer Speed Comparison

```python
"""
Week 12: RNNs & LSTMs — Understanding and Comparing
"""
import torch
import torch.nn as nn
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══════════════════════════════════════════════
# 1. BUILD: Simple RNN from scratch
# ═══════════════════════════════════════════════

class SimpleRNN(nn.Module):
    """
    Manual RNN implementation for understanding.
    
    At each timestep:
      h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(seq_len):  # SEQUENTIAL — this is the bottleneck!
            h = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h))
        
        return self.W_hy(h)  # Use final hidden state


# ═══════════════════════════════════════════════
# 2. COMPARE: RNN vs LSTM vs Transformer
# ═══════════════════════════════════════════════

batch_size = 32
seq_len = 100
d_model = 128
n_classes = 10

# Input: batch of sequences
x = torch.randn(batch_size, seq_len, d_model).to(device)

# Model 1: Simple RNN
rnn = SimpleRNN(d_model, 256, n_classes).to(device)

# Model 2: LSTM (PyTorch built-in)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(d_model, 256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last timestep

lstm = LSTMModel().to(device)

# Model 3: Transformer-based (preview of Phase 4!)
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, n_classes)
    def forward(self, x):
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

transformer = TransformerModel().to(device)

# Benchmark
models = [("RNN", rnn), ("LSTM", lstm), ("Transformer", transformer)]

print("=" * 60)
print(f"Speed Comparison: seq_len={seq_len}, batch={batch_size}")
print("=" * 60)

for name, model in models:
    # Warm up
    with torch.no_grad():
        _ = model(x)
    
    # Time it
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    elapsed = (time.time() - start) / 100
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name:12s}: {elapsed*1000:6.2f}ms per batch | {params:,} params")

print("""
Key Observation:
  RNN/LSTM: Must process tokens ONE AT A TIME (sequential for-loop)
  Transformer: Processes ALL tokens simultaneously (parallelizable!)
  
  This is why Transformers are so much faster on GPU:
  GPUs are designed for PARALLEL operations (matrix multiplies),
  and Transformers are ALL matrix multiplies — no sequential loops.
  
  This parallelism is what made GPT possible.
  Training GPT-3 with LSTMs would have taken 100x longer.
""")
```

---

# WEEK 13: Self-Attention from Scratch — THE Core Innovation

## 📖 THEORY: The Attention Mechanism (Deep Dive)

This is the single most important concept in the entire course. Everything in modern AI builds on this.

### The Intuition

When you read: "The **animal** didn't cross the street because **it** was too tired."

What does "it" refer to? **The animal.** You know this because you **attend** to the relevant earlier word.

Self-attention gives each word the ability to "look at" every other word in the sentence and decide which ones are relevant.

### The Q, K, V Framework

Think of it as a **search engine**:

```
Q (Query)  = "What am I looking for?"
K (Key)    = "What do I contain?"  
V (Value)  = "What information should I provide?"

Attention = "For each query, find the most relevant keys, 
             and return a weighted sum of their values."
```

**Salesforce analogy:**
```
Q = SOQL WHERE clause  (what are you searching for?)
K = Record field values (what's in each record?)
V = Record data         (what you actually want to retrieve)

SELECT V FROM Records WHERE K matches Q
```

### The Math

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Step by step:
1. **$QK^T$**: Compute similarity between every pair of positions → $(n \times n)$ score matrix
2. **$/ \sqrt{d_k}$**: Scale to prevent scores from getting too large (which would make softmax too "peaked")
3. **softmax**: Convert scores to probabilities (each row sums to 1)
4. **$× V$**: Weighted sum of value vectors using the attention weights

---

## 💻 CODE: Self-Attention from Absolute Scratch

```python
"""
Week 13: Self-Attention — The Complete Implementation

This is the EXACT computation inside every Transformer layer.
If you understand this code, you understand Transformers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════
# SINGLE-HEAD SELF-ATTENTION (from scratch)
# ═══════════════════════════════════════════════

class SelfAttention(nn.Module):
    """
    Single-head self-attention.
    
    For each position in the sequence:
    1. Create a Query: "What am I looking for?"
    2. Create a Key: "What do I represent?"
    3. Create a Value: "What information do I hold?"
    4. Compute attention scores: Q @ K^T / sqrt(d_k)
    5. Apply softmax to get weights
    6. Compute weighted sum of Values
    """
    
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        
        # Three learned projection matrices
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: optional causal mask for autoregressive models
        """
        # Step 1: Project input to Q, K, V
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_k)
        
        # Step 2: Compute attention scores
        # (B, T, d_k) @ (B, d_k, T) → (B, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        # Step 3: Apply causal mask (for GPT-style models)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Softmax → attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Weighted sum of values
        # (B, T, T) @ (B, T, d_k) → (B, T, d_k)
        output = attention_weights @ V
        
        return output, attention_weights


# ═══════════════════════════════════════════════
# TEST: Visualize Attention Patterns
# ═══════════════════════════════════════════════

import matplotlib.pyplot as plt

torch.manual_seed(42)

# Simulate: "The cat sat on the mat"
batch_size = 1
seq_len = 6
d_model = 32
d_k = 16

# Random input embeddings (in practice, these come from the embedding layer)
x = torch.randn(batch_size, seq_len, d_model)

# Create causal mask (each position can only attend to itself and earlier positions)
# This is what makes GPT "autoregressive"
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print("Causal Mask:")
print(causal_mask[0].int())
print("""
1 0 0 0 0 0  → "The" can only see itself
1 1 0 0 0 0  → "cat" can see "The" and "cat"
1 1 1 0 0 0  → "sat" can see "The", "cat", "sat"
1 1 1 1 0 0  → ...
1 1 1 1 1 0  
1 1 1 1 1 1  → "mat" can see everything
""")

# Run self-attention
attn = SelfAttention(d_model, d_k)
output, weights = attn(x, mask=causal_mask)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

# Visualize
words = ["The", "cat", "sat", "on", "the", "mat"]
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(weights[0].detach().numpy(), cmap='Blues')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(words)
ax.set_yticklabels(words)
ax.set_xlabel("Key (attending TO)")
ax.set_ylabel("Query (attending FROM)")
ax.set_title("Self-Attention Weights (Causal Masked)")
plt.colorbar(im)
plt.savefig('self_attention_weights.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: run self-attention multiple times in parallel,
    each with different learned projections.
    
    Why multiple heads?
    - Head 1 might learn to attend to syntactic relationships
    - Head 2 might learn to attend to semantic relationships
    - Head 3 might learn to attend to positional patterns
    - Each head can focus on a different "type" of relationship
    
    GPT-2:  12 heads, each with d_k = 64 (total: 12 × 64 = 768 = d_model)
    GPT-3:  96 heads, each with d_k = 128 (total: 96 × 128 = 12288 = d_model)
    """
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Single large projection for efficiency (instead of n_heads separate ones)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # Output projection
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project and reshape into multiple heads
        # (B, T, d_model) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention for all heads simultaneously
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) → (B, n_heads, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) → (B, n_heads, T, d_k)
        out = weights @ V
        
        # Concatenate heads: (B, n_heads, T, d_k) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        out = self.W_O(out)
        
        return out, weights


# Test multi-head attention
mha = MultiHeadAttention(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

out, weights = mha(x, mask=mask)
print(f"\nMulti-Head Attention:")
print(f"  Input:   {x.shape}")
print(f"  Output:  {out.shape}")
print(f"  Weights: {weights.shape}")  # (B, n_heads, T, T)

print("""
🎉 You just built Multi-Head Self-Attention from scratch!

This is the CORE building block of the Transformer.
In Phase 4, you'll combine this with:
  - LayerNorm
  - Feed-Forward Network
  - Residual Connections
to build a complete Transformer block.
""")
```

---

# WEEK 14: The Feed-Forward Network, LayerNorm & Residual Connections

## 📖 THEORY: The Other Half of the Transformer Block

Self-attention is only HALF of a Transformer block. The other half:

```
┌─────────────────────────────────────────────┐
│          TRANSFORMER BLOCK                    │
│                                               │
│  Input ──→ LayerNorm ──→ Multi-Head Attention │
│    │                          │                │
│    └────────── + ◄────────────┘  ← Residual   │
│                │                  Connection   │
│                ▼                               │
│         LayerNorm ──→ Feed-Forward Network     │
│    │                          │                │
│    └────────── + ◄────────────┘  ← Residual   │
│                │                  Connection   │
│                ▼                               │
│             Output                             │
└─────────────────────────────────────────────┘
```

### Layer Normalization

Normalizes each token's embedding to have mean=0, std=1:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Where $\mu$ and $\sigma$ are computed over the embedding dimension (not the batch).

### Feed-Forward Network (FFN)

A simple 2-layer MLP applied to each position independently:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

In GPT-2: $W_1$ expands from 768 → 3072 (4× expansion), then $W_2$ projects back 3072 → 768.

**Why 4× expansion?** The FFN is where the model "thinks" and stores factual knowledge. More neurons = more storage capacity.

### Residual Connections

The `+` in the diagram. Instead of `output = f(input)`, we do `output = input + f(input)`.

**Why?** Without residuals, gradients must pass through every layer during backpropagation. With residuals, gradients can skip layers — solving the vanishing gradient problem for deep networks.

```python
# Without residual: output = attention(layernorm(x))
# With residual:    output = x + attention(layernorm(x))
#                           ↑ skip connection — gradient flows directly!
```

---

## 💻 CODE: Complete Transformer Block

```python
"""
Week 14: Building a Complete Transformer Block

After this, you have ALL the pieces to build GPT in Phase 4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    
    In GPT-2: 768 → 3072 → 768 (4× expansion)
    
    Jay Alammar calls this "the transformer's memory bank" —
    it's where factual knowledge is believed to be stored.
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4  # Default: 4× expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block (GPT-style).
    
    This is the REPEATING UNIT in GPT:
      GPT-2 Small:  12 of these blocks stacked
      GPT-2 Medium: 24 blocks
      GPT-2 Large:  36 blocks
      GPT-2 XL:     48 blocks
      GPT-3:        96 blocks
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)        # Pre-norm (GPT-2 style)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)  # Residual connection!
        
        # Feed-forward with residual connection
        x = x + self.dropout(self.ffn(self.ln2(x)))  # Residual connection!
        
        return x, attn_weights


# Need MultiHeadAttention from Week 13 (copy it here or import)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out), weights


# ═══════════════════════════════════════════════
# TEST: Stack Multiple Transformer Blocks
# ═══════════════════════════════════════════════

d_model = 128
n_heads = 4
n_layers = 6
seq_len = 32
batch_size = 4

# Stack blocks
blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])

# Create input and causal mask
x = torch.randn(batch_size, seq_len, d_model)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

# Forward through all blocks
for i, block in enumerate(blocks):
    x, _ = block(x, mask=mask)

print(f"Input shape:  (batch={batch_size}, seq={seq_len}, d_model={d_model})")
print(f"Output shape: {x.shape}")
print(f"Blocks:       {n_layers}")
print(f"Total params: {sum(p.numel() for b in blocks for p in b.parameters()):,}")

print("""
🎉 You just built and ran a 6-layer Transformer!

This is the COMPLETE architecture of GPT (minus the embedding 
and output layers, which you'll add in Phase 4).

Summary of what each component does:
  LayerNorm   → Stabilizes training
  Attention   → Lets tokens communicate (mix information)
  Residual +  → Lets gradients flow through deep networks
  FFN         → Per-token processing (stores knowledge)
  
In Phase 4, you'll wrap this in a full GPT model with:
  - Token + Position embeddings
  - LM head (output projection)
  - Training loop on real text data
""")
```

---

## 📚 PHASE 4 COMPLETE RESOURCE LIST

### 🎥 Videos (Watch in Order)

| # | Video | Duration | Week | Purpose |
|---|-------|----------|------|---------|
| 1 | **[Karpathy: GPT Tokenizer](https://youtu.be/zduSFxRajkE)** | **2h13m** | **Week 10** | **Build BPE tokenizer from scratch** |
| 2 | [Karpathy: makemore Part 1 (bigrams)](https://youtu.be/PaCmpygFfXo) | 1h57m | Week 10 | Character-level language model fundamentals |
| 3 | [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) | article | Week 11 | Visual guide to word embeddings |
| 4 | [StatQuest: Word Embedding & Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0) | 17 min | Week 11 | Clear, step-by-step explanation |
| 5 | [3B1B: But what is a GPT? (Transformers visualized)](https://www.youtube.com/watch?v=wjZofJX0v4M) | 27 min | Week 12 | LLM overview before deep dive |
| 6 | [3B1B: Attention in Transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 26 min | Week 13 | THE best attention visualization |
| 7 | [3B1B: How might LLMs store facts (MLPs in Transformers)](https://www.youtube.com/watch?v=9-Jl0dxWQs8) | 24 min | Week 14 | Feed-forward network role |
| 8 | **[Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** | article | **Week 13-14** | **THE definitive visual guide** |
| 9 | **[Jay Alammar: The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)** | article | **Week 14** | **Decoder-only architecture explained** |
| 10 | [Karpathy: makemore Part 5 (WaveNet)](https://youtu.be/t3YJ5hKiMQ0) | 56 min | Week 12 | Dilated causal convolutions |

### 📖 Papers & Reading

| # | Resource | Week | Purpose |
|---|----------|------|---------|
| 1 | ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Week 13 | THE Transformer paper — read after building attention |
| 2 | [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) | Week 14 | Line-by-line implementation of the paper |
| 3 | [Sennrich et al.: BPE for NMT](https://arxiv.org/abs/1508.07909) | Week 10 | Original BPE for NLP paper |
| 4 | [Mikolov et al.: Word2Vec](https://arxiv.org/abs/1301.3781) | Week 11 | Original word embeddings paper |
| 5 | [Ba et al.: Layer Normalization](https://arxiv.org/abs/1607.06450) | Week 14 | LayerNorm paper |

---

## ✅ PHASE 4 COMPLETION CHECKLIST

- [ ] Built a BPE tokenizer from scratch
- [ ] Can use tiktoken to tokenize text with GPT-2's vocabulary
- [ ] Understand embeddings: token embeddings + positional embeddings
- [ ] Built an RNN from scratch, understand its sequential bottleneck
- [ ] Understand why Transformers replaced RNNs (parallelism + long-range attention)
- [ ] Built single-head self-attention from scratch
- [ ] Built multi-head attention from scratch
- [ ] Can explain Q, K, V in your own words (and with the SOQL analogy)
- [ ] Built the feed-forward network with 4× expansion + GELU
- [ ] Understand residual connections and why they're essential
- [ ] Understand LayerNorm and where it goes in the block (pre-norm vs post-norm)
- [ ] Built a complete Transformer block and stacked multiple blocks
- [ ] Read "The Illustrated Transformer" by Jay Alammar
- [ ] Skimmed "Attention Is All You Need" paper

**🧪 Phase 3 Project:** Implement a complete Transformer block from scratch. Pass a sequence through it and visualize the attention weights. Be able to explain every line.

**Next:** [Phase 5: The Transformer (Weeks 21–25)](./Phase_5_The_Transformer.md)
