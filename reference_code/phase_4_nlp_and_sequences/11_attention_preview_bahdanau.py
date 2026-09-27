# Source: AI_Learning_Cursor lines 13545-13819
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: The Bottleneck Problem and Attention Introduction
# Title: THE ATTENTION MECHANISM PREVIEW
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
THE ATTENTION MECHANISM PREVIEW
===============================
Understanding why attention was invented and how it works.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =================================
# 1. THE BOTTLENECK PROBLEM
# =================================

print("=" * 60)
print("1. THE BOTTLENECK PROBLEM")
print("=" * 60)

"""
PROBLEM WITH VANILLA SEQ2SEQ:

The entire source sequence is compressed into a single fixed-size vector.

For short sequences: Works okay
For long sequences: Information loss!

Imagine compressing a 500-word paragraph into a single vector.
The decoder has to reconstruct everything from this one vector.

Example:
    Input:  "The quick brown fox that we saw yesterday in the park..."
    Context: [0.2, -0.1, 0.8, ...]  (Just 256 numbers!)
    
How can 256 numbers encode everything about a long sentence?
They can't! This is the bottleneck.

SOLUTION: ATTENTION

Instead of one fixed context vector, let the decoder "look back"
at ALL encoder hidden states and decide which ones are relevant
for each output step.

When generating "rapide" (French for "quick"), pay attention to "quick".
When generating "renard" (French for "fox"), pay attention to "fox".
"""

# Visualize the bottleneck
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Without attention
ax1 = axes[0]
ax1.text(0.1, 0.9, 'Encoder', fontsize=14, fontweight='bold')
for i, word in enumerate(['The', 'quick', 'brown', 'fox', 'jumps']):
    ax1.add_patch(plt.Rectangle((0.1 + i*0.15, 0.6), 0.12, 0.2, fill=True, color='lightblue'))
    ax1.text(0.16 + i*0.15, 0.7, word, fontsize=10, ha='center')

# Arrows converging to single point
for i in range(5):
    ax1.annotate('', xy=(0.5, 0.35), xytext=(0.16 + i*0.15, 0.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Context vector
ax1.add_patch(plt.Circle((0.5, 0.35), 0.08, fill=True, color='red', alpha=0.5))
ax1.text(0.5, 0.35, 'C', fontsize=12, ha='center', va='center', fontweight='bold')
ax1.text(0.5, 0.2, 'Context\n(bottleneck)', fontsize=10, ha='center')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('Without Attention: Single Bottleneck', fontsize=12)

# With attention
ax2 = axes[1]
ax2.text(0.1, 0.9, 'Encoder', fontsize=14, fontweight='bold')
for i, word in enumerate(['The', 'quick', 'brown', 'fox', 'jumps']):
    ax2.add_patch(plt.Rectangle((0.1 + i*0.15, 0.6), 0.12, 0.2, fill=True, color='lightblue'))
    ax2.text(0.16 + i*0.15, 0.7, word, fontsize=10, ha='center')

ax2.text(0.1, 0.15, 'Decoder', fontsize=14, fontweight='bold')
ax2.add_patch(plt.Rectangle((0.4, 0.05), 0.2, 0.15, fill=True, color='lightgreen'))
ax2.text(0.5, 0.12, 'Output', fontsize=10, ha='center')

# Attention weights (different thicknesses)
weights = [0.1, 0.5, 0.2, 0.15, 0.05]  # Attention to each input
for i, w in enumerate(weights):
    ax2.annotate('', xy=(0.5, 0.2), xytext=(0.16 + i*0.15, 0.6),
                arrowprops=dict(arrowstyle='->', color='purple', lw=w*5, alpha=0.7))

ax2.text(0.8, 0.5, 'Attention\nweights', fontsize=10, ha='center')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('With Attention: Direct Access to All States', fontsize=12)

plt.tight_layout()
plt.savefig('attention_motivation.png', dpi=100)
print("Saved attention_motivation.png")
plt.show()

# =================================
# 2. SIMPLE ATTENTION MECHANISM
# =================================

print("\n" + "=" * 60)
print("2. SIMPLE ATTENTION")
print("=" * 60)

"""
ATTENTION MECHANISM (Simplified)

Given:
- Decoder hidden state h_t (what we're looking for)
- Encoder hidden states H = [h_1, h_2, ..., h_n] (where to look)

1. Compute ATTENTION SCORES (how relevant is each encoder state)
   score_i = dot(h_t, h_i)  # or other scoring function

2. Convert to ATTENTION WEIGHTS (probability distribution)
   α = softmax(scores)

3. Compute CONTEXT VECTOR (weighted sum of encoder states)
   context = Σ α_i * h_i

4. Use context for prediction
   output = f(context, h_t)
"""

class SimpleAttention(nn.Module):
    """
    Basic dot-product attention mechanism.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch, hidden) current decoder state
            encoder_outputs: (batch, src_len, hidden) all encoder states
            
        Returns:
            context: (batch, hidden) weighted sum of encoder states
            attention_weights: (batch, src_len) attention distribution
        """
        # Compute attention scores
        # (batch, hidden) @ (batch, hidden, src_len) -> (batch, src_len)
        scores = torch.bmm(
            decoder_hidden.unsqueeze(1),  # (batch, 1, hidden)
            encoder_outputs.transpose(1, 2)  # (batch, hidden, src_len)
        ).squeeze(1)  # (batch, src_len)
        
        # Convert to weights (probability distribution)
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context vector
        # (batch, 1, src_len) @ (batch, src_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


# Test attention
batch_size = 2
src_len = 5
hidden_dim = 8

encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
decoder_hidden = torch.randn(batch_size, hidden_dim)

attention = SimpleAttention(hidden_dim)
context, weights = attention(decoder_hidden, encoder_outputs)

print(f"Encoder outputs shape: {encoder_outputs.shape}")
print(f"Decoder hidden shape: {decoder_hidden.shape}")
print(f"Context vector shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nSample attention weights:")
print(f"  {weights[0].detach().numpy()}")
print(f"  Sum: {weights[0].sum().item():.4f} (should be 1.0)")

# =================================
# 3. VISUALIZING ATTENTION
# =================================

print("\n" + "=" * 60)
print("3. VISUALIZING ATTENTION")
print("=" * 60)

def visualize_attention(attention_weights, src_words, trg_words):
    """Visualize attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cax = ax.matshow(attention_weights, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticklabels([''] + src_words, rotation=45, ha='left')
    ax.set_yticklabels([''] + trg_words)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=100)
    print("Saved attention_visualization.png")
    plt.show()


# Simulated attention for translation
# "The quick brown fox" -> "Le renard brun rapide"
src_words = ['The', 'quick', 'brown', 'fox', '<EOS>']
trg_words = ['Le', 'renard', 'brun', 'rapide', '<EOS>']

# Simulated attention matrix (what we'd expect)
attention_matrix = np.array([
    [0.8, 0.1, 0.05, 0.04, 0.01],  # "Le" attends to "The"
    [0.05, 0.1, 0.1, 0.7, 0.05],   # "renard" attends to "fox"
    [0.05, 0.1, 0.7, 0.1, 0.05],   # "brun" attends to "brown"
    [0.05, 0.7, 0.1, 0.1, 0.05],   # "rapide" attends to "quick"
    [0.1, 0.1, 0.1, 0.1, 0.6],     # "<EOS>" attends to "<EOS>"
])

visualize_attention(attention_matrix, src_words, trg_words)

print("\nKey observations:")
print("- Each output word attends strongly to its corresponding input")
print("- Word order can be different (French adjectives after nouns)")
print("- Attention learns which input positions matter for each output")

# =================================
# 4. PREVIEW: TRANSFORMERS
# =================================

print("\n" + "=" * 60)
print("4. PREVIEW: TRANSFORMERS (PHASE 4)")
print("=" * 60)

"""
TRANSFORMERS: Attention Is All You Need

What if we ONLY used attention, without any RNNs?

Key innovations:
1. SELF-ATTENTION: Each position attends to all positions
2. MULTI-HEAD ATTENTION: Multiple attention mechanisms in parallel
3. POSITIONAL ENCODING: Add position information since no recurrence

Advantages over RNN+Attention:
- Parallelizable (no sequential processing)
- Better at capturing long-range dependencies
- More efficient to train

This is what you'll build in Phase 4!

GPT = Decoder-only Transformer + massive data + massive parameters
"""

print("\nComing in Phase 4:")
print("- Self-Attention mechanism")
print("- Multi-Head Attention")
print("- Positional Encoding")
print("- Full Transformer architecture")
print("- Building your own mini-GPT!")
