# Source: AI_Learning_Cursor lines 23260-23420
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Language Modeling Objective
# Title: LANGUAGE MODELING OBJECTIVE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
LANGUAGE MODELING OBJECTIVE
===========================
Understanding how GPT learns from text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================
# 1. NEXT TOKEN PREDICTION
# =================================

print("=" * 60)
print("1. NEXT TOKEN PREDICTION")
print("=" * 60)

"""
GPT'S TRAINING OBJECTIVE: Predict the next token

Given: "The cat sat on the"
Predict: "mat"

More precisely, at EACH position, predict the NEXT token:

Input:    [The] [cat] [sat] [on]  [the]
Target:   [cat] [sat] [on]  [the] [mat]

Position 0: Given "The" -> predict "cat"
Position 1: Given "The cat" -> predict "sat"
Position 2: Given "The cat sat" -> predict "on"
...

This is called CAUSAL LANGUAGE MODELING.

Loss = Cross-entropy between predicted and actual next tokens

The model learns:
- Grammar (syntax)
- Facts (knowledge)
- Reasoning patterns
- Style and tone

All from just predicting the next word!
"""

# Demonstration
def demonstrate_lm_objective():
    """Show how the LM objective works."""
    
    # Example sentence
    sentence = "The quick brown fox jumps"
    tokens = sentence.split()
    
    print("Sentence:", sentence)
    print("\nTraining examples extracted:")
    print("-" * 50)
    
    for i in range(len(tokens) - 1):
        context = " ".join(tokens[:i+1])
        target = tokens[i+1]
        print(f"  Context: '{context}'")
        print(f"  Target:  '{target}'")
        print()


demonstrate_lm_objective()

# =================================
# 2. CROSS-ENTROPY LOSS
# =================================

print("\n" + "=" * 60)
print("2. CROSS-ENTROPY LOSS FOR LANGUAGE MODELING")
print("=" * 60)

"""
For each position, the model outputs a probability distribution
over the entire vocabulary.

Example with vocab_size = 5:
    Vocabulary: {0: "the", 1: "cat", 2: "sat", 3: "on", 4: "mat"}
    
    At position where target is "cat" (index 1):
    Model outputs: [0.1, 0.6, 0.15, 0.1, 0.05]  # Probabilities for each token
    Target: [0, 1, 0, 0, 0]  # One-hot for "cat"
    
    Cross-entropy loss = -log(0.6) = 0.51
    
If model was more confident:
    Model outputs: [0.01, 0.95, 0.02, 0.01, 0.01]
    Cross-entropy loss = -log(0.95) = 0.05  # Lower loss = better!

If model was wrong:
    Model outputs: [0.6, 0.1, 0.15, 0.1, 0.05]  # Predicted "the"
    Cross-entropy loss = -log(0.1) = 2.30  # Higher loss = worse!
"""

# Demonstration
vocab = ["the", "cat", "sat", "on", "mat"]
vocab_size = len(vocab)

# Simulated model outputs (logits) for predicting "cat"
logits = torch.tensor([[2.0, 3.5, 1.0, 0.5, 0.2]])  # Higher for "cat"
probs = F.softmax(logits, dim=-1)
target = torch.tensor([1])  # "cat"

loss = F.cross_entropy(logits, target)

print(f"Vocabulary: {vocab}")
print(f"Target token: '{vocab[target.item()]}' (index {target.item()})")
print(f"\nModel outputs:")
print(f"  Logits: {logits.squeeze().tolist()}")
print(f"  Probs:  {probs.squeeze().tolist()}")
print(f"\nCross-entropy loss: {loss.item():.4f}")
print(f"Probability assigned to correct token: {probs[0, target.item()].item():.4f}")

# =================================
# 3. PERPLEXITY
# =================================

print("\n" + "=" * 60)
print("3. PERPLEXITY - THE LM EVALUATION METRIC")
print("=" * 60)

"""
PERPLEXITY = exp(cross_entropy_loss)

Interpretation: 
"How many tokens is the model choosing between at each position?"

Perplexity = 1:   Model is 100% certain (perfect predictions)
Perplexity = 10:  Model is choosing between ~10 equally likely tokens
Perplexity = V:   Model is random (V = vocabulary size)

Lower perplexity = Better model

GPT-2 perplexity on WikiText-103: ~18-20
GPT-3 perplexity: ~10-15
Human-level on typical text: ~5-10
"""

def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(loss).item()


# Example losses and their perplexities
losses = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
print("Loss vs Perplexity:")
for l in losses:
    ppl = calculate_perplexity(torch.tensor(l))
    print(f"  Loss = {l:.1f} -> Perplexity = {ppl:.1f}")

print("\nFor reference:")
print("  Random guessing (vocab=50000): Perplexity ≈ 50000")
print("  Good language model: Perplexity ≈ 10-30")
print("  Excellent language model: Perplexity < 10")
