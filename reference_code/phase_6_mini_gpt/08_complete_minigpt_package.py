# Source: AI_Learning_Cursor lines 25148-25358
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Complete Mini-GPT Package
# Title: COMPLETE MINI-GPT PACKAGE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
COMPLETE MINI-GPT PACKAGE
=========================
Everything packaged together for easy use.
"""

import torch
import torch.nn as nn
import math
import json
import os

# =================================
# 1. FINAL MODEL EVALUATION
# =================================

print("=" * 60)
print("1. MODEL EVALUATION")
print("=" * 60)

def evaluate_perplexity(model, data, block_size, batch_size=32, num_batches=100, device='cpu'):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            # Get random batch
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[j:j+block_size] for j in ix]).to(device)
            y = torch.stack([data[j+1:j+block_size+1] for j in ix]).to(device)
            
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


print("Evaluation metrics:")
print("  - Perplexity: Lower is better (1 = perfect)")
print("  - Loss: Cross-entropy loss")
print("  - Tokens/second: Generation speed")

# =================================
# 2. SAVE/LOAD UTILITIES
# =================================

print("\n" + "=" * 60)
print("2. MODEL PACKAGING")
print("=" * 60)

def save_model(model, tokenizer_info, path):
    """Save complete model package."""
    os.makedirs(path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    
    # Save config
    config_dict = {
        'vocab_size': model.config.vocab_size,
        'block_size': model.config.block_size,
        'n_layer': model.config.n_layer,
        'n_head': model.config.n_head,
        'n_embd': model.config.n_embd,
        'dropout': model.config.dropout,
    }
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save tokenizer
    with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    print(f"Model saved to {path}/")


def load_model(path, device='cpu'):
    """Load complete model package."""
    # Load config
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    # Create model
    config = GPTConfig(**config_dict)
    model = GPT(config)
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    with open(os.path.join(path, 'tokenizer.json'), 'r') as f:
        tokenizer_info = json.load(f)
    
    return model, tokenizer_info


print("Model packaging utilities ready!")

# =================================
# 3. USAGE EXAMPLE
# =================================

print("\n" + "=" * 60)
print("3. COMPLETE USAGE EXAMPLE")
print("=" * 60)

example_code = '''
# Complete Mini-GPT Usage Example
# ================================

import torch
from mini_gpt import GPT, GPTConfig, GPTGenerator

# 1. Load or create model
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Option A: Load pre-trained model
model, tokenizer = load_model("gpt_shakespeare", device=device)

# Option B: Create new model
config = GPTConfig(
    vocab_size=65,      # Character vocabulary
    block_size=256,     # Context length
    n_layer=6,          # Transformer layers
    n_head=6,           # Attention heads
    n_embd=384,         # Embedding dimension
)
model = GPT(config).to(device)

# 2. Create generator
generator = GPTGenerator(model, encode, decode, device)

# 3. Generate text
prompt = "To be, or not to be"

# Simple generation
output = generator.generate(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_k=40,
)
print(output)

# Creative generation
creative = generator.generate(
    prompt,
    max_tokens=200,
    temperature=1.2,
    top_p=0.9,
)

# Focused generation
focused = generator.generate(
    prompt,
    max_tokens=200,
    temperature=0.5,
    repetition_penalty=1.2,
)

# 4. Evaluate model
perplexity, loss = evaluate_perplexity(model, val_data, block_size)
print(f"Perplexity: {perplexity:.2f}")
print(f"Loss: {loss:.4f}")
'''

print(example_code)

# =================================
# 4. FINAL SUMMARY
# =================================

print("\n" + "=" * 60)
print("🎉 CONGRATULATIONS! YOUR MINI-GPT IS COMPLETE! 🎉")
print("=" * 60)

print("""
You have built:
  ✅ Complete GPT architecture from scratch
  ✅ Tokenizer (character-level)
  ✅ Training pipeline with all best practices
  ✅ Multiple generation strategies
  ✅ Model evaluation with perplexity
  ✅ Save/load functionality

Your model can:
  📝 Generate Shakespeare-style text
  🎭 Complete prompts creatively
  📊 Be evaluated quantitatively
  💾 Be saved and loaded

Next steps:
  1. Train on larger datasets (books, code, etc.)
  2. Implement BPE tokenization
  3. Scale up model size
  4. Add instruction fine-tuning
  5. Build a web interface

You now understand how GPT works from the ground up!
This is the same architecture powering ChatGPT, Claude, and other LLMs.
""")
