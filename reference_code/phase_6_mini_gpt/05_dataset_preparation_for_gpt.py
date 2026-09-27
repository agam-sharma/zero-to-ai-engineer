# Source: AI_Learning_Cursor lines 24087-24318
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Dataset Preparation
# Title: PREPARING DATA FOR GPT TRAINING
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
PREPARING DATA FOR GPT TRAINING
===============================
Creating datasets from raw text.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests

# =================================
# 1. DOWNLOADING TEXT DATA
# =================================

print("=" * 60)
print("1. DOWNLOADING TEXT DATA")
print("=" * 60)

def download_shakespeare():
    """Download the complete works of Shakespeare."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if os.path.exists('shakespeare.txt'):
        print("Shakespeare data already exists.")
        with open('shakespeare.txt', 'r') as f:
            return f.read()
    
    print("Downloading Shakespeare...")
    response = requests.get(url)
    text = response.text
    
    with open('shakespeare.txt', 'w') as f:
        f.write(text)
    
    print(f"Downloaded {len(text):,} characters")
    return text


try:
    shakespeare = download_shakespeare()
    print(f"\nShakespeare dataset:")
    print(f"  Total characters: {len(shakespeare):,}")
    print(f"  First 500 characters:\n")
    print(shakespeare[:500])
except Exception as e:
    print(f"Could not download Shakespeare: {e}")
    # Use a fallback text
    shakespeare = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""" * 100  # Repeat to have more data
    print("Using fallback Shakespeare text")

# =================================
# 2. CHARACTER-LEVEL TOKENIZATION
# =================================

print("\n" + "=" * 60)
print("2. CHARACTER-LEVEL TOKENIZATION")
print("=" * 60)

"""
For our mini-GPT, we'll use character-level tokenization.
This is simpler and works well for Shakespeare-like text.

GPT-2/3/4 use BPE, but character-level is:
- Easier to implement
- Good for learning
- Actually used in some models (like character RNNs)
"""

class CharacterTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        """Build vocabulary from text."""
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars[:50])}...")
        
        return self
    
    def encode(self, text):
        """Encode text to token ids."""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, ids):
        """Decode token ids to text."""
        return ''.join([self.idx_to_char[i] for i in ids])


tokenizer = CharacterTokenizer()
tokenizer.fit(shakespeare)

# Test
test_text = "To be, or not to be"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"\nTest:")
print(f"  Original: '{test_text}'")
print(f"  Encoded: {encoded}")
print(f"  Decoded: '{decoded}'")

# =================================
# 3. GPT DATASET
# =================================

print("\n" + "=" * 60)
print("3. GPT DATASET")
print("=" * 60)

class GPTDataset(Dataset):
    """
    Dataset for GPT training.
    
    Each sample is a sequence of tokens.
    Input: tokens[:-1]
    Target: tokens[1:]
    
    This implements the "predict next token" objective.
    """
    
    def __init__(self, text, tokenizer, block_size):
        """
        Args:
            text: Raw text data
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Encode entire text
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"Dataset created:")
        print(f"  Total tokens: {len(self.data):,}")
        print(f"  Block size: {block_size}")
        print(f"  Number of samples: {len(self):,}")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get block_size + 1 tokens
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Input is first block_size tokens
        x = chunk[:-1]
        
        # Target is last block_size tokens (shifted by 1)
        y = chunk[1:]
        
        return x, y


# Create dataset
block_size = 128  # Context length
dataset = GPTDataset(shakespeare, tokenizer, block_size)

# Show sample
x, y = dataset[0]
print(f"\nSample from dataset:")
print(f"  Input shape: {x.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Input text: '{tokenizer.decode(x[:50].tolist())}'...")
print(f"  Target text: '{tokenizer.decode(y[:50].tolist())}'...")

# =================================
# 4. TRAIN/VAL SPLIT
# =================================

print("\n" + "=" * 60)
print("4. TRAIN/VAL SPLIT")
print("=" * 60)

def create_datasets(text, tokenizer, block_size, train_ratio=0.9):
    """Create train and validation datasets."""
    
    # Split text
    n = len(text)
    train_text = text[:int(n * train_ratio)]
    val_text = text[int(n * train_ratio):]
    
    train_dataset = GPTDataset(train_text, tokenizer, block_size)
    val_dataset = GPTDataset(val_text, tokenizer, block_size)
    
    return train_dataset, val_dataset


train_dataset, val_dataset = create_datasets(shakespeare, tokenizer, block_size)

print(f"\nDataset split:")
print(f"  Training samples: {len(train_dataset):,}")
print(f"  Validation samples: {len(val_dataset):,}")

# Create dataloaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test batch
x_batch, y_batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  x: {x_batch.shape}")
print(f"  y: {y_batch.shape}")
