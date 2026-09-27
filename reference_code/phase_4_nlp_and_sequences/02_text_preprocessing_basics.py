# Source: AI_Learning_Cursor lines 10463-10746
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Text Preprocessing and Tokenization
# Title: TEXT PREPROCESSING AND TOKENIZATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TEXT PREPROCESSING AND TOKENIZATION
====================================
Converting raw text into numbers that neural networks can process.
"""

import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple

# =================================
# 1. BASIC TEXT CLEANING
# =================================

print("=" * 60)
print("1. TEXT CLEANING")
print("=" * 60)

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters
    3. Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


# Test
sample_text = "Hello, World! This is an EXAMPLE text... with 123 numbers!!!"
cleaned = clean_text(sample_text)
print(f"Original: {sample_text}")
print(f"Cleaned:  {cleaned}")

# =================================
# 2. TOKENIZATION
# =================================

print("\n" + "=" * 60)
print("2. TOKENIZATION")
print("=" * 60)

"""
TOKENIZATION: Breaking text into smaller units (tokens)

Types of tokenization:
1. Word-level: "Hello world" -> ["Hello", "world"]
2. Character-level: "Hello" -> ["H", "e", "l", "l", "o"]
3. Subword-level: "unhappiness" -> ["un", "happi", "ness"] (used in GPT!)
"""

class SimpleTokenizer:
    """
    A word-level tokenizer built from scratch.
    
    This is similar to what happens inside GPT's tokenizer,
    but simpler (GPT uses subword tokenization).
    """
    
    def __init__(self, min_freq: int = 1):
        """
        Initialize tokenizer.
        
        Args:
            min_freq: Minimum frequency for a word to be included
        """
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.pad_token = "<PAD>"    # Padding for batch processing
        self.unk_token = "<UNK>"    # Unknown words
        self.bos_token = "<BOS>"    # Beginning of sequence
        self.eos_token = "<EOS>"    # End of sequence
    
    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        This is like "training" the tokenizer on your data.
        """
        # Count word frequencies
        for text in texts:
            words = clean_text(text).split()
            self.word_freq.update(words)
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        # Add words that meet frequency threshold
        idx = len(special_tokens)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Unique words: {len(self.word_freq)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to list of token IDs.
        """
        words = clean_text(text).split()
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.bos_token])
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx[self.unk_token])
        
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.eos_token])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        """
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        words = []
        
        for idx in tokens:
            word = self.idx_to_word.get(idx, self.unk_token)
            if skip_special_tokens and word in special:
                continue
            words.append(word)
        
        return ' '.join(words)
    
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)


# Test the tokenizer
print("\nBuilding tokenizer...")

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown dog outpaces a lazy fox.",
    "The dog and the fox are both quick.",
    "Machine learning is transforming the world.",
    "Deep learning enables amazing applications.",
    "Neural networks learn from data.",
    "The transformer architecture revolutionized NLP.",
    "GPT models generate human-like text.",
]

tokenizer = SimpleTokenizer(min_freq=1)
tokenizer.fit(corpus)

# Test encoding and decoding
test_sentence = "The quick fox learns from data."
encoded = tokenizer.encode(test_sentence)
decoded = tokenizer.decode(encoded)

print(f"\nOriginal:  {test_sentence}")
print(f"Encoded:   {encoded}")
print(f"Decoded:   {decoded}")

# Show vocabulary sample
print(f"\nVocabulary sample (first 15 words):")
for idx in range(min(15, tokenizer.vocab_size)):
    word = tokenizer.idx_to_word[idx]
    print(f"  {idx}: '{word}'")

# =================================
# 3. ONE-HOT ENCODING
# =================================

print("\n" + "=" * 60)
print("3. ONE-HOT ENCODING")
print("=" * 60)

"""
ONE-HOT ENCODING: Simplest word representation

Each word is a vector of length V (vocabulary size).
Only one position is 1, rest are 0.

Example with V=5:
- "cat" = [1, 0, 0, 0, 0]
- "dog" = [0, 1, 0, 0, 0]
- "fish" = [0, 0, 1, 0, 0]

Problem: No relationship between words!
"cat" and "dog" are just as different as "cat" and "quantum physics"
"""

def one_hot_encode(token_ids: List[int], vocab_size: int) -> np.ndarray:
    """Convert token IDs to one-hot vectors."""
    one_hot = np.zeros((len(token_ids), vocab_size))
    for i, idx in enumerate(token_ids):
        one_hot[i, idx] = 1
    return one_hot


# Example
sample_ids = [4, 5, 6]  # Some token IDs
one_hot = one_hot_encode(sample_ids, vocab_size=10)

print(f"Token IDs: {sample_ids}")
print(f"One-hot shape: {one_hot.shape}")
print(f"One-hot matrix:\n{one_hot}")

print("\nProblems with one-hot encoding:")
print("1. Huge vectors (vocab_size can be 50,000+)")
print("2. All words are equally different (no similarity)")
print("3. No semantic meaning captured")
print("\nSolution: Word Embeddings!")

# =================================
# 4. WORD EMBEDDINGS CONCEPT
# =================================

print("\n" + "=" * 60)
print("4. WORD EMBEDDINGS CONCEPT")
print("=" * 60)

"""
WORD EMBEDDINGS: Dense, learned representations

Instead of sparse one-hot vectors, we use dense vectors of fixed size.
These are LEARNED from data to capture semantic relationships.

Example (embedding_dim=4):
- "king"  = [0.2, 0.8, -0.3, 0.5]
- "queen" = [0.2, 0.7, -0.3, 0.6]  # Similar to king!
- "apple" = [-0.5, 0.1, 0.8, -0.2] # Different

Famous property: king - man + woman ≈ queen
"""

import torch
import torch.nn as nn

# Create an embedding layer
vocab_size = tokenizer.vocab_size
embedding_dim = 64  # Each word becomes a 64-dimensional vector

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Embedding weight shape: {embedding_layer.weight.shape}")
print(f"Total parameters: {vocab_size * embedding_dim:,}")

# Get embeddings for a sentence
sentence = "the quick fox"
token_ids = tokenizer.encode(sentence, add_special_tokens=False)
token_tensor = torch.tensor(token_ids)

embeddings = embedding_layer(token_tensor)

print(f"\nSentence: '{sentence}'")
print(f"Token IDs: {token_ids}")
print(f"Embeddings shape: {embeddings.shape}")  # (3, 64)
print(f"First word embedding (first 10 dims): {embeddings[0, :10].detach().numpy()}")
