# Source: AI_Learning_Cursor lines 23803-24079
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Tokenization from Scratch
# Title: TOKENIZATION FOR GPT
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TOKENIZATION FOR GPT
====================
Converting text to tokens that GPT can process.
"""

import re
from collections import Counter, defaultdict
import json

# =================================
# 1. WHY TOKENIZATION MATTERS
# =================================

print("=" * 60)
print("1. WHY TOKENIZATION MATTERS")
print("=" * 60)

"""
TOKENIZATION: Breaking text into pieces the model can understand

Options:
1. CHARACTER-LEVEL: "hello" -> ['h', 'e', 'l', 'l', 'o']
   - Pros: Small vocabulary (~100)
   - Cons: Very long sequences, hard to learn

2. WORD-LEVEL: "hello world" -> ['hello', 'world']
   - Pros: Meaningful units
   - Cons: Huge vocabulary, can't handle new words

3. SUBWORD-LEVEL (BPE): "hello" -> ['hel', 'lo'] or ['hello']
   - Pros: Balance between character and word
   - Cons: More complex to implement
   
GPT uses BPE (Byte Pair Encoding):
- Common words are single tokens: "the" = [1234]
- Rare words are split: "tokenization" = ["token", "ization"]
- Unknown words are still representable

GPT-2 vocabulary: 50,257 tokens
GPT-4: ~100,000 tokens
"""

# Example: Different tokenization strategies
text = "Hello, I am learning tokenization!"

# Character level
char_tokens = list(text)
print(f"Character-level ({len(char_tokens)} tokens):")
print(f"  {char_tokens}")

# Word level (simple split)
word_tokens = text.split()
print(f"\nWord-level ({len(word_tokens)} tokens):")
print(f"  {word_tokens}")

# Subword level (simulated)
subword_tokens = ["Hello", ",", "I", "am", "learn", "ing", "token", "ization", "!"]
print(f"\nSubword-level ({len(subword_tokens)} tokens):")
print(f"  {subword_tokens}")

# =================================
# 2. BYTE PAIR ENCODING (BPE) FROM SCRATCH
# =================================

print("\n" + "=" * 60)
print("2. BPE FROM SCRATCH")
print("=" * 60)

"""
BPE ALGORITHM:

1. Start with character-level vocabulary
2. Count frequency of adjacent pairs
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

Example:
    "low lower lowest" -> ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ...]
    Most frequent pair: ('l', 'o') -> merge into 'lo'
    "low lower lowest" -> ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ...]
    Most frequent pair: ('lo', 'w') -> merge into 'low'
    ...
"""

class SimpleBPE:
    """
    Simple BPE tokenizer implementation.
    
    This is educational - real BPE is more optimized.
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+""")
    
    def _get_stats(self, tokens_list):
        """Count frequency of adjacent pairs."""
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs
    
    def _merge(self, tokens_list, pair, new_token):
        """Merge all occurrences of pair into new_token."""
        new_tokens_list = []
        for tokens in tokens_list:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens_list.append(new_tokens)
        return new_tokens_list
    
    def train(self, text, verbose=True):
        """Train BPE on text."""
        # Pre-tokenize into words
        words = self.pattern.findall(text)
        
        # Start with characters
        tokens_list = [list(word) for word in words]
        
        # Initialize vocabulary with characters
        vocab = set()
        for tokens in tokens_list:
            vocab.update(tokens)
        
        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")
            print(f"Initial tokens sample: {tokens_list[0][:10]}")
        
        # Merge until vocab_size reached
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            # Get pair frequencies
            pairs = self._get_stats(tokens_list)
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            
            # Merge
            tokens_list = self._merge(tokens_list, best_pair, new_token)
            
            # Update vocabulary
            vocab.add(new_token)
            self.merges.append((best_pair, new_token))
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}: {best_pair} -> '{new_token}' (freq: {pairs[best_pair]})")
        
        # Create final vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        
        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
        
        return self.vocab
    
    def encode(self, text):
        """Encode text to token ids."""
        # Pre-tokenize
        words = self.pattern.findall(text)
        
        # Start with characters
        tokens_list = [list(word) for word in words]
        
        # Apply learned merges
        for (pair, new_token) in self.merges:
            tokens_list = self._merge(tokens_list, pair, new_token)
        
        # Flatten and convert to ids
        all_tokens = []
        for tokens in tokens_list:
            all_tokens.extend(tokens)
        
        ids = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in all_tokens]
        
        return ids
    
    def decode(self, ids):
        """Decode token ids to text."""
        tokens = [self.inv_vocab.get(idx, '<UNK>') for idx in ids]
        return ''.join(tokens)


# Train BPE
print("\nTraining BPE tokenizer...")

sample_text = """
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with many layers.
Natural language processing enables computers to understand human language.
The transformer architecture revolutionized NLP.
GPT models can generate human-like text.
Training large language models requires significant computational resources.
Tokenization is the process of converting text into tokens.
Byte pair encoding is a popular subword tokenization algorithm.
"""

bpe = SimpleBPE(vocab_size=200)
vocab = bpe.train(sample_text)

# Test encoding/decoding
test_text = "Machine learning is amazing!"
encoded = bpe.encode(test_text)
decoded = bpe.decode(encoded)

print(f"\nTest encoding:")
print(f"  Original: '{test_text}'")
print(f"  Encoded: {encoded}")
print(f"  Decoded: '{decoded}'")

# Show vocabulary sample
print(f"\nVocabulary sample (first 30 tokens):")
for i, (token, idx) in enumerate(sorted(bpe.vocab.items(), key=lambda x: x[1])[:30]):
    print(f"  {idx:3d}: '{token}'")

# =================================
# 3. USING TIKTOKEN (GPT'S TOKENIZER)
# =================================

print("\n" + "=" * 60)
print("3. USING TIKTOKEN (GPT'S TOKENIZER)")
print("=" * 60)

"""
tiktoken is OpenAI's official tokenizer for GPT models.
It's highly optimized and uses the same encoding as GPT-3/4.
"""

try:
    import tiktoken
    
    # Get GPT-2 encoding
    enc = tiktoken.get_encoding("gpt2")
    
    test_texts = [
        "Hello, world!",
        "The quick brown fox",
        "Machine learning is amazing",
        "I love artificial intelligence!",
        "Supercalifragilisticexpialidocious",  # Long rare word
    ]
    
    print("tiktoken (GPT-2 encoding) examples:")
    for text in test_texts:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        print(f"  '{text}'")
        print(f"    Tokens: {tokens}")
        print(f"    Count: {len(tokens)}")
        print()
    
    print(f"GPT-2 vocabulary size: {enc.n_vocab}")
    
except ImportError:
    print("tiktoken not installed. Install with: pip install tiktoken")
    print("Skipping tiktoken examples.")
