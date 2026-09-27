# Source: AI_Learning_Cursor lines 11254-11516
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Loading and Using GloVe
# Title: USING PRE-TRAINED EMBEDDINGS (GloVe)
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
USING PRE-TRAINED EMBEDDINGS (GloVe)
====================================
Loading and using pre-trained word vectors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import os
import urllib.request
import zipfile

# =================================
# 1. DOWNLOAD GLOVE
# =================================

print("=" * 60)
print("1. LOADING GLOVE EMBEDDINGS")
print("=" * 60)

"""
GloVe (Global Vectors for Word Representation)

Pre-trained on massive text corpora:
- Wikipedia
- Common Crawl
- Twitter

Available dimensions: 50, 100, 200, 300

We'll use GloVe 6B (trained on 6 billion tokens, 400K vocabulary)
"""

def load_glove_embeddings(glove_path: str, embedding_dim: int = 100) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from file.
    
    Returns dictionary: word -> embedding vector
    """
    embeddings = {}
    
    filename = f"glove.6B.{embedding_dim}d.txt"
    filepath = os.path.join(glove_path, filename)
    
    if not os.path.exists(filepath):
        print(f"GloVe file not found at {filepath}")
        print("Please download from: https://nlp.stanford.edu/projects/glove/")
        print("Or use the download code below.")
        return None
    
    print(f"Loading GloVe from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings


# For this example, let's create a simple simulated GloVe
# In practice, you would download the real GloVe files

print("Creating simulated embeddings for demonstration...")

# Simulated embeddings (in practice, load real GloVe)
simulated_glove = {}
np.random.seed(42)

# Create embeddings with semantic structure
base_vectors = {
    'king': np.array([0.5, 0.8, 0.1, -0.2]),
    'queen': np.array([0.5, 0.75, 0.15, -0.25]),
    'man': np.array([0.3, 0.2, 0.1, 0.1]),
    'woman': np.array([0.3, 0.15, 0.15, 0.05]),
    'prince': np.array([0.45, 0.7, 0.08, -0.18]),
    'princess': np.array([0.45, 0.65, 0.12, -0.22]),
    'dog': np.array([-0.3, 0.1, 0.6, 0.2]),
    'cat': np.array([-0.25, 0.05, 0.65, 0.25]),
    'happy': np.array([0.1, 0.2, 0.3, 0.8]),
    'sad': np.array([0.1, 0.2, 0.3, -0.8]),
    'good': np.array([0.15, 0.25, 0.35, 0.75]),
    'bad': np.array([0.15, 0.25, 0.35, -0.75]),
}

embedding_dim = 100
for word in base_vectors:
    # Expand to full dimension
    full_vec = np.random.randn(embedding_dim) * 0.1
    full_vec[:4] = base_vectors[word]  # Set first 4 dims to our structure
    simulated_glove[word] = full_vec

# Add more random words
common_words = ['the', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these']

for word in common_words:
    if word not in simulated_glove:
        simulated_glove[word] = np.random.randn(embedding_dim) * 0.1

print(f"Created {len(simulated_glove)} simulated embeddings")

# =================================
# 2. CREATE EMBEDDING LAYER FROM GLOVE
# =================================

print("\n" + "=" * 60)
print("2. CREATING EMBEDDING LAYER FROM PRE-TRAINED VECTORS")
print("=" * 60)

class PretrainedEmbedding(nn.Module):
    """
    Embedding layer initialized with pre-trained vectors.
    
    This is how you use GloVe/Word2Vec in PyTorch models.
    """
    
    def __init__(self, pretrained_embeddings: Dict[str, np.ndarray], 
                 freeze: bool = True):
        """
        Args:
            pretrained_embeddings: Dictionary of word -> vector
            freeze: If True, don't update embeddings during training
        """
        super().__init__()
        
        # Build vocabulary
        self.word_to_idx = {word: i for i, word in enumerate(pretrained_embeddings.keys())}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        # Get embedding dimension
        first_word = list(pretrained_embeddings.keys())[0]
        self.embedding_dim = len(pretrained_embeddings[first_word])
        
        # Create weight matrix
        weights = np.zeros((self.vocab_size, self.embedding_dim))
        for word, idx in self.word_to_idx.items():
            weights[idx] = pretrained_embeddings[word]
        
        # Create embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data = torch.from_numpy(weights).float()
        
        # Freeze if requested
        if freeze:
            self.embedding.weight.requires_grad = False
        
        print(f"Created embedding layer:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Frozen: {freeze}")
    
    def forward(self, word_indices):
        return self.embedding(word_indices)
    
    def get_word_vector(self, word: str) -> torch.Tensor:
        if word not in self.word_to_idx:
            return None
        idx = self.word_to_idx[word]
        return self.embedding.weight[idx]
    
    def find_similar(self, word: str, top_n: int = 5):
        """Find most similar words."""
        vec = self.get_word_vector(word)
        if vec is None:
            return []
        
        # Compute cosine similarity with all words
        all_vecs = self.embedding.weight
        similarities = torch.nn.functional.cosine_similarity(
            vec.unsqueeze(0), all_vecs, dim=1
        )
        
        # Get top-N (excluding the word itself)
        values, indices = similarities.topk(top_n + 1)
        
        results = []
        for val, idx in zip(values, indices):
            w = self.idx_to_word[idx.item()]
            if w != word:
                results.append((w, val.item()))
        
        return results[:top_n]


# Create embedding layer
pretrained = PretrainedEmbedding(simulated_glove, freeze=True)

# Test
print("\nTesting pre-trained embeddings:")
for word in ['king', 'queen', 'dog', 'happy']:
    if word in pretrained.word_to_idx:
        similar = pretrained.find_similar(word, top_n=3)
        print(f"  Similar to '{word}': {similar}")

# =================================
# 3. WORD ANALOGY WITH PRE-TRAINED EMBEDDINGS
# =================================

print("\n" + "=" * 60)
print("3. WORD ANALOGIES")
print("=" * 60)

def word_analogy(embedding_layer, word_a, word_b, word_c):
    """
    Solve: word_a is to word_b as word_c is to ???
    
    Example: king is to queen as man is to ??? (woman)
    Formula: result = word_b - word_a + word_c
    """
    vec_a = embedding_layer.get_word_vector(word_a)
    vec_b = embedding_layer.get_word_vector(word_b)
    vec_c = embedding_layer.get_word_vector(word_c)
    
    if any(v is None for v in [vec_a, vec_b, vec_c]):
        return None
    
    # Compute analogy vector
    result_vec = vec_b - vec_a + vec_c
    
    # Find closest word
    all_vecs = embedding_layer.embedding.weight
    similarities = torch.nn.functional.cosine_similarity(
        result_vec.unsqueeze(0), all_vecs, dim=1
    )
    
    # Exclude input words
    exclude = {word_a, word_b, word_c}
    
    best_word = None
    best_sim = -1
    
    for idx, sim in enumerate(similarities):
        word = embedding_layer.idx_to_word[idx]
        if word not in exclude and sim.item() > best_sim:
            best_sim = sim.item()
            best_word = word
    
    return best_word, best_sim


# Test analogies
analogies = [
    ('king', 'queen', 'man'),      # king:queen :: man:???
    ('man', 'woman', 'king'),      # man:woman :: king:???
    ('good', 'bad', 'happy'),      # good:bad :: happy:???
]

print("Word analogies:")
for a, b, c in analogies:
    result = word_analogy(pretrained, a, b, c)
    if result:
        word, sim = result
        print(f"  {a} : {b} :: {c} : {word} (similarity: {sim:.3f})")
