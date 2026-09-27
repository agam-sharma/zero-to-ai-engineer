# Source: AI_Learning_Cursor lines 10765-11246
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Word2Vec Implementation
# Title: WORD2VEC FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
WORD2VEC FROM SCRATCH
=====================
Implementing the Skip-gram Word2Vec model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. WORD2VEC THEORY
# =================================

print("=" * 60)
print("1. WORD2VEC THEORY")
print("=" * 60)

"""
WORD2VEC: Learning word embeddings from context

Two architectures:
1. CBOW (Continuous Bag of Words): Predict center word from context
2. Skip-gram: Predict context words from center word (we'll implement this)

Skip-gram example:
    Sentence: "The quick brown fox jumps"
    Center word: "brown"
    Context window: 2
    Context words: ["quick", "fox"]
    
    Training pairs: (brown, quick), (brown, fox)
    
    The model learns: "If you see 'brown', you'll likely see 'quick' or 'fox' nearby"
    
    After training:
    - Words appearing in similar contexts have similar embeddings
    - "quick" and "fast" will have similar vectors
    - "king" and "queen" will have similar vectors (both appear with "royal", "throne", etc.)
"""

# =================================
# 2. DATA PREPARATION
# =================================

print("\n" + "=" * 60)
print("2. DATA PREPARATION")
print("=" * 60)

# Sample corpus (in practice, you'd use millions of sentences)
corpus = """
the king sat on his throne in the castle
the queen sat beside the king
a prince is the son of a king and queen
a princess is the daughter of a king and queen
the royal family lived in the castle
the man walked to the market
the woman walked to the store
a boy is a young man
a girl is a young woman
the quick brown fox jumps over the lazy dog
the fast brown fox leaps over the slow dog
machine learning is a field of artificial intelligence
deep learning uses neural networks
neural networks learn patterns from data
transformers revolutionized natural language processing
gpt models can generate human text
language models predict the next word
the cat sat on the mat
the dog lay on the rug
cats and dogs are common pets
""".lower().strip()

# Clean and tokenize
def preprocess(text):
    # Remove punctuation and split
    words = text.replace('\n', ' ').split()
    words = [w for w in words if w.isalpha()]
    return words

words = preprocess(corpus)
print(f"Total words: {len(words)}")
print(f"Sample: {words[:20]}")

# Build vocabulary
word_counts = Counter(words)
vocab = sorted(word_counts.keys())
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

print(f"Vocabulary size: {vocab_size}")
print(f"Most common words: {word_counts.most_common(10)}")

# =================================
# 3. CREATE TRAINING PAIRS
# =================================

print("\n" + "=" * 60)
print("3. CREATING TRAINING PAIRS")
print("=" * 60)

def create_skipgram_pairs(words, word_to_idx, window_size=2):
    """
    Create (center_word, context_word) pairs for Skip-gram training.
    """
    pairs = []
    
    for i, center_word in enumerate(words):
        center_idx = word_to_idx.get(center_word)
        if center_idx is None:
            continue
        
        # Get context words within window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:  # Skip the center word itself
                context_word = words[j]
                context_idx = word_to_idx.get(context_word)
                if context_idx is not None:
                    pairs.append((center_idx, context_idx))
    
    return pairs


window_size = 2
pairs = create_skipgram_pairs(words, word_to_idx, window_size)

print(f"Window size: {window_size}")
print(f"Total training pairs: {len(pairs)}")
print(f"\nSample pairs:")
for i in range(min(10, len(pairs))):
    center_idx, context_idx = pairs[i]
    print(f"  ({idx_to_word[center_idx]}, {idx_to_word[context_idx]})")

# =================================
# 4. DATASET AND DATALOADER
# =================================

class Word2VecDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)


dataset = Word2VecDataset(pairs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =================================
# 5. SKIP-GRAM MODEL
# =================================

print("\n" + "=" * 60)
print("5. SKIP-GRAM MODEL")
print("=" * 60)

class SkipGram(nn.Module):
    """
    Skip-gram Word2Vec model.
    
    Architecture:
    1. Embedding layer for center words
    2. Embedding layer for context words (output embeddings)
    3. Dot product to compute similarity
    
    The center word embeddings are what we use as word vectors.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # Input embeddings (the ones we'll use as word vectors)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output embeddings (used for prediction)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_words, context_words):
        """
        Compute similarity scores between center and context words.
        
        Args:
            center_words: (batch_size,) tensor of center word indices
            context_words: (batch_size,) tensor of context word indices
            
        Returns:
            scores: (batch_size,) tensor of similarity scores
        """
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)   # (batch, embed_dim)
        context_embeds = self.context_embeddings(context_words) # (batch, embed_dim)
        
        # Compute dot product (similarity)
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        
        return scores
    
    def get_word_vector(self, word_idx):
        """Get the embedding vector for a word."""
        return self.center_embeddings.weight[word_idx].detach()


embedding_dim = 50
model = SkipGram(vocab_size, embedding_dim).to(device)

print(f"Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# =================================
# 6. TRAINING WITH NEGATIVE SAMPLING
# =================================

print("\n" + "=" * 60)
print("6. TRAINING")
print("=" * 60)

"""
NEGATIVE SAMPLING:

Instead of using softmax over entire vocabulary (expensive!),
we sample a few "negative" (random) words and train the model to:
- Give HIGH score to real (center, context) pairs
- Give LOW score to fake (center, random) pairs

This is much faster and works just as well.
"""

class NegativeSamplingLoss(nn.Module):
    """Binary cross-entropy loss with negative sampling."""
    
    def __init__(self, num_negatives=5):
        super().__init__()
        self.num_negatives = num_negatives
    
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Scores for positive (real) pairs
            neg_scores: Scores for negative (fake) pairs
        """
        # Positive pairs should have high scores (label = 1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
        
        # Negative pairs should have low scores (label = 0)
        neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-10).mean()
        
        return pos_loss + neg_loss


def get_negative_samples(batch_size, num_negatives, vocab_size, device):
    """Sample random negative context words."""
    return torch.randint(0, vocab_size, (batch_size, num_negatives), device=device)


# Training setup
num_negatives = 5
criterion = NegativeSamplingLoss(num_negatives)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
losses = []

print(f"Training for {n_epochs} epochs...")
print(f"Negative samples per positive: {num_negatives}")

for epoch in range(n_epochs):
    total_loss = 0
    
    for center_words, context_words in dataloader:
        center_words = center_words.to(device)
        context_words = context_words.to(device)
        batch_size = center_words.size(0)
        
        # Positive scores
        pos_scores = model(center_words, context_words)
        
        # Negative samples
        neg_words = get_negative_samples(batch_size, num_negatives, vocab_size, device)
        neg_scores = []
        for i in range(num_negatives):
            neg_score = model(center_words, neg_words[:, i])
            neg_scores.append(neg_score)
        neg_scores = torch.stack(neg_scores, dim=1)  # (batch, num_neg)
        
        # Compute loss
        loss = criterion(pos_scores, neg_scores.mean(dim=1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}")

print("\nTraining complete!")

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Word2Vec Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('word2vec_loss.png', dpi=100)
print("Saved word2vec_loss.png")
plt.show()

# =================================
# 7. EXPLORING LEARNED EMBEDDINGS
# =================================

print("\n" + "=" * 60)
print("7. EXPLORING LEARNED EMBEDDINGS")
print("=" * 60)

def get_embedding(word):
    """Get embedding vector for a word."""
    if word not in word_to_idx:
        print(f"Word '{word}' not in vocabulary")
        return None
    idx = word_to_idx[word]
    return model.center_embeddings.weight[idx].detach().cpu().numpy()


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_similar_words(word, top_n=5):
    """Find most similar words to a given word."""
    if word not in word_to_idx:
        print(f"Word '{word}' not in vocabulary")
        return []
    
    word_vec = get_embedding(word)
    similarities = []
    
    for other_word in vocab:
        if other_word != word:
            other_vec = get_embedding(other_word)
            sim = cosine_similarity(word_vec, other_vec)
            similarities.append((other_word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


# Test word similarities
test_words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'learning', 'neural']

print("Most similar words:")
for word in test_words:
    if word in word_to_idx:
        similar = find_similar_words(word, top_n=3)
        similar_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
        print(f"  {word}: {similar_str}")

# Word analogy (king - man + woman = ?)
print("\nWord analogy: king - man + woman = ?")
if all(w in word_to_idx for w in ['king', 'man', 'woman']):
    king_vec = get_embedding('king')
    man_vec = get_embedding('man')
    woman_vec = get_embedding('woman')
    
    result_vec = king_vec - man_vec + woman_vec
    
    # Find closest word
    best_word = None
    best_sim = -1
    for word in vocab:
        if word not in ['king', 'man', 'woman']:
            vec = get_embedding(word)
            sim = cosine_similarity(result_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_word = word
    
    print(f"  Result: {best_word} (similarity: {best_sim:.3f})")

# =================================
# 8. VISUALIZE EMBEDDINGS
# =================================

print("\n" + "=" * 60)
print("8. VISUALIZING EMBEDDINGS")
print("=" * 60)

# Get all embeddings
all_embeddings = model.center_embeddings.weight.detach().cpu().numpy()

# Use t-SNE to reduce to 2D
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, vocab_size-1))
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(14, 10))

# Color words by category
categories = {
    'royalty': ['king', 'queen', 'prince', 'princess', 'royal', 'throne', 'castle'],
    'people': ['man', 'woman', 'boy', 'girl', 'son', 'daughter', 'family'],
    'animals': ['fox', 'dog', 'cat', 'pets'],
    'ml': ['learning', 'neural', 'networks', 'deep', 'machine', 'models', 'data'],
}

colors = {'royalty': 'purple', 'people': 'blue', 'animals': 'green', 'ml': 'red', 'other': 'gray'}

for i, word in enumerate(vocab):
    # Find category
    cat = 'other'
    for category, words in categories.items():
        if word in words:
            cat = category
            break
    
    color = colors[cat]
    alpha = 0.8 if cat != 'other' else 0.3
    
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=color, alpha=alpha, s=50)
    
    # Label important words
    if cat != 'other':
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=8, alpha=0.8)

# Add legend
for cat, color in colors.items():
    if cat != 'other':
        plt.scatter([], [], c=color, label=cat.capitalize())

plt.legend()
plt.title('Word2Vec Embeddings (t-SNE Visualization)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.savefig('word2vec_visualization.png', dpi=100)
print("Saved word2vec_visualization.png")
plt.show()

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'word_to_idx': word_to_idx,
    'idx_to_word': idx_to_word,
    'embedding_dim': embedding_dim,
}, 'word2vec_model.pth')
print("\nModel saved to word2vec_model.pth")
