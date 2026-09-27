# Source: AI_Learning_Cursor lines 12753-13090
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Sentiment Classifier with LSTM
# Title: SENTIMENT CLASSIFICATION WITH LSTM
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
SENTIMENT CLASSIFICATION WITH LSTM
==================================
Building a real NLP application: classifying movie reviews.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. DATA PREPARATION
# =================================

print("=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

# Sample dataset (in practice, use IMDB or similar)
positive_reviews = [
    "This movie was fantastic! I loved every moment of it.",
    "Amazing film with great acting and stunning visuals.",
    "One of the best movies I have ever seen. Highly recommend!",
    "Brilliant story and wonderful performances by the cast.",
    "A masterpiece of cinema. Truly outstanding work.",
    "Incredible movie that kept me engaged from start to finish.",
    "Loved the characters and the plot was so exciting.",
    "Beautiful cinematography and a touching story.",
    "Perfect blend of action and emotion. Five stars!",
    "The director did an amazing job with this film.",
    "Heartwarming and inspiring. A must watch!",
    "Exceeded all my expectations. Absolutely wonderful.",
    "Great soundtrack and memorable scenes throughout.",
    "The acting was superb and the story was captivating.",
    "A delightful movie that the whole family can enjoy.",
]

negative_reviews = [
    "Terrible movie. Complete waste of time and money.",
    "Boring and predictable. I fell asleep halfway through.",
    "The acting was awful and the plot made no sense.",
    "One of the worst films I have ever watched.",
    "Disappointing from start to finish. Would not recommend.",
    "The story was confusing and the characters were flat.",
    "Poor script and even worse direction. Avoid this movie.",
    "I wanted my two hours back after watching this.",
    "Dull and uninspiring. Nothing new or interesting.",
    "The movie failed to engage me at any point.",
    "Waste of talented actors on a terrible script.",
    "Predictable plot with no emotional depth whatsoever.",
    "I struggled to finish this movie. So boring.",
    "The special effects were laughable and the story weak.",
    "A forgettable film that leaves no impression.",
]

# Combine and create labels
all_reviews = positive_reviews + negative_reviews
all_labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

print(f"Total reviews: {len(all_reviews)}")
print(f"Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}")

# =================================
# 2. TOKENIZATION AND VOCABULARY
# =================================

print("\n" + "=" * 60)
print("2. BUILDING VOCABULARY")
print("=" * 60)

def tokenize(text):
    """Simple word tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()


# Build vocabulary
all_words = []
for review in all_reviews:
    all_words.extend(tokenize(review))

word_counts = Counter(all_words)
vocab = ['<PAD>', '<UNK>'] + [w for w, c in word_counts.most_common()]
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print(f"Vocabulary size: {len(vocab)}")
print(f"Most common words: {word_counts.most_common(10)}")


def encode_review(review, max_length=50):
    """Convert review to tensor of word indices."""
    tokens = tokenize(review)[:max_length]
    indices = [word_to_idx.get(t, 1) for t in tokens]  # 1 = <UNK>
    return torch.tensor(indices)


# =================================
# 3. DATASET AND DATALOADER
# =================================

print("\n" + "=" * 60)
print("3. CREATING DATASET")
print("=" * 60)

class SentimentDataset(Dataset):
    def __init__(self, reviews, labels, max_length=50):
        self.reviews = [encode_review(r, max_length) for r in reviews]
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]


def collate_fn(batch):
    """Pad sequences to same length in a batch."""
    reviews, labels = zip(*batch)
    reviews_padded = pad_sequence(reviews, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return reviews_padded, labels


# Split data
split = int(0.8 * len(all_reviews))
train_reviews, test_reviews = all_reviews[:split], all_reviews[split:]
train_labels, test_labels = all_labels[:split], all_labels[split:]

train_dataset = SentimentDataset(train_reviews, train_labels)
test_dataset = SentimentDataset(test_reviews, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# =================================
# 4. LSTM SENTIMENT CLASSIFIER
# =================================

print("\n" + "=" * 60)
print("4. LSTM SENTIMENT MODEL")
print("=" * 60)

class SentimentLSTM(nn.Module):
    """
    LSTM-based sentiment classifier.
    
    Architecture:
    1. Embedding: Words -> Vectors
    2. LSTM: Process sequence
    3. Fully connected: Hidden state -> Sentiment score
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Process sequence in both directions
        )
        
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_length)
        
        # Embed words
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # h_n: (num_layers * 2, batch, hidden) for bidirectional
        
        # Concatenate final forward and backward hidden states
        h_forward = h_n[-2, :, :]  # Last layer, forward
        h_backward = h_n[-1, :, :] # Last layer, backward
        hidden = torch.cat([h_forward, h_backward], dim=1)
        
        # Classify
        output = self.fc(self.dropout(hidden))
        return self.sigmoid(output).squeeze(1)


# Create model
vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 64

model = SentimentLSTM(vocab_size, embed_dim, hidden_dim).to(device)

print(f"Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# =================================
# 5. TRAINING
# =================================

print("\n" + "=" * 60)
print("5. TRAINING")
print("=" * 60)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50
train_losses = []
test_accuracies = []

print("Training sentiment classifier...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    for reviews, labels in train_loader:
        reviews = reviews.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(reviews)
        loss = criterion(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews = reviews.to(device)
            labels = labels.to(device)
            predictions = model(reviews)
            predicted = (predictions >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    test_accuracies.append(accuracy)
    
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2%}")

print("\nTraining complete!")

# =================================
# 6. VISUALIZATION AND TESTING
# =================================

print("\n" + "=" * 60)
print("6. RESULTS")
print("=" * 60)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_accuracies)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Test Accuracy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_training.png', dpi=100)
print("Saved sentiment_training.png")
plt.show()

# Test on new reviews
def predict_sentiment(review):
    """Predict sentiment of a review."""
    model.eval()
    with torch.no_grad():
        encoded = encode_review(review).unsqueeze(0).to(device)
        prediction = model(encoded).item()
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return sentiment, prediction


print("\nTesting on new reviews:")
test_reviews_new = [
    "This movie was absolutely wonderful!",
    "I hated every minute of this film.",
    "Pretty good movie overall, enjoyed it.",
    "Not my cup of tea, quite boring.",
    "The best film of the year!",
]

for review in test_reviews_new:
    sentiment, score = predict_sentiment(review)
    print(f"  '{review[:50]}...'")
    print(f"    -> {sentiment} (score: {score:.3f})")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'word_to_idx': word_to_idx,
}, 'sentiment_model.pth')
print("\nModel saved to sentiment_model.pth")
