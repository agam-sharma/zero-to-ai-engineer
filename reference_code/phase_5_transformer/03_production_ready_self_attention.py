# Source: AI_Learning_Cursor lines 17809-18092
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Production-Ready Self-Attention
# Title: PRODUCTION-READY SELF-ATTENTION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
PRODUCTION-READY SELF-ATTENTION
===============================
A complete implementation ready for use in transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =================================
# 1. COMPLETE SELF-ATTENTION CLASS
# =================================

print("=" * 60)
print("1. COMPLETE SELF-ATTENTION MODULE")
print("=" * 60)

class SelfAttentionComplete(nn.Module):
    """
    Complete Self-Attention module with all features needed for transformers.
    
    Features:
    - Learned Q, K, V projections
    - Output projection
    - Dropout
    - Support for attention masks
    - Returns attention weights for visualization
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model  # Usually d_k = d_model for single-head
        
        # Projection layers
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, 1, seq_len) or (batch, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, seq_len, seq_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.W_O(context)
        
        if return_attention:
            return output, attention_weights
        return output


# Test the module
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
d_model = 64
batch_size = 2
seq_len = 10

attention = SelfAttentionComplete(d_model).to(device)
x = torch.randn(batch_size, seq_len, d_model, device=device)

# Without mask
output = attention(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# With causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
output_masked, weights = attention(x, mask=causal_mask, return_attention=True)
print(f"Masked output shape: {output_masked.shape}")
print(f"Attention weights shape: {weights.shape}")

# Verify causal masking
print(f"\nAttention weights for position 5 (should only see 0-5):")
print(f"  {weights[0, 5, :].detach().cpu().numpy().round(3)}")
print(f"  Positions 6-9 should be 0: {weights[0, 5, 6:].sum().item():.6f}")

# =================================
# 2. TESTING ON A REAL TASK
# =================================

print("\n" + "=" * 60)
print("2. SELF-ATTENTION FOR SEQUENCE CLASSIFICATION")
print("=" * 60)

class AttentionClassifier(nn.Module):
    """
    Simple classifier using self-attention.
    
    Uses attention to aggregate sequence into fixed representation,
    then classifies.
    """
    
    def __init__(self, vocab_size, d_model, num_classes, max_len=100, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = SelfAttentionComplete(d_model, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embed
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        embedded = self.dropout(embedded)
        
        # Self-attention
        attended = self.attention(embedded, mask)  # (batch, seq_len, d_model)
        
        # Pool: mean of attended representations
        pooled = attended.mean(dim=1)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits


# Quick test
vocab_size = 1000
d_model = 64
num_classes = 2

model = AttentionClassifier(vocab_size, d_model, num_classes).to(device)
x = torch.randint(0, vocab_size, (4, 20), device=device)
logits = model(x)

print(f"Input shape: {x.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# =================================
# 3. VISUALIZING LEARNED ATTENTION
# =================================

print("\n" + "=" * 60)
print("3. TRAINING AND VISUALIZING ATTENTION")
print("=" * 60)

# Create a simple task: classify if first token is odd or even
def create_first_token_data(n_samples, seq_len, vocab_size):
    """
    Task: Predict if first token is odd (1) or even (0).
    
    This requires the model to learn to attend to the first position!
    """
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = (X[:, 0] % 2).long()  # Label based on first token
    return X, y


# Generate data
train_X, train_y = create_first_token_data(1000, 20, vocab_size)
test_X, test_y = create_first_token_data(200, 20, vocab_size)

train_X, train_y = train_X.to(device), train_y.to(device)
test_X, test_y = test_X.to(device), test_y.to(device)

# Model with attention weights tracking
class AttentionClassifierWithWeights(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = SelfAttentionComplete(d_model, dropout=0.1)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, return_attention=False):
        embedded = self.embedding(x)
        
        if return_attention:
            attended, weights = self.attention(embedded, return_attention=True)
        else:
            attended = self.attention(embedded)
            weights = None
        
        pooled = attended.mean(dim=1)
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, weights
        return logits


# Train
model = AttentionClassifierWithWeights(vocab_size, d_model, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training attention classifier on 'first token parity' task...")
for epoch in range(20):
    model.train()
    
    # Forward
    logits = model(train_X)
    loss = criterion(logits, train_y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(test_X)
        test_pred = test_logits.argmax(dim=-1)
        accuracy = (test_pred == test_y).float().mean().item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2%}")

# Visualize attention
print("\nVisualizing learned attention patterns...")
model.eval()
with torch.no_grad():
    sample = test_X[0:1]
    logits, attention_weights = model(sample, return_attention=True)

# Average attention across positions (how much does each input position get attended to?)
avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()

plt.figure(figsize=(12, 4))
plt.bar(range(len(avg_attention)), avg_attention)
plt.xlabel('Input Position')
plt.ylabel('Average Attention Weight')
plt.title('Learned Attention: Model should focus on position 0 for this task')
plt.axvline(x=0, color='red', linestyle='--', label='Position 0 (answer is here)')
plt.legend()
plt.savefig('learned_attention.png', dpi=100)
print("Saved learned_attention.png")
plt.show()

print("\nThe model should learn to attend heavily to position 0,")
print("since that's where the answer (odd/even) is determined!")
