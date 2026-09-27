# Source: AI_Learning_Cursor lines 13132-13537
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Seq2Seq Model
# Title: SEQUENCE-TO-SEQUENCE MODELS
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
SEQUENCE-TO-SEQUENCE MODELS
===========================
The foundation for machine translation, summarization, and more.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. SEQ2SEQ THEORY
# =================================

print("=" * 60)
print("1. SEQ2SEQ ARCHITECTURE")
print("=" * 60)

"""
SEQUENCE-TO-SEQUENCE (Seq2Seq)

Used for tasks where input and output are both sequences:
- Machine Translation: English -> French
- Summarization: Long document -> Short summary
- Question Answering: Question -> Answer

Architecture:
    ENCODER: Input sequence -> Context vector (fixed-size representation)
    DECODER: Context vector -> Output sequence

    Input: "Hello world"
           ↓
    [ENCODER LSTM] → Context Vector → [DECODER LSTM]
                                            ↓
                                      "Bonjour monde"

Key insight: The entire input sequence is compressed into one vector.
This is a bottleneck! Attention (Phase 4) solves this.
"""

# =================================
# 2. SIMPLE TASK: REVERSE SEQUENCE
# =================================

print("\n" + "=" * 60)
print("2. TASK: SEQUENCE REVERSAL")
print("=" * 60)

"""
We'll train a seq2seq model to reverse sequences.
Input:  [1, 2, 3, 4, 5]
Output: [5, 4, 3, 2, 1]

This is a good test because:
- Fixed vocabulary (digits 0-9)
- Clear correct answer
- Requires understanding the whole input before producing output
"""

# Data generation
def generate_reverse_data(n_samples, seq_length, vocab_size=10):
    """Generate pairs of (sequence, reversed_sequence)."""
    # Random sequences of digits
    X = torch.randint(1, vocab_size, (n_samples, seq_length))  # 1-9 (0 reserved for padding)
    y = X.flip(dims=[1])  # Reverse
    return X, y


seq_length = 5
vocab_size = 10  # Digits 0-9

X_train, y_train = generate_reverse_data(2000, seq_length, vocab_size)
X_test, y_test = generate_reverse_data(200, seq_length, vocab_size)

print(f"Training samples: {len(X_train)}")
print(f"Sequence length: {seq_length}")
print(f"Sample input:  {X_train[0].tolist()}")
print(f"Sample output: {y_train[0].tolist()}")

# =================================
# 3. ENCODER
# =================================

print("\n" + "=" * 60)
print("3. ENCODER")
print("=" * 60)

class Encoder(nn.Module):
    """
    Encoder: Process input sequence and produce context.
    
    Input sequence -> Embedding -> LSTM -> Final hidden state (context)
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length) input tokens
            
        Returns:
            hidden: (h, c) final hidden states as context
        """
        # Embed
        embedded = self.dropout(self.embedding(x))  # (batch, seq, embed)
        
        # Encode
        outputs, hidden = self.lstm(embedded)
        # hidden: (h, c) each of shape (num_layers, batch, hidden)
        
        return hidden


# =================================
# 4. DECODER
# =================================

print("\n" + "=" * 60)
print("4. DECODER")
print("=" * 60)

class Decoder(nn.Module):
    """
    Decoder: Generate output sequence from context.
    
    At each step:
    1. Take previous output (or start token)
    2. Embed it
    3. Run through LSTM with context
    4. Predict next token
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        """
        Single decoding step.
        
        Args:
            x: (batch_size, 1) current input token
            hidden: (h, c) context from encoder or previous step
            
        Returns:
            output: (batch_size, vocab_size) prediction logits
            hidden: Updated hidden state
        """
        # Embed
        embedded = self.dropout(self.embedding(x))  # (batch, 1, embed)
        
        # Decode
        output, hidden = self.lstm(embedded, hidden)  # (batch, 1, hidden)
        
        # Predict
        prediction = self.fc(output.squeeze(1))  # (batch, vocab)
        
        return prediction, hidden


# =================================
# 5. SEQ2SEQ MODEL
# =================================

print("\n" + "=" * 60)
print("5. SEQ2SEQ MODEL")
print("=" * 60)

class Seq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence model.
    
    Combines encoder and decoder with teacher forcing.
    """
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len) source sequence
            trg: (batch, trg_len) target sequence (for teacher forcing)
            teacher_forcing_ratio: Probability of using true target vs prediction
            
        Returns:
            outputs: (batch, trg_len, vocab_size) predictions
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        
        # Encode source
        hidden = self.encoder(src)
        
        # First decoder input (we'll use the first target token)
        decoder_input = trg[:, 0:1]  # (batch, 1)
        
        for t in range(1, trg_len):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing: use true target or prediction
            teacher_force = random.random() < teacher_forcing_ratio
            
            if teacher_force:
                decoder_input = trg[:, t:t+1]  # True target
            else:
                decoder_input = output.argmax(dim=1, keepdim=True)  # Prediction
        
        return outputs
    
    def predict(self, src, max_len=None):
        """Generate output sequence without teacher forcing."""
        if max_len is None:
            max_len = src.size(1)
        
        batch_size = src.size(0)
        
        # Encode
        hidden = self.encoder(src)
        
        # Start with first source token (or could use special <SOS> token)
        decoder_input = src[:, 0:1]
        
        outputs = []
        for _ in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            predicted = output.argmax(dim=1, keepdim=True)
            outputs.append(predicted)
            decoder_input = predicted
        
        return torch.cat(outputs, dim=1)


# Create model
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers)
decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"Model architecture:")
print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters")
print(f"  Total: {sum(p.numel() for p in model.parameters()):,} parameters")

# =================================
# 6. TRAINING
# =================================

print("\n" + "=" * 60)
print("6. TRAINING")
print("=" * 60)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

n_epochs = 100
batch_size = 64
losses = []
accuracies = []

print("Training seq2seq model...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    # Shuffle
    perm = torch.randperm(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch_X, batch_y, teacher_forcing_ratio=0.5)
        
        # Reshape for loss: (batch * seq, vocab) vs (batch * seq,)
        output_flat = output[:, 1:, :].contiguous().view(-1, vocab_size)
        target_flat = batch_y[:, 1:].contiguous().view(-1)
        
        loss = criterion(output_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (len(X_train) // batch_size)
    losses.append(avg_loss)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X_test)
        correct = (predictions == y_test).all(dim=1).float().mean().item()
        accuracies.append(correct)
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Sequence Accuracy = {correct:.2%}")

print("\nTraining complete!")

# =================================
# 7. VISUALIZATION AND TESTING
# =================================

print("\n" + "=" * 60)
print("7. RESULTS")
print("=" * 60)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(accuracies)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Sequence Accuracy')
axes[1].set_title('Test Accuracy (Full Sequence Match)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seq2seq_training.png', dpi=100)
print("Saved seq2seq_training.png")
plt.show()

# Test examples
print("\nTest predictions:")
model.eval()
with torch.no_grad():
    for i in range(5):
        src = X_test[i:i+1]
        pred = model.predict(src)
        
        src_list = src[0].tolist()
        pred_list = pred[0].tolist()
        target_list = y_test[i].tolist()
        
        correct = "✓" if pred_list == target_list else "✗"
        print(f"  Input:    {src_list}")
        print(f"  Target:   {target_list}")
        print(f"  Predicted: {pred_list} {correct}")
        print()

# Save model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
}, 'seq2seq_model.pth')
print("Model saved to seq2seq_model.pth")
