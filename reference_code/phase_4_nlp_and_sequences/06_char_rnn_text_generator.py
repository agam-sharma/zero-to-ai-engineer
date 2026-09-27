# Source: AI_Learning_Cursor lines 11833-12153
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Text Generator RNN
# Title: CHARACTER-LEVEL TEXT GENERATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
CHARACTER-LEVEL TEXT GENERATION
===============================
Building an RNN that generates text one character at a time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. PREPARE TEXT DATA
# =================================

print("=" * 60)
print("1. PREPARING TEXT DATA")
print("=" * 60)

# Sample text (you can use any text file)
text = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be that is the question.
All that glitters is not gold.
The only thing we have to fear is fear itself.
In the beginning was the word and the word was with god.
Ask not what your country can do for you but what you can do for your country.
I think therefore I am.
The unexamined life is not worth living.
Knowledge is power but enthusiasm pulls the switch.
Machine learning is transforming the world one model at a time.
Neural networks can learn complex patterns from data.
Deep learning enables computers to understand images and language.
The transformer architecture revolutionized natural language processing.
Attention is all you need to build powerful language models.
""".lower().strip()

# Create character vocabulary
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(chars)

print(f"Text length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Encode entire text
encoded_text = [char_to_idx[ch] for ch in text]

# =================================
# 2. CREATE DATASET
# =================================

print("\n" + "=" * 60)
print("2. CREATING DATASET")
print("=" * 60)

class CharDataset(Dataset):
    """
    Dataset for character-level language modeling.
    
    Given a sequence of characters, predict the next character.
    """
    
    def __init__(self, encoded_text, seq_length):
        self.encoded_text = encoded_text
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.encoded_text) - self.seq_length
    
    def __getitem__(self, idx):
        # Input: seq_length characters
        x = torch.tensor(self.encoded_text[idx:idx + self.seq_length])
        # Target: next seq_length characters (shifted by 1)
        y = torch.tensor(self.encoded_text[idx + 1:idx + self.seq_length + 1])
        return x, y


seq_length = 50
dataset = CharDataset(encoded_text, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Sequence length: {seq_length}")
print(f"Dataset size: {len(dataset)} sequences")

# Show sample
x, y = dataset[0]
print(f"\nSample input:  '{''.join([idx_to_char[i.item()] for i in x])}'")
print(f"Sample target: '{''.join([idx_to_char[i.item()] for i in y])}'")

# =================================
# 3. CHARACTER RNN MODEL
# =================================

print("\n" + "=" * 60)
print("3. CHARACTER RNN MODEL")
print("=" * 60)

class CharRNN(nn.Module):
    """
    Character-level RNN for text generation.
    
    Architecture:
    1. Embedding layer: Convert char indices to vectors
    2. RNN layers: Process sequence
    3. Linear layer: Predict next character
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN (using LSTM for better gradient flow)
        self.rnn = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_length) character indices
            hidden: Previous hidden state (optional)
            
        Returns:
            output: (batch_size, seq_length, vocab_size) logits
            hidden: New hidden state
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch, seq, embed_size)
        embedded = self.dropout(embedded)
        
        # RNN forward
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch, seq, hidden)
        
        # Predict next character
        output = self.fc(self.dropout(rnn_out))  # (batch, seq, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    def generate(self, start_str, max_length=200, temperature=1.0):
        """
        Generate text starting from a prompt.
        
        Args:
            start_str: Starting string
            max_length: Maximum characters to generate
            temperature: Higher = more random, Lower = more deterministic
        """
        self.eval()
        
        # Encode start string
        chars = [char_to_idx.get(ch, 0) for ch in start_str.lower()]
        input_seq = torch.tensor(chars).unsqueeze(0).to(device)
        
        # Initialize hidden state
        hidden = self.init_hidden(1, device)
        
        # Process start string
        with torch.no_grad():
            for i in range(len(chars) - 1):
                _, hidden = self.forward(input_seq[:, i:i+1], hidden)
        
        # Generate new characters
        generated = list(start_str.lower())
        current_char = input_seq[:, -1:]
        
        for _ in range(max_length):
            output, hidden = self.forward(current_char, hidden)
            
            # Apply temperature
            logits = output[0, 0] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample from distribution
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            
            generated.append(next_char)
            current_char = torch.tensor([[next_idx]], device=device)
        
        return ''.join(generated)


# Create model
embed_size = 64
hidden_size = 128
num_layers = 2

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)

print(f"Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# =================================
# 4. TRAINING
# =================================

print("\n" + "=" * 60)
print("4. TRAINING")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

n_epochs = 50
losses = []

print("Training character RNN...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size, device)
        
        # Forward pass
        output, hidden = model(x, hidden)
        
        # Reshape for loss: (batch * seq, vocab) vs (batch * seq,)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for RNNs!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
        
        # Generate sample text
        sample = model.generate("the ", max_length=50, temperature=0.8)
        print(f"  Sample: '{sample}'")

print("\nTraining complete!")

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Character RNN Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('char_rnn_loss.png', dpi=100)
print("Saved char_rnn_loss.png")
plt.show()

# =================================
# 5. TEXT GENERATION
# =================================

print("\n" + "=" * 60)
print("5. TEXT GENERATION")
print("=" * 60)

prompts = ["the ", "machine ", "learning ", "neural "]
temperatures = [0.5, 0.8, 1.0, 1.5]

print("Generated text with different temperatures:")
for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    for temp in temperatures:
        generated = model.generate(prompt, max_length=80, temperature=temp)
        print(f"  T={temp}: '{generated}'")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
}, 'char_rnn_model.pth')
print("\nModel saved to char_rnn_model.pth")
