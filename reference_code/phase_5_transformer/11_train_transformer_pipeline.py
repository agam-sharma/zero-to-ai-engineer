# Source: AI_Learning_Cursor lines 19618-19985
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Complete Training Pipeline
# Title: TRAINING A TRANSFORMER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TRAINING A TRANSFORMER
======================
Complete training pipeline with all best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# =================================
# 1. DATA PREPARATION
# =================================

print("=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

class SequenceReverseDataset(Dataset):
    """
    Dataset for sequence reversal task.
    
    Input:  [1, 2, 3, 4, 5]
    Output: [5, 4, 3, 2, 1]
    
    This is a good test task because:
    - Clear correct answer
    - Requires understanding full sequence
    - Can verify correctness easily
    """
    
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate data
        # Use tokens 3+ (0=pad, 1=start, 2=end)
        self.src = torch.randint(3, vocab_size, (num_samples, seq_len))
        self.tgt = self.src.flip(dims=[1])
        
        # Add start/end tokens to target
        start_tokens = torch.ones(num_samples, 1, dtype=torch.long)  # 1 = start
        end_tokens = torch.full((num_samples, 1), 2, dtype=torch.long)  # 2 = end
        
        # Target input (for teacher forcing): [START, reversed_seq]
        self.tgt_input = torch.cat([start_tokens, self.tgt], dim=1)
        
        # Target output (what we predict): [reversed_seq, END]
        self.tgt_output = torch.cat([self.tgt, end_tokens], dim=1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt_input[idx], self.tgt_output[idx]


# Create datasets
vocab_size = 100
seq_len = 10
train_dataset = SequenceReverseDataset(5000, seq_len, vocab_size)
val_dataset = SequenceReverseDataset(500, seq_len, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"Vocabulary size: {vocab_size}")
print(f"Sequence length: {seq_len}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Show sample
src, tgt_in, tgt_out = train_dataset[0]
print(f"\nSample:")
print(f"  Source:       {src.tolist()}")
print(f"  Target input: {tgt_in.tolist()}")
print(f"  Target output:{tgt_out.tolist()}")

# =================================
# 2. MODEL
# =================================

print("\n" + "=" * 60)
print("2. MODEL")
print("=" * 60)

# Use the Transformer we built earlier (simplified version for this task)
class SimpleTransformer(nn.Module):
    """Simplified Transformer for sequence-to-sequence."""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Shared embedding for src and tgt (same vocabulary)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(100, d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt):
        src_len, tgt_len = src.size(1), tgt.size(1)
        device = src.device
        
        # Embeddings
        src_pos = torch.arange(src_len, device=device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0)
        
        src_emb = self.dropout(self.embedding(src) + self.pos_embedding(src_pos))
        tgt_emb = self.dropout(self.embedding(tgt) + self.pos_embedding(tgt_pos))
        
        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
        
        # Transformer forward
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        return self.fc_out(output)


model = SimpleTransformer(vocab_size).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# =================================
# 3. TRAINING LOOP
# =================================

print("\n" + "=" * 60)
print("3. TRAINING")
print("=" * 60)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler (warmup then decay)
def lr_lambda(step):
    warmup_steps = 500
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.1, (10000 - step) / 10000)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    for src, tgt_in, tgt_out in loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(src, tgt_in)
        
        # Reshape for loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = tgt_out.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        mask = tgt_out != 0
        correct_tokens += ((preds == tgt_out) & mask).sum().item()
        total_tokens += mask.sum().item()
    
    return total_loss / len(loader), correct_tokens / total_tokens


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_seqs = 0
    total_seqs = 0
    
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            
            logits = model(src, tgt_in)
            
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = tgt_out.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            
            # Sequence accuracy
            preds = logits.argmax(dim=-1)
            # Check if entire sequence matches (excluding padding)
            for pred, target in zip(preds, tgt_out):
                if (pred[:-1] == target[:-1]).all():  # Exclude last token comparison
                    correct_seqs += 1
                total_seqs += 1
    
    return total_loss / len(loader), correct_seqs / total_seqs


# Training
n_epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training...")
start_time = time.time()

for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Seq Acc: {val_acc:.2%} | "
              f"LR: {current_lr:.6f}")

training_time = time.time() - start_time
print(f"\nTraining complete in {training_time:.1f} seconds")
print(f"Final validation sequence accuracy: {val_accs[-1]:.2%}")

# =================================
# 4. VISUALIZATION
# =================================

print("\n" + "=" * 60)
print("4. RESULTS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train Token Acc')
axes[1].plot(val_accs, label='Val Sequence Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transformer_training.png', dpi=100)
print("Saved transformer_training.png")
plt.show()

# =================================
# 5. TESTING
# =================================

print("\n" + "=" * 60)
print("5. TESTING")
print("=" * 60)

def generate_greedy(model, src, max_len=15, start_token=1, end_token=2):
    """Generate output sequence using greedy decoding."""
    model.eval()
    device = src.device
    
    # Start with start token
    generated = torch.tensor([[start_token]], device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == end_token:
                break
    
    return generated.squeeze().tolist()


print("Test predictions:")
model.eval()

for i in range(5):
    src, tgt_in, tgt_out = val_dataset[i]
    src = src.unsqueeze(0).to(device)
    
    generated = generate_greedy(model, src)
    
    print(f"  Input:    {val_dataset.src[i].tolist()}")
    print(f"  Expected: {val_dataset.tgt[i].tolist()}")
    print(f"  Generated:{generated[1:-1] if 2 in generated else generated[1:]}")
    correct = generated[1:-1] == val_dataset.tgt[i].tolist() if 2 in generated else False
    print(f"  Correct: {'✓' if correct else '✗'}")
    print()

# =================================
# 6. SAVE MODEL
# =================================

print("=" * 60)
print("6. SAVING MODEL")
print("=" * 60)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': vocab_size,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 3,
    'train_losses': train_losses,
    'val_losses': val_losses,
}

torch.save(checkpoint, 'transformer_checkpoint.pth')
print("Model saved to transformer_checkpoint.pth")
