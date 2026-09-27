# Source: AI_Learning_Cursor lines 23428-23761
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: GPT Training Infrastructure
# Title: GPT TRAINING INFRASTRUCTURE
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
GPT TRAINING INFRASTRUCTURE
===========================
Everything needed to train GPT.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =================================
# 1. LEARNING RATE SCHEDULE
# =================================

print("=" * 60)
print("1. LEARNING RATE SCHEDULE")
print("=" * 60)

"""
GPT uses a specific learning rate schedule:
1. LINEAR WARMUP: Gradually increase LR from 0 to max
2. COSINE DECAY: Smoothly decrease LR to min

This helps:
- Warmup: Prevents early training instability
- Decay: Allows fine-tuning in later stages
"""

class CosineWarmupScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps
        elif self.current_step > self.max_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# Visualize schedule
import matplotlib.pyplot as plt

dummy_model = nn.Linear(10, 10)
optimizer = optim.AdamW(dummy_model.parameters(), lr=1e-4)

scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=100,
    max_steps=1000,
    max_lr=6e-4,
    min_lr=6e-5
)

lrs = []
for _ in range(1000):
    lr = scheduler.step()
    lrs.append(lr)

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Warmup Learning Rate Schedule')
plt.axvline(x=100, color='r', linestyle='--', label='Warmup ends')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedule_gpt.png', dpi=100)
print("Saved lr_schedule_gpt.png")
plt.show()

# =================================
# 2. GRADIENT ACCUMULATION
# =================================

print("\n" + "=" * 60)
print("2. GRADIENT ACCUMULATION")
print("=" * 60)

"""
GRADIENT ACCUMULATION

Problem: Large batch sizes are better for training, but don't fit in memory.
Solution: Accumulate gradients over multiple forward passes before updating.

Effective batch size = micro_batch_size × gradient_accumulation_steps

Example:
    - GPU memory allows batch_size = 8
    - We want effective batch_size = 64
    - Use gradient_accumulation_steps = 8
    - Process 8 batches, sum gradients, then update once
"""

def train_with_gradient_accumulation(
    model, 
    dataloader, 
    optimizer, 
    accumulation_steps=4,
    max_steps=None
):
    """Training loop with gradient accumulation."""
    
    model.train()
    total_loss = 0
    step = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            
            if max_steps and step >= max_steps:
                break
    
    return total_loss / (batch_idx + 1)


print("Gradient accumulation allows larger effective batch sizes!")
print("Example: 4 micro-batches of 16 = effective batch of 64")

# =================================
# 3. TRAINING UTILITIES
# =================================

print("\n" + "=" * 60)
print("3. TRAINING UTILITIES")
print("=" * 60)

class GPTTrainer:
    """
    Complete GPT training class with all best practices.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        max_steps=10000,
        warmup_steps=100,
        max_lr=6e-4,
        min_lr=6e-5,
        weight_decay=0.1,
        gradient_accumulation_steps=1,
        checkpoint_dir='checkpoints',
        log_interval=100,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # Optimizer with weight decay (L2 regularization)
        # Don't apply weight decay to biases and LayerNorm
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if 'bias' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        self.optimizer = optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95))
        
        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, max_steps, max_lr, min_lr
        )
        
        # Tracking
        self.step = 0
        self.train_losses = []
        self.val_losses = []
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        train_iter = iter(self.train_dataloader)
        running_loss = 0.0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for step in range(1, self.max_steps + 1):
            self.step = step
            
            # Get batch (restart iterator if exhausted)
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = self.model(x, y)
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update weights
            if step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                lr = self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if step % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                self.train_losses.append(avg_loss)
                
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                
                perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
                
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                      f"LR: {lr:.2e} | {steps_per_sec:.1f} steps/s")
                
                running_loss = 0.0
                
                # Validation
                if self.val_dataloader is not None:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    self.model.train()
        
        return self.train_losses, self.val_losses
    
    @torch.no_grad()
    def validate(self, max_batches=50):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= max_batches:
                break
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.model.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint loaded from {path}")


print("GPTTrainer class ready!")
print("Includes: AdamW optimizer, cosine warmup, gradient accumulation, checkpointing")
