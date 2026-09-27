# Source: AI_Learning_Cursor lines 28302-28557
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: LoRA Implementation
# Title: LoRA (LOW-RANK ADAPTATION)
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
LoRA (LOW-RANK ADAPTATION)
==========================
Efficient fine-tuning by training only small adapter layers.
"""

import torch
import torch.nn as nn
import math

# =================================
# 1. LORA CONCEPT
# =================================

print("=" * 60)
print("1. LORA CONCEPT")
print("=" * 60)

"""
LoRA: Low-Rank Adaptation of Large Language Models

Key Insight:
    Weight updates during fine-tuning have low "intrinsic rank"
    Instead of updating W directly, we learn: W + BA
    Where B and A are low-rank matrices

Original: W is (d x k) with d*k parameters
LoRA: B is (d x r) and A is (r x k) with r*(d+k) parameters

Example with d=1000, k=1000:
    Original: 1,000,000 parameters
    LoRA (r=8): 8 * 2000 = 16,000 parameters (62x reduction!)

Benefits:
    - 10-100x fewer trainable parameters
    - Much less memory needed
    - Faster training
    - Easy to switch between different fine-tunes
    - Original weights unchanged (no forgetting)
"""

# =================================
# 2. LORA LAYER FROM SCRATCH
# =================================

print("\n" + "=" * 60)
print("2. LORA LAYER IMPLEMENTATION")
print("=" * 60)

class LoRALayer(nn.Module):
    """
    LoRA adapter layer.
    
    Adds low-rank update to existing linear layer.
    Output = Original(x) + scale * B(A(x))
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            original_layer: The linear layer to adapt
            rank: Rank of the low-rank matrices (r)
            alpha: Scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # Start with zero contribution
    
    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        
        # Combined
        return original_output + self.scaling * lora_output
    
    @property
    def lora_parameters(self):
        """Return only LoRA parameters (for optimizer)."""
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())


# Demonstrate
original = nn.Linear(1024, 1024)
lora = LoRALayer(original, rank=8, alpha=16)

print("LoRA Layer created:")
print(f"  Original parameters: {sum(p.numel() for p in original.parameters()):,}")
print(f"  LoRA parameters: {sum(p.numel() for p in lora.lora_parameters):,}")
print(f"  Reduction: {sum(p.numel() for p in original.parameters()) / sum(p.numel() for p in lora.lora_parameters):.1f}x")

# =================================
# 3. APPLY LORA TO GPT
# =================================

print("\n" + "=" * 60)
print("3. APPLYING LORA TO GPT")
print("=" * 60)

def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['c_attn', 'c_proj']):
    """
    Apply LoRA to specified modules in a GPT model.
    
    Args:
        model: The GPT model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: Names of modules to adapt
        
    Returns:
        Modified model with LoRA layers
    """
    lora_layers = []
    
    for name, module in model.named_modules():
        # Check if this is a target module
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Replace with LoRA layer
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_layers.append(lora_layer)
                
                print(f"Applied LoRA to: {name}")
    
    return model, lora_layers


# Example (would apply to real GPT-2):
print("\nLoRA would be applied to attention layers (c_attn, c_proj)")
print("This reduces trainable parameters by ~60-100x!")

# =================================
# 4. LORA TRAINING LOOP
# =================================

print("\n" + "=" * 60)
print("4. LORA TRAINING")
print("=" * 60)

def get_lora_optimizer(lora_layers, lr=1e-4):
    """Create optimizer for only LoRA parameters."""
    lora_params = []
    for layer in lora_layers:
        lora_params.extend(layer.lora_parameters)
    
    return torch.optim.AdamW(lora_params, lr=lr)


def train_with_lora(model, lora_layers, train_loader, num_epochs=3, lr=1e-4):
    """Training loop for LoRA fine-tuning."""
    
    optimizer = get_lora_optimizer(lora_layers, lr=lr)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


print("LoRA training utilities ready!")
print("Only LoRA parameters are updated during training.")

# =================================
# 5. SAVE/LOAD LORA WEIGHTS
# =================================

print("\n" + "=" * 60)
print("5. SAVING LORA WEIGHTS")
print("=" * 60)

def save_lora_weights(lora_layers, path):
    """Save only LoRA weights (very small file!)."""
    lora_state = {}
    for i, layer in enumerate(lora_layers):
        lora_state[f'layer_{i}_A'] = layer.lora_A.state_dict()
        lora_state[f'layer_{i}_B'] = layer.lora_B.state_dict()
    
    torch.save(lora_state, path)
    print(f"LoRA weights saved to {path}")
    
    # Show file size
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
    print(f"File size: {size_mb:.2f} MB (vs ~500MB for full GPT-2)")


def load_lora_weights(lora_layers, path):
    """Load LoRA weights."""
    lora_state = torch.load(path)
    
    for i, layer in enumerate(lora_layers):
        layer.lora_A.load_state_dict(lora_state[f'layer_{i}_A'])
        layer.lora_B.load_state_dict(lora_state[f'layer_{i}_B'])
    
    print(f"LoRA weights loaded from {path}")


print("\nLoRA advantage: Switch between different fine-tunes instantly!")
print("Keep one base model, load different LoRA weights for different tasks.")
