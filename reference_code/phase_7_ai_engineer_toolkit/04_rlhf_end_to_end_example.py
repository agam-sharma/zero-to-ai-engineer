# Source: AI_Learning_Cursor lines 28965-29206
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: End-to-End RLHF Example
# Title: SIMPLE RLHF IMPLEMENTATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
SIMPLE RLHF IMPLEMENTATION
==========================
A minimal but complete RLHF example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# =================================
# 1. PREFERENCE DATASET
# =================================

print("=" * 60)
print("1. CREATING PREFERENCE DATASET")
print("=" * 60)

class PreferenceDataset(Dataset):
    """
    Dataset of human preferences.
    
    Each item: (prompt, chosen_response, rejected_response)
    """
    
    def __init__(self, data):
        """
        Args:
            data: List of (prompt, chosen, rejected) tuples
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Example preference data
preference_data = [
    (
        "How do I make coffee?",
        "Here's how to make coffee: 1) Boil water 2) Add coffee grounds 3) Pour water over grounds 4) Wait 4 minutes 5) Strain and enjoy!",
        "Coffee is a beverage.",
    ),
    (
        "Explain machine learning",
        "Machine learning is a type of AI where computers learn patterns from data. Instead of being explicitly programmed, the system improves through experience.",
        "It's complicated.",
    ),
    (
        "What's the weather like?",
        "I don't have access to current weather data, but you can check a weather service like weather.com or your phone's weather app for accurate local conditions.",
        "I know everything about weather because I'm an AI.",
    ),
    (
        "Help me write code",
        "I'd be happy to help! What programming language are you using, and what are you trying to accomplish? Please share your current code if you have any.",
        "Write it yourself.",
    ),
] * 25  # Repeat for more data

dataset = PreferenceDataset(preference_data)
print(f"Created preference dataset with {len(dataset)} examples")

# =================================
# 2. SIMPLE REWARD MODEL TRAINING
# =================================

print("\n" + "=" * 60)
print("2. TRAINING REWARD MODEL")
print("=" * 60)

class SimpleRewardModel(nn.Module):
    """
    Simple reward model using embeddings.
    
    In practice, you'd use a full transformer.
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, input_ids):
        # Embed
        x = self.embedding(input_ids)
        
        # Encode
        _, (hidden, _) = self.encoder(x)
        
        # Get reward
        reward = self.reward_head(hidden.squeeze(0))
        
        return reward.squeeze(-1)


def train_reward_model(model, dataset, tokenizer, epochs=5, lr=1e-3):
    """Train the reward model on preferences."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Training reward model...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for prompt, chosen, rejected in dataset:
            # Tokenize (simplified)
            full_chosen = prompt + " " + chosen
            full_rejected = prompt + " " + rejected
            
            chosen_ids = torch.tensor([tokenizer.get(c, 0) for c in full_chosen[:100]])
            rejected_ids = torch.tensor([tokenizer.get(c, 0) for c in full_rejected[:100]])
            
            # Pad to same length
            max_len = max(len(chosen_ids), len(rejected_ids))
            chosen_ids = F.pad(chosen_ids, (0, max_len - len(chosen_ids)))
            rejected_ids = F.pad(rejected_ids, (0, max_len - len(rejected_ids)))
            
            # Get rewards
            reward_chosen = model(chosen_ids.unsqueeze(0))
            reward_rejected = model(rejected_ids.unsqueeze(0))
            
            # Loss: chosen should have higher reward
            loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            if reward_chosen > reward_rejected:
                correct += 1
            total += 1
        
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataset):.4f}, Accuracy = {acc:.2%}")
    
    return model


# Create simple tokenizer
chars = list(set(''.join([p + c + r for p, c, r in preference_data])))
tokenizer = {c: i for i, c in enumerate(chars)}

# Train
reward_model = SimpleRewardModel(vocab_size=len(tokenizer))
reward_model = train_reward_model(reward_model, preference_data[:20], tokenizer, epochs=10)

# =================================
# 3. TEST REWARD MODEL
# =================================

print("\n" + "=" * 60)
print("3. TESTING REWARD MODEL")
print("=" * 60)

def score_response(model, prompt, response, tokenizer):
    """Get reward score for a response."""
    model.eval()
    with torch.no_grad():
        full_text = prompt + " " + response
        ids = torch.tensor([tokenizer.get(c, 0) for c in full_text[:100]])
        ids = F.pad(ids, (0, 100 - len(ids)))
        reward = model(ids.unsqueeze(0))
    return reward.item()


# Test on new examples
test_cases = [
    {
        'prompt': "How do I learn Python?",
        'good': "Start with basics: variables, loops, and functions. Practice daily with small projects. Use resources like Python.org tutorial or Codecademy.",
        'bad': "Python is a snake.",
    },
    {
        'prompt': "What is AI?",
        'good': "AI (Artificial Intelligence) refers to computer systems designed to perform tasks that typically require human intelligence, such as understanding language, recognizing images, and making decisions.",
        'bad': "I don't know.",
    },
]

print("Testing reward model on new examples:\n")
for case in test_cases:
    good_score = score_response(reward_model, case['prompt'], case['good'], tokenizer)
    bad_score = score_response(reward_model, case['prompt'], case['bad'], tokenizer)
    
    print(f"Prompt: {case['prompt']}")
    print(f"  Good response score: {good_score:.3f}")
    print(f"  Bad response score:  {bad_score:.3f}")
    print(f"  Correctly ranked: {'✓' if good_score > bad_score else '✗'}")
    print()

# =================================
# 4. SUMMARY
# =================================

print("=" * 60)
print("RLHF SUMMARY")
print("=" * 60)

print("""
What you've learned:

1. REWARD MODELS
   - Trained on human preferences
   - Predict which response is better
   - Guide LLM optimization

2. PPO TRAINING
   - Use reward model as objective
   - Clip updates for stability
   - KL penalty to stay close to base model

3. ALIGNMENT
   - Helpful, Harmless, Honest (HHH)
   - Avoid reward hacking
   - Constitutional AI for self-improvement

Production RLHF uses:
   - TRL library (HuggingFace)
   - Full transformer reward models
   - Distributed training
   - Extensive red-teaming
""")
