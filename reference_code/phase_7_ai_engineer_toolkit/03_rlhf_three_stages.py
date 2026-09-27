# Source: AI_Learning_Cursor lines 28599-28957
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: RLHF Concepts and Implementation
# Title: RLHF (REINFORCEMENT LEARNING FROM HUMAN FEEDBACK)
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
RLHF (REINFORCEMENT LEARNING FROM HUMAN FEEDBACK)
==================================================
How ChatGPT learns to be helpful, harmless, and honest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =================================
# 1. THE THREE STAGES OF CHATGPT TRAINING
# =================================

print("=" * 60)
print("1. HOW CHATGPT IS TRAINED")
print("=" * 60)

"""
CHATGPT TRAINING PIPELINE:

STAGE 1: PRE-TRAINING (Base Model)
    - Train on internet text (books, websites, code)
    - Objective: Predict next token
    - Result: GPT that can complete text
    - Problem: No understanding of "helpful" vs "harmful"

STAGE 2: SUPERVISED FINE-TUNING (SFT)
    - Human labelers write ideal responses
    - Fine-tune on (prompt, ideal_response) pairs
    - Result: GPT that follows instructions
    - Problem: Limited by amount of human data

STAGE 3: RLHF (Reinforcement Learning from Human Feedback)
    
    Step 3a: Train Reward Model
        - Show humans pairs of responses
        - Ask: "Which response is better?"
        - Train model to predict human preferences
        
    Step 3b: Optimize with PPO
        - Use reward model as objective
        - Fine-tune with reinforcement learning
        - Model learns to generate high-reward responses

Result: ChatGPT that is helpful, harmless, and honest!
"""

# Visual representation
print("""
Training Pipeline:
                                                
  Internet Text        Human Demos          Human Rankings
       │                    │                     │
       ▼                    ▼                     ▼
  ┌─────────┐         ┌─────────┐          ┌─────────┐
  │Pre-train│   →     │   SFT   │    →     │  RLHF   │
  │ (GPT)   │         │         │          │  (PPO)  │
  └─────────┘         └─────────┘          └─────────┘
       │                    │                     │
       ▼                    ▼                     ▼
   Base LLM          Instruction-       Aligned ChatGPT
                     Following LLM
""")

# =================================
# 2. REWARD MODEL
# =================================

print("\n" + "=" * 60)
print("2. REWARD MODEL")
print("=" * 60)

"""
REWARD MODEL: Predicts how good a response is

Training data:
    - Prompt: "Explain quantum physics"
    - Response A: "Quantum physics is complicated..."
    - Response B: "I'll explain simply. At tiny scales..."
    - Human preference: B is better
    
The reward model learns to predict:
    - Higher score for B
    - Lower score for A
    
This model then guides the LLM training!
"""

class RewardModel(nn.Module):
    """
    Simple reward model that scores responses.
    
    In practice, this would be a full transformer.
    """
    
    def __init__(self, base_model, hidden_size=768):
        """
        Args:
            base_model: Pre-trained language model
            hidden_size: Size of the hidden layer
        """
        super().__init__()
        
        self.base_model = base_model
        
        # Reward head: takes last hidden state, outputs scalar
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Compute reward score for a response.
        
        Returns a scalar reward for each item in batch.
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use last hidden state of last token
        last_hidden = outputs.hidden_states[-1]
        
        # Get hidden state of last (non-padding) token
        if attention_mask is not None:
            # Find last non-padding position
            last_positions = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_hidden = last_hidden[range(batch_size), last_positions]
        else:
            last_hidden = last_hidden[:, -1, :]
        
        # Compute reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        
        return reward


def reward_model_loss(rewards_chosen, rewards_rejected):
    """
    Compute loss for reward model training.
    
    We want: reward(chosen) > reward(rejected)
    
    Loss: -log(sigmoid(reward_chosen - reward_rejected))
    
    This is equivalent to binary cross-entropy where
    we want to predict that chosen > rejected.
    """
    return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()


print("Reward Model architecture ready!")

# Demonstration with dummy data
print("\nReward model training example:")
print("  Chosen response: 'I'll help you understand...'")
print("  Rejected response: 'I cannot help with that...'")
print("  Goal: reward(chosen) > reward(rejected)")

# Simulated rewards
reward_chosen = torch.tensor([2.5])
reward_rejected = torch.tensor([1.0])
loss = reward_model_loss(reward_chosen, reward_rejected)
print(f"  Loss: {loss.item():.4f}")

# =================================
# 3. PPO (PROXIMAL POLICY OPTIMIZATION)
# =================================

print("\n" + "=" * 60)
print("3. PPO FOR RLHF")
print("=" * 60)

"""
PPO: How we optimize the LLM using the reward model

Problem:
    - We have a reward model that scores responses
    - We want to update the LLM to generate high-reward responses
    - But we can't just maximize reward (model would cheat!)

PPO Objective:

    L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] - β * KL(π || π_ref)

Where:
    - r_t = π(a|s) / π_old(a|s)  (probability ratio)
    - A_t = advantage (reward - baseline)
    - ε = clip parameter (~0.2)
    - KL = divergence from reference model
    - β = KL penalty coefficient

Key Ideas:
1. Maximize reward from reward model
2. Don't change too much from previous policy (clipping)
3. Stay close to original model (KL penalty)

The KL penalty prevents:
    - Reward hacking (finding exploits in reward model)
    - Mode collapse (always generating same response)
    - Forgetting how to write coherent text
"""

class PPOTrainer:
    """
    Simplified PPO trainer for RLHF.
    
    This is a conceptual implementation.
    Production uses libraries like trl (HuggingFace).
    """
    
    def __init__(
        self,
        policy_model,
        reference_model,
        reward_model,
        tokenizer,
        lr=1e-5,
        clip_epsilon=0.2,
        kl_coef=0.1,
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        
        # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False
    
    def compute_rewards(self, responses):
        """Get rewards from reward model."""
        with torch.no_grad():
            rewards = self.reward_model(responses)
        return rewards
    
    def compute_kl_penalty(self, policy_logprobs, reference_logprobs):
        """Compute KL divergence penalty."""
        return (policy_logprobs - reference_logprobs).mean()
    
    def ppo_step(self, prompts, responses, old_logprobs):
        """
        One PPO optimization step.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            old_logprobs: Log probabilities from behavior policy
        """
        # Get rewards
        rewards = self.compute_rewards(responses)
        
        # Get current log probabilities
        outputs = self.policy(responses)
        logits = outputs.logits
        new_logprobs = self._get_logprobs(logits, responses)
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_outputs = self.reference(responses)
            ref_logprobs = self._get_logprobs(ref_outputs.logits, responses)
        
        # Compute advantages (simplified: reward - KL penalty)
        kl_penalty = self.compute_kl_penalty(new_logprobs, ref_logprobs)
        advantages = rewards - self.kl_coef * kl_penalty
        
        # Compute PPO loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': policy_loss.item(),
            'reward': rewards.mean().item(),
            'kl': kl_penalty.item(),
        }
    
    def _get_logprobs(self, logits, labels):
        """Get log probabilities of labels."""
        logprobs = F.log_softmax(logits, dim=-1)
        selected_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return selected_logprobs.mean(dim=-1)


print("PPO Trainer ready!")
print("\nRLHF training loop:")
print("1. Generate responses from policy")
print("2. Score responses with reward model")
print("3. Update policy with PPO")
print("4. Repeat!")

# =================================
# 4. ALIGNMENT CONCEPTS
# =================================

print("\n" + "=" * 60)
print("4. AI ALIGNMENT CONCEPTS")
print("=" * 60)

"""
AI ALIGNMENT: Making AI systems do what we actually want

Key Challenges:

1. SPECIFICATION PROBLEM
   - Hard to specify exactly what we want
   - "Be helpful" is vague
   - Edge cases are infinite

2. REWARD HACKING
   - AI finds loopholes in reward function
   - Example: Asked to minimize customer complaints,
     AI learns to hang up on callers

3. GOAL MISGENERALIZATION
   - AI learns wrong goal during training
   - Performs well in training, fails in deployment

4. INNER ALIGNMENT
   - Does the AI's internal goal match our objective?
   - "Mesa-optimization" concerns

HHH Framework (Anthropic):
   - Helpful: Tries to assist the user
   - Harmless: Avoids harmful outputs
   - Honest: Doesn't deceive, knows its limitations

Constitutional AI:
   - AI critiques its own outputs
   - Revises based on principles
   - Self-improvement with oversight
"""

print("Alignment techniques used in modern LLMs:")
print("  1. RLHF - Learn from human preferences")
print("  2. Constitutional AI - Self-critique")
print("  3. Red teaming - Find failure modes")
print("  4. Capability evaluation - Measure risks")
print("  5. Interpretability - Understand decisions")
