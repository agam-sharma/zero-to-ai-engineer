# Source: AI_Learning_Cursor lines 24844-25140
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: Sampling Strategies
# Title: TEXT GENERATION STRATEGIES
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TEXT GENERATION STRATEGIES
==========================
Different ways to sample from your GPT.
"""

import torch
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =================================
# 1. SAMPLING STRATEGIES
# =================================

print("=" * 60)
print("1. SAMPLING STRATEGIES")
print("=" * 60)

"""
GENERATION METHODS:

1. GREEDY: Always pick the highest probability token
   - Pros: Deterministic, fast
   - Cons: Repetitive, boring

2. TEMPERATURE: Scale logits before softmax
   - Low temp (0.1-0.5): More focused, less random
   - High temp (1.0-2.0): More creative, more random

3. TOP-K: Only sample from top K tokens
   - Prevents sampling very unlikely tokens
   - K=40-100 typical

4. TOP-P (NUCLEUS): Sample from tokens until cumulative probability >= P
   - Dynamically adjusts number of tokens
   - P=0.9-0.95 typical

5. BEAM SEARCH: Keep top N sequences at each step
   - Better for translation/summarization
   - More expensive
"""

class GPTGenerator:
    """
    Flexible text generator with multiple sampling strategies.
    """
    
    def __init__(self, model, tokenizer_encode, tokenizer_decode, device):
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        stop_sequences=None,
    ):
        """
        Generate text with various sampling strategies.
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Penalty for repeating tokens
            stop_sequences: List of strings to stop at
        """
        # Encode prompt
        idx = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        generated_tokens = []
        
        for _ in range(max_tokens):
            # Get model output
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]  # Last position
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
            generated_tokens.append(idx_next.item())
            
            # Check stop sequences
            if stop_sequences:
                current_text = self.decode(generated_tokens)
                for stop in stop_sequences:
                    if stop in current_text:
                        # Return up to stop sequence
                        return prompt + current_text.split(stop)[0]
        
        return self.decode(idx[0].tolist())
    
    @torch.no_grad()
    def generate_greedy(self, prompt, max_tokens=100):
        """Greedy generation (always pick highest probability)."""
        idx = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            logits, _ = self.model(idx_cond)
            idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return self.decode(idx[0].tolist())
    
    @torch.no_grad()
    def generate_beam(self, prompt, max_tokens=100, beam_width=5):
        """Beam search generation."""
        idx = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        
        # Initialize beams: (sequence, cumulative_log_prob)
        beams = [(idx, 0.0)]
        
        for _ in range(max_tokens):
            all_candidates = []
            
            for seq, score in beams:
                idx_cond = seq if seq.size(1) <= self.model.config.block_size else seq[:, -self.model.config.block_size:]
                logits, _ = self.model(idx_cond)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=1)
                    new_score = score + topk_log_probs[0, i].item()
                    all_candidates.append((new_seq, new_score))
            
            # Keep top beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
        
        # Return best beam
        best_seq, _ = beams[0]
        return self.decode(best_seq[0].tolist())


# Demonstration of different strategies
print("\nSampling strategy comparison:")
print("-" * 50)

# Create dummy examples
strategies = [
    ("Greedy", {"temperature": 0.0001}),
    ("Low Temperature (0.3)", {"temperature": 0.3}),
    ("Medium Temperature (0.7)", {"temperature": 0.7}),
    ("High Temperature (1.5)", {"temperature": 1.5}),
    ("Top-K (K=10)", {"temperature": 1.0, "top_k": 10}),
    ("Top-K (K=50)", {"temperature": 1.0, "top_k": 50}),
    ("Top-P (P=0.9)", {"temperature": 1.0, "top_p": 0.9}),
    ("Top-P (P=0.5)", {"temperature": 1.0, "top_p": 0.5}),
]

print("Different sampling strategies produce different outputs:")
for name, params in strategies:
    print(f"\n{name}:")
    print(f"  Parameters: {params}")
    print(f"  Behavior: ", end="")
    if params.get('temperature', 1.0) < 0.5:
        print("Deterministic, repetitive")
    elif params.get('temperature', 1.0) > 1.2:
        print("Random, creative, may be incoherent")
    elif params.get('top_k'):
        print(f"Only considers top {params['top_k']} tokens")
    elif params.get('top_p'):
        print(f"Considers tokens until {params['top_p']*100}% probability mass")
    else:
        print("Balanced creativity and coherence")

# =================================
# 2. INTERACTIVE GENERATION
# =================================

print("\n" + "=" * 60)
print("2. INTERACTIVE GENERATION INTERFACE")
print("=" * 60)

def create_chat_interface(generator):
    """Create a simple chat-like interface."""
    
    print("\n" + "="*50)
    print("SHAKESPEARE GPT - Interactive Generator")
    print("="*50)
    print("Commands:")
    print("  /temp <value>  - Set temperature (0.1-2.0)")
    print("  /topk <value>  - Set top-k (10-100)")
    print("  /topp <value>  - Set top-p (0.1-1.0)")
    print("  /len <value>   - Set max tokens (50-500)")
    print("  /quit          - Exit")
    print("="*50)
    
    settings = {
        'temperature': 0.8,
        'top_k': 40,
        'top_p': None,
        'max_tokens': 200,
    }
    
    while True:
        try:
            user_input = input("\nPrompt> ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                # Handle commands
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == '/quit':
                    print("Goodbye!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    settings['temperature'] = float(parts[1])
                    print(f"Temperature set to {settings['temperature']}")
                elif cmd == '/topk' and len(parts) > 1:
                    settings['top_k'] = int(parts[1])
                    settings['top_p'] = None
                    print(f"Top-k set to {settings['top_k']}")
                elif cmd == '/topp' and len(parts) > 1:
                    settings['top_p'] = float(parts[1])
                    settings['top_k'] = None
                    print(f"Top-p set to {settings['top_p']}")
                elif cmd == '/len' and len(parts) > 1:
                    settings['max_tokens'] = int(parts[1])
                    print(f"Max tokens set to {settings['max_tokens']}")
                else:
                    print("Unknown command")
                continue
            
            # Generate
            print("\nGenerating...")
            output = generator.generate(
                user_input,
                max_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
            )
            
            print("\n" + "-"*40)
            print(output)
            print("-"*40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


print("\nInteractive interface ready!")
print("(Would run with trained model)")
