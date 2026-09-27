# Source: AI_Learning_Cursor lines 28002-28288
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: Understanding Fine-Tuning
# Title: FINE-TUNING PRE-TRAINED MODELS
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
FINE-TUNING PRE-TRAINED MODELS
==============================
Adapting large models to your specific use case.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =================================
# 1. TRANSFER LEARNING CONCEPTS
# =================================

print("=" * 60)
print("1. TRANSFER LEARNING FOR LLMs")
print("=" * 60)

"""
TRANSFER LEARNING WORKFLOW:

1. PRE-TRAINING (done by OpenAI, Meta, etc.)
   - Train on MASSIVE data (trillions of tokens)
   - Learn general language understanding
   - Extremely expensive ($1M - $100M+)

2. FINE-TUNING (what YOU do)
   - Start with pre-trained weights
   - Train on YOUR specific data
   - Much cheaper (hours to days on single GPU)

Types of Fine-Tuning:

A. FULL FINE-TUNING
   - Update all model weights
   - Most flexible
   - Risk of catastrophic forgetting
   - Requires more compute/memory

B. PARAMETER-EFFICIENT FINE-TUNING (PEFT)
   - Freeze most weights
   - Only train small adapter layers
   - Examples: LoRA, Prefix Tuning, Adapters
   - Much more efficient

C. PROMPT TUNING
   - Freeze all weights
   - Learn soft prompts (embeddings)
   - Very efficient but less flexible

Salesforce Analogy:
- Pre-training = Building Salesforce platform
- Fine-tuning = Customizing for your org
- You don't rebuild Salesforce; you configure and extend it!
"""

# =================================
# 2. LOADING PRE-TRAINED GPT-2
# =================================

print("\n" + "=" * 60)
print("2. LOADING GPT-2")
print("=" * 60)

# Load tokenizer and model
print("Loading GPT-2 model and tokenizer...")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"Model loaded!")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
print(f"  Max length: {model.config.n_positions}")

# Test generation before fine-tuning
print("\n--- Generation BEFORE fine-tuning ---")
input_text = "The meaning of life is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

print(f"Prompt: '{input_text}'")
print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# =================================
# 3. PREPARING CUSTOM DATASET
# =================================

print("\n" + "=" * 60)
print("3. PREPARING CUSTOM DATASET")
print("=" * 60)

# Example: Fine-tune on customer service responses
custom_data = [
    "Customer: I can't log into my account.\nSupport: I'm sorry to hear that. Let me help you reset your password. Please click the 'Forgot Password' link on the login page.",
    "Customer: My order hasn't arrived yet.\nSupport: I apologize for the delay. Let me check your order status. Can you please provide your order number?",
    "Customer: I want to cancel my subscription.\nSupport: I understand. Before you cancel, may I ask what's prompting this decision? Perhaps there's something we can do to help.",
    "Customer: The product I received is damaged.\nSupport: I'm very sorry about that. We'll send you a replacement right away. No need to return the damaged item.",
    "Customer: How do I upgrade my plan?\nSupport: Great choice! You can upgrade directly from your account settings, or I can help you do it right now.",
    "Customer: I was charged twice.\nSupport: That shouldn't happen. Let me look into this immediately. I'll process a refund for the duplicate charge.",
    "Customer: The app keeps crashing.\nSupport: I apologize for the inconvenience. Have you tried clearing the cache and reinstalling the app? That often resolves this issue.",
    "Customer: I need a refund.\nSupport: I understand. Let me check your purchase history and process that refund for you right away.",
] * 50  # Repeat for more data

print(f"Custom dataset size: {len(custom_data)} examples")
print(f"\nSample:")
print(custom_data[0])

def prepare_dataset(texts, tokenizer, max_length=256):
    """Prepare dataset for fine-tuning."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({'text': texts})
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Add labels (same as input_ids for language modeling)
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples
    
    tokenized = tokenized.map(add_labels, batched=True)
    
    return tokenized


# Prepare dataset
dataset = prepare_dataset(custom_data, tokenizer)

# Split
train_size = int(0.9 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"\nPrepared dataset:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Evaluation samples: {len(eval_dataset)}")

# =================================
# 4. FINE-TUNING WITH HUGGINGFACE
# =================================

print("\n" + "=" * 60)
print("4. FINE-TUNING")
print("=" * 60)

# Training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-customer-service',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=False,  # Set to True if using CUDA with FP16 support
    report_to='none',  # Disable wandb/tensorboard
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")

# Train
print("\nStarting fine-tuning...")
# Uncomment to actually train:
# trainer.train()

print("\n[Training would run here - commented out for demo]")

# =================================
# 5. TEST FINE-TUNED MODEL
# =================================

print("\n" + "=" * 60)
print("5. TESTING FINE-TUNED MODEL")
print("=" * 60)

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate response from fine-tuned model."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test prompts
test_prompts = [
    "Customer: I forgot my password.\nSupport:",
    "Customer: When will my order arrive?\nSupport:",
    "Customer: I want to speak to a manager.\nSupport:",
]

print("After fine-tuning, the model would respond like this:")
print("(Showing expected behavior - actual requires training)")

for prompt in test_prompts:
    print(f"\n{prompt}")
    # response = generate_response(model, tokenizer, prompt)
    # print(response)
    print("[Fine-tuned response would appear here]")

# =================================
# 6. SAVE FINE-TUNED MODEL
# =================================

print("\n" + "=" * 60)
print("6. SAVING MODEL")
print("=" * 60)

def save_fine_tuned_model(model, tokenizer, path):
    """Save the fine-tuned model."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")


def load_fine_tuned_model(path):
    """Load a fine-tuned model."""
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    return model, tokenizer


print("Model saving/loading utilities ready!")
print("Use save_fine_tuned_model() and load_fine_tuned_model()")
