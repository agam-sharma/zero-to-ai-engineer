# Source: AI_Learning_Cursor lines 12161-12367
# Original transcript phase: 3 - SEQUENCE MODELING AND NLP FOUNDATIONS
# Nearest header: #### CODE: Understanding Vanishing Gradients
# Title: VANISHING GRADIENT PROBLEM
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
VANISHING GRADIENT PROBLEM
==========================
Why vanilla RNNs struggle with long sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =================================
# 1. THE PROBLEM
# =================================

print("=" * 60)
print("1. THE VANISHING GRADIENT PROBLEM")
print("=" * 60)

"""
BACKPROPAGATION THROUGH TIME (BPTT)

When training RNNs, gradients must flow back through all time steps.
At each step, gradients are multiplied by the weight matrix.

If weights have eigenvalues < 1: Gradients shrink exponentially (VANISHING)
If weights have eigenvalues > 1: Gradients grow exponentially (EXPLODING)

For a sequence of length T:
    gradient ∝ W^T

If W = 0.9:
    0.9^10 ≈ 0.35
    0.9^50 ≈ 0.005
    0.9^100 ≈ 0.00003
    
The gradient becomes so small that early layers barely learn!
"""

# Demonstrate gradient decay
def simulate_gradient_flow(weight_factor, seq_length):
    """Simulate gradient magnitude over sequence length."""
    gradient = 1.0
    gradients = [gradient]
    
    for t in range(seq_length):
        gradient *= weight_factor
        gradients.append(gradient)
    
    return gradients


# Compare different weight magnitudes
plt.figure(figsize=(12, 4))

seq_lengths = range(100)
for w in [0.9, 0.95, 0.99, 1.0, 1.01, 1.05]:
    grads = simulate_gradient_flow(w, 100)
    plt.plot(grads, label=f'w={w}')

plt.xlabel('Time Steps (going backward)')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow in RNNs')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('vanishing_gradients.png', dpi=100)
print("Saved vanishing_gradients.png")
plt.show()

# =================================
# 2. DEMONSTRATING THE PROBLEM
# =================================

print("\n" + "=" * 60)
print("2. LONG-TERM DEPENDENCY TASK")
print("=" * 60)

"""
Task: Remember the first character to predict the last.

Sequence: "A...........B"
Target: "A" (first character)

For short sequences: Easy
For long sequences: Vanilla RNN fails!
"""

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


def create_memory_task_data(n_samples, seq_length, n_classes=5):
    """
    Create data for long-term memory task.
    First element is the class, rest is noise.
    Target is the first element.
    """
    # Random class labels
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create sequences: one-hot first element, zeros rest
    X = torch.zeros(n_samples, seq_length, n_classes)
    X[:, 0, :] = torch.nn.functional.one_hot(labels, n_classes).float()
    
    # Add noise to make it harder
    X += torch.randn_like(X) * 0.1
    
    return X, labels


def train_and_evaluate(model, seq_length, n_epochs=100):
    """Train model and return final accuracy."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create data
    X_train, y_train = create_memory_task_data(500, seq_length)
    X_test, y_test = create_memory_task_data(100, seq_length)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Training
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
        accuracy = (pred == y_test).float().mean().item()
    
    return accuracy


# Compare RNN vs LSTM on different sequence lengths
print("Comparing RNN vs LSTM on long-term memory task...")
print("(Predicting first element after seeing long sequence)")
print()

sequence_lengths = [10, 25, 50, 100, 150, 200]
rnn_accuracies = []
lstm_accuracies = []

for seq_len in sequence_lengths:
    print(f"Sequence length: {seq_len}")
    
    # RNN
    rnn = VanillaRNN(5, 32, 5)
    rnn_acc = train_and_evaluate(rnn, seq_len)
    rnn_accuracies.append(rnn_acc)
    print(f"  RNN accuracy: {rnn_acc:.2%}")
    
    # LSTM
    lstm = LSTMModel(5, 32, 5)
    lstm_acc = train_and_evaluate(lstm, seq_len)
    lstm_accuracies.append(lstm_acc)
    print(f"  LSTM accuracy: {lstm_acc:.2%}")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(sequence_lengths, rnn_accuracies, 'o-', label='Vanilla RNN', linewidth=2)
plt.plot(sequence_lengths, lstm_accuracies, 's-', label='LSTM', linewidth=2)
plt.axhline(y=0.2, color='gray', linestyle='--', label='Random baseline (20%)')
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.title('Long-Term Memory Task: RNN vs LSTM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rnn_vs_lstm_memory.png', dpi=100)
print("\nSaved rnn_vs_lstm_memory.png")
plt.show()

print("\nObservation:")
print("- Vanilla RNN accuracy drops as sequence length increases")
print("- LSTM maintains better performance on longer sequences")
print("- This is the vanishing gradient problem in action!")
