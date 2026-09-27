# Source: AI_Learning_Cursor lines 2313-2516
# Original transcript phase: None - None
# Nearest header: ### Day 1-3: The Neuron - Building Block of AI
# Title: THE NEURON: The Fundamental Unit of Neural Networks
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
THE NEURON: The Fundamental Unit of Neural Networks

A neuron:
1. Receives inputs (x1, x2, ..., xn)
2. Multiplies each by a learned weight (w1, w2, ..., wn)
3. Adds a bias (b)
4. Applies an activation function (σ)

Output = σ(w1*x1 + w2*x2 + ... + wn*xn + b)
       = σ(w · x + b)
       = σ(dot_product(weights, inputs) + bias)

Salesforce Analogy:
This is like a super-powered Formula Field that:
- Takes multiple field values as inputs
- Weights each field by importance
- Produces a single output
- The weights are LEARNED from data, not hand-coded!
"""

import numpy as np
import matplotlib.pyplot as plt

# =================================
# ACTIVATION FUNCTIONS
# =================================

def sigmoid(x):
    """Squashes to (0, 1). Good for probabilities."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid: σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    ReLU (Rectified Linear Unit): max(0, x)
    - If negative: output 0
    - If positive: output as-is
    
    This is the most popular activation in modern neural networks!
    Why? It's simple and avoids the "vanishing gradient" problem.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def tanh(x):
    """
    Tanh: squashes to (-1, 1)
    Often used in RNNs/LSTMs
    """
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


# Visualize activation functions
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e^-x)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(x, sigmoid_derivative(x), 'b-', linewidth=2)
axes[1, 0].set_title('Sigmoid Derivative')
axes[1, 0].grid(True, alpha=0.3)

# ReLU
axes[0, 1].plot(x, relu(x), 'r-', linewidth=2)
axes[0, 1].set_title('ReLU: max(0, x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(x, relu_derivative(x), 'r-', linewidth=2)
axes[1, 1].set_title('ReLU Derivative')
axes[1, 1].grid(True, alpha=0.3)

# Tanh
axes[0, 2].plot(x, tanh(x), 'g-', linewidth=2)
axes[0, 2].set_title('Tanh: (e^x - e^-x)/(e^x + e^-x)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(x, tanh_derivative(x), 'g-', linewidth=2)
axes[1, 2].set_title('Tanh Derivative')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=100)
print("Saved activation_functions.png")
plt.show()


# =================================
# SINGLE NEURON IMPLEMENTATION
# =================================

class Neuron:
    """
    A single neuron with learnable weights and bias.
    
    This is the building block of ALL neural networks.
    """
    
    def __init__(self, num_inputs, activation='sigmoid'):
        """
        Initialize neuron with random weights.
        
        Parameters:
        - num_inputs: Number of input features
        - activation: 'sigmoid', 'relu', or 'tanh'
        """
        # Xavier initialization (helps training stability)
        self.weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
        self.bias = 0.0
        
        # Store activation function
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        
        # Cache for backpropagation
        self._linear_output = None
        self._input = None
    
    def forward(self, x):
        """
        Compute neuron output.
        
        Steps:
        1. Linear combination: z = w · x + b
        2. Activation: a = σ(z)
        """
        self._input = x
        self._linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(self._linear_output)
    
    def backward(self, gradient_from_next):
        """
        Backpropagation: compute gradients.
        
        Parameters:
        - gradient_from_next: How much the loss changes with our output
        
        Returns:
        - gradient_to_prev: How much the loss changes with our inputs
        """
        # Gradient through activation
        activation_gradient = self.activation_derivative(self._linear_output)
        delta = gradient_from_next * activation_gradient
        
        # Gradients for our parameters
        self.grad_weights = self._input * delta
        self.grad_bias = delta
        
        # Gradient to pass back to previous layer
        gradient_to_prev = self.weights * delta
        
        return gradient_to_prev
    
    def update(self, learning_rate):
        """Gradient descent step."""
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


# Test single neuron
print("=" * 60)
print("SINGLE NEURON TEST")
print("=" * 60)

# Create a neuron with 3 inputs
neuron = Neuron(num_inputs=3, activation='sigmoid')
print(f"Initial weights: {neuron.weights}")
print(f"Initial bias: {neuron.bias}")

# Forward pass
x = np.array([0.5, 0.8, -0.2])
output = neuron.forward(x)
print(f"\nInput: {x}")
print(f"Output: {output:.4f}")

# Simulated backprop (pretend gradient from loss is 1.0)
grad_back = neuron.backward(gradient_from_next=1.0)
print(f"\nGradient for weights: {neuron.grad_weights}")
print(f"Gradient for bias: {neuron.grad_bias:.4f}")
