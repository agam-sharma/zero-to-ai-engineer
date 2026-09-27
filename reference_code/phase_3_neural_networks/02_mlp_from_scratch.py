# Source: AI_Learning_Cursor lines 2524-2835
# Original transcript phase: None - None
# Nearest header: ### Day 4-7: Multi-Layer Neural Network from Scratch
# Title: MULTI-LAYER NEURAL NETWORK FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MULTI-LAYER NEURAL NETWORK FROM SCRATCH

This is it! We're building a complete neural network with:
- Multiple layers
- Forward propagation
- Backpropagation
- Gradient descent training

This is EXACTLY what PyTorch does, just without the optimization.
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A fully-connected neural network built from scratch.
    
    Example: NeuralNetwork([2, 4, 4, 1]) creates:
    - Input layer: 2 neurons
    - Hidden layer 1: 4 neurons
    - Hidden layer 2: 4 neurons
    - Output layer: 1 neuron
    """
    
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        """
        Initialize the network.
        
        Parameters:
        - layer_sizes: List of layer sizes, e.g., [2, 4, 4, 1]
        - activation: Activation for hidden layers ('relu', 'sigmoid', 'tanh')
        - output_activation: Activation for output layer ('sigmoid', 'linear')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Store weights and biases for each layer
        self.weights = []
        self.biases = []
        
        # Initialize weights using Xavier/He initialization
        for i in range(self.num_layers - 1):
            # He initialization (good for ReLU)
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Activation functions
        self.activation = activation
        self.output_activation = output_activation
        
        # Cache for backpropagation
        self.cache = {}
        self.loss_history = []
    
    def _activate(self, z, is_output=False):
        """Apply activation function."""
        if is_output:
            if self.output_activation == 'sigmoid':
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            else:  # linear
                return z
        else:
            if self.activation == 'relu':
                return np.maximum(0, z)
            elif self.activation == 'sigmoid':
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            else:  # tanh
                return np.tanh(z)
    
    def _activate_derivative(self, z, is_output=False):
        """Compute derivative of activation."""
        if is_output:
            if self.output_activation == 'sigmoid':
                s = self._activate(z, is_output=True)
                return s * (1 - s)
            else:
                return np.ones_like(z)
        else:
            if self.activation == 'relu':
                return (z > 0).astype(float)
            elif self.activation == 'sigmoid':
                s = self._activate(z)
                return s * (1 - s)
            else:  # tanh
                return 1 - np.tanh(z) ** 2
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        This is where data flows through the network to make predictions.
        """
        self.cache = {'A0': X}  # Store input
        A = X  # Current activation
        
        for i in range(self.num_layers - 1):
            # Linear transformation: Z = A @ W + b
            Z = A @ self.weights[i] + self.biases[i]
            self.cache[f'Z{i+1}'] = Z
            
            # Apply activation
            is_output = (i == self.num_layers - 2)
            A = self._activate(Z, is_output=is_output)
            self.cache[f'A{i+1}'] = A
        
        return A
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_true):
        """
        Backpropagation: compute gradients for all weights and biases.
        
        This is the magic that allows the network to learn!
        """
        m = y_true.shape[0]  # Number of samples
        self.gradients = {}
        
        # Start from the output layer
        y_pred = self.cache[f'A{self.num_layers - 1}']
        
        # Gradient of loss with respect to output
        dA = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        
        # Backpropagate through each layer
        for i in range(self.num_layers - 2, -1, -1):
            Z = self.cache[f'Z{i+1}']
            A_prev = self.cache[f'A{i}']
            
            # Gradient through activation
            is_output = (i == self.num_layers - 2)
            dZ = dA * self._activate_derivative(Z, is_output=is_output)
            
            # Gradients for weights and biases
            self.gradients[f'dW{i}'] = (1 / m) * (A_prev.T @ dZ)
            self.gradients[f'db{i}'] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Gradient to pass to previous layer
            if i > 0:
                dA = dZ @ self.weights[i].T
    
    def update(self, learning_rate):
        """Gradient descent update for all parameters."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.gradients[f'dW{i}']
            self.biases[i] -= learning_rate * self.gradients[f'db{i}']
    
    def fit(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """
        Train the neural network.
        """
        if verbose:
            print(f"Training neural network: {self.layer_sizes}")
            print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
            print("-" * 50)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(y)
            
            # Update parameters
            self.update(learning_rate)
            
            if verbose and (epoch < 5 or epoch % 100 == 0 or epoch == epochs - 1):
                accuracy = np.mean((y_pred >= 0.5) == y)
                print(f"Epoch {epoch+1:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.2%}")
        
        print("\n✅ Training complete!")
    
    def predict(self, X):
        """Make predictions."""
        return (self.forward(X) >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return probabilities."""
        return self.forward(X)
    
    def plot_training(self):
        """Visualize training progress."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('nn_training.png', dpi=100)
        print("Saved nn_training.png")
        plt.show()


# =================================
# TEST: XOR Problem (Classic NN Test)
# =================================

print("=" * 60)
print("NEURAL NETWORK TEST: XOR Problem")
print("=" * 60)
print("""
XOR is a famous problem that single neurons CAN'T solve.
It requires a multi-layer network!

Inputs → Output
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
""")

# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Create and train network
nn = NeuralNetwork([2, 4, 1], activation='relu', output_activation='sigmoid')
nn.fit(X_xor, y_xor, epochs=1000, learning_rate=0.5)

# Test
print("\nFinal predictions:")
for x, y in zip(X_xor, y_xor):
    pred = nn.predict_proba(x.reshape(1, -1))[0, 0]
    print(f"  Input: {x} → Predicted: {pred:.4f}, True: {y[0]}")

nn.plot_training()


# =================================
# TEST 2: Circle Classification
# =================================

print("\n" + "=" * 60)
print("NEURAL NETWORK TEST: Circle Classification")
print("=" * 60)

# Generate circular data
np.random.seed(42)
n_samples = 200

# Inner circle (class 0)
theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
r_inner = np.random.uniform(0, 1, n_samples // 2)
X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

# Outer ring (class 1)
theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
r_outer = np.random.uniform(1.5, 2.5, n_samples // 2)
X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

X = np.vstack([X_inner, X_outer])
y = np.array([[0]] * (n_samples // 2) + [[1]] * (n_samples // 2))

# Shuffle
shuffle_idx = np.random.permutation(n_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]

# Train
nn_circle = NeuralNetwork([2, 8, 8, 1], activation='relu', output_activation='sigmoid')
nn_circle.fit(X, y, epochs=500, learning_rate=0.5)

# Visualize decision boundary
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', label='Class 1', alpha=0.6)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Circle Dataset')
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
# Create mesh grid
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = nn_circle.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
plt.colorbar(label='P(Class 1)')
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', edgecolor='white', label='Class 0')
plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', edgecolor='white', label='Class 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Neural Network Decision Boundary')
plt.axis('equal')

plt.tight_layout()
plt.savefig('nn_decision_boundary.png', dpi=100)
print("Saved nn_decision_boundary.png")
plt.show()

nn_circle.plot_training()
