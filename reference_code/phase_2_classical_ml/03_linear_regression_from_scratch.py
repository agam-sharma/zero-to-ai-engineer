# Source: AI_Learning_Cursor lines 1915-2087
# Original transcript phase: None - None
# Nearest header: #### The Complete Implementation
# Title: LINEAR REGRESSION FROM SCRATCH
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
LINEAR REGRESSION FROM SCRATCH

The goal: Given data points, find the best line (or hyperplane) that fits them.

The math:
- Prediction: y_pred = X @ weights + bias
- Loss (Mean Squared Error): L = mean((y_pred - y_true)²)
- Gradients: 
    dL/dweights = (2/n) * X.T @ (y_pred - y_true)
    dL/dbias = (2/n) * sum(y_pred - y_true)
- Update: weights = weights - learning_rate * gradient
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionFromScratch:
    """
    Full implementation of linear regression with gradient descent.
    This is exactly what happens inside ML libraries!
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _initialize_parameters(self, n_features):
        """Initialize weights and bias (like initializing neural network weights)."""
        # Small random values work well for initialization
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        print(f"Initialized weights: {self.weights}")
        print(f"Initialized bias: {self.bias}")
    
    def _forward(self, X):
        """
        Forward pass: compute predictions.
        This is the same as a neural network forward pass!
        """
        return X @ self.weights + self.bias
    
    def _compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss.
        This measures how wrong our predictions are.
        """
        n = len(y_true)
        mse = np.sum((y_pred - y_true) ** 2) / n
        return mse
    
    def _compute_gradients(self, X, y_pred, y_true):
        """
        Compute gradients using calculus.
        This tells us how to adjust weights to reduce loss.
        """
        n = len(y_true)
        error = y_pred - y_true
        
        # Gradient of loss with respect to weights
        dL_dweights = (2 / n) * (X.T @ error)
        
        # Gradient of loss with respect to bias
        dL_dbias = (2 / n) * np.sum(error)
        
        return dL_dweights, dL_dbias
    
    def _update_parameters(self, dL_dweights, dL_dbias):
        """
        Gradient descent update step.
        Move parameters in the direction that reduces loss.
        """
        self.weights = self.weights - self.learning_rate * dL_dweights
        self.bias = self.bias - self.learning_rate * dL_dbias
    
    def fit(self, X, y, n_epochs=100, verbose=True):
        """
        Train the model using gradient descent.
        
        Parameters:
        - X: Input features, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        - n_epochs: Number of training iterations
        """
        n_samples, n_features = X.shape
        
        # Step 1: Initialize parameters
        self._initialize_parameters(n_features)
        
        if verbose:
            print(f"\nTraining for {n_epochs} epochs...")
            print("-" * 50)
        
        for epoch in range(n_epochs):
            # Step 2: Forward pass (make predictions)
            y_pred = self._forward(X)
            
            # Step 3: Compute loss (how wrong are we?)
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Step 4: Backward pass (compute gradients)
            dL_dweights, dL_dbias = self._compute_gradients(X, y_pred, y)
            
            # Step 5: Update parameters (gradient descent step)
            self._update_parameters(dL_dweights, dL_dbias)
            
            if verbose and (epoch < 5 or epoch % 20 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch+1:3d}: Loss = {loss:.6f}")
        
        print(f"\n✅ Training complete!")
        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias:.4f}")
    
    def predict(self, X):
        """Make predictions on new data."""
        return self._forward(X)
    
    def plot_training(self):
        """Visualize the training process."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig('training_loss.png', dpi=100)
        print("Saved training_loss.png")
        plt.show()


# =================================
# TEST THE IMPLEMENTATION
# =================================

print("=" * 60)
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)

# Create synthetic data: y = 3*x1 + 2*x2 + 1 + noise
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)  # 2 features
true_weights = np.array([3.0, 2.0])
true_bias = 1.0
y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.5  # Add noise

print(f"True relationship: y = {true_weights[0]}*x1 + {true_weights[1]}*x2 + {true_bias}")
print(f"Data shape: X={X.shape}, y={y.shape}")

# Train our model
model = LinearRegressionFromScratch(learning_rate=0.1)
model.fit(X, y, n_epochs=100)

# Compare with true values
print(f"\nComparison:")
print(f"  True weights: {true_weights}, Learned weights: {model.weights}")
print(f"  True bias: {true_bias}, Learned bias: {model.bias:.4f}")

# Plot training curve
model.plot_training()

# Test predictions
print(f"\nTest predictions:")
X_test = np.array([[1.0, 1.0], [0.5, 0.5], [-1.0, 2.0]])
predictions = model.predict(X_test)
for x, pred in zip(X_test, predictions):
    true_val = x @ true_weights + true_bias
    print(f"  Input: {x}, Predicted: {pred:.3f}, True: {true_val:.1f}")
