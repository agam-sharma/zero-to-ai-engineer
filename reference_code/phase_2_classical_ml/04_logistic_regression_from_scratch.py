# Source: AI_Learning_Cursor lines 2104-2291
# Original transcript phase: None - None
# Nearest header: #### The "Why"
# Title: LOGISTIC REGRESSION: Classification
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
LOGISTIC REGRESSION: Classification

The key difference from linear regression:
- Linear regression: Predict a continuous value (price, temperature)
- Logistic regression: Predict a probability (0 to 1) for classification

We add a SIGMOID function to squash outputs into [0, 1] range.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    The sigmoid function: converts any number to range (0, 1).
    
    This is used EVERYWHERE in deep learning:
    - Final layer of classifiers
    - Gates in LSTMs
    - Attention scores
    
    σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))


# Visualize sigmoid
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 4))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary (0.5)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('z (linear combination: w·x + b)')
plt.ylabel('σ(z) (probability)')
plt.title('The Sigmoid Function: Converts Numbers to Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sigmoid.png', dpi=100)
print("Saved sigmoid.png")
plt.show()


class LogisticRegressionFromScratch:
    """
    Logistic regression for binary classification.
    
    Same structure as linear regression, but:
    1. Apply sigmoid to get probabilities
    2. Use Binary Cross-Entropy loss instead of MSE
    """
    
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)  # Zero init works for logistic
        self.bias = 0.0
    
    def _sigmoid(self, z):
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _forward(self, X):
        """Forward pass: linear combination → sigmoid → probability."""
        linear = X @ self.weights + self.bias
        return self._sigmoid(linear)
    
    def _compute_loss(self, y_prob, y_true):
        """
        Binary Cross-Entropy Loss.
        
        This measures how wrong our PROBABILITIES are.
        It heavily penalizes confident wrong predictions.
        
        L = -mean(y * log(p) + (1-y) * log(1-p))
        """
        # Clip to avoid log(0)
        epsilon = 1e-15
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        )
        return loss
    
    def _compute_gradients(self, X, y_prob, y_true):
        """
        Gradients for logistic regression.
        Interestingly, they have the same FORM as linear regression!
        """
        n = len(y_true)
        error = y_prob - y_true
        
        dL_dweights = (1 / n) * (X.T @ error)
        dL_dbias = (1 / n) * np.sum(error)
        
        return dL_dweights, dL_dbias
    
    def _update_parameters(self, dL_dweights, dL_dbias):
        self.weights = self.weights - self.learning_rate * dL_dweights
        self.bias = self.bias - self.learning_rate * dL_dbias
    
    def fit(self, X, y, n_epochs=100, verbose=True):
        """Train the classifier."""
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        if verbose:
            print(f"Training logistic regression for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            # Forward pass
            y_prob = self._forward(X)
            
            # Compute loss
            loss = self._compute_loss(y_prob, y)
            self.loss_history.append(loss)
            
            # Backward pass
            dL_dweights, dL_dbias = self._compute_gradients(X, y_prob, y)
            
            # Update
            self._update_parameters(dL_dweights, dL_dbias)
            
            if verbose and (epoch < 3 or epoch % 25 == 0 or epoch == n_epochs - 1):
                accuracy = np.mean((y_prob >= 0.5) == y)
                print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
        
        print(f"\n✅ Training complete!")
    
    def predict_proba(self, X):
        """Return probabilities."""
        return self._forward(X)
    
    def predict(self, X, threshold=0.5):
        """Return class predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)


# =================================
# TEST: Spam Detection Example
# =================================

print("=" * 60)
print("LOGISTIC REGRESSION: Email Spam Classifier")
print("=" * 60)

# Synthetic spam data: [word_count, link_count, caps_ratio]
np.random.seed(42)

# Generate spam emails (class 1)
spam_features = np.random.randn(50, 3) + np.array([2, 3, 0.5])  # More links, caps
# Generate normal emails (class 0)
normal_features = np.random.randn(50, 3) + np.array([0, 0, 0])

X = np.vstack([spam_features, normal_features])
y = np.array([1] * 50 + [0] * 50)

# Shuffle
shuffle_idx = np.random.permutation(100)
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Dataset: {len(X)} emails")
print(f"Features: [word_count, link_count, caps_ratio]")

# Train
model = LogisticRegressionFromScratch(learning_rate=0.5)
model.fit(X, y, n_epochs=100)

# Test on new emails
print("\nPredictions on new emails:")
X_new = np.array([
    [2.5, 4.0, 0.6],    # Looks like spam
    [-0.5, 0.2, -0.1],  # Looks normal
    [1.0, 2.0, 0.3],    # Borderline
])

for features, prob in zip(X_new, model.predict_proba(X_new)):
    label = "SPAM" if prob >= 0.5 else "Normal"
    print(f"  Features: {features} → P(spam) = {prob:.2%} → {label}")
