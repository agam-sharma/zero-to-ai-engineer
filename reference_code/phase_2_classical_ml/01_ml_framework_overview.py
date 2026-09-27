# Source: AI_Learning_Cursor lines 1703-1812
# Original transcript phase: None - None
# Nearest header: #### Mental Model: The ML Framework
# Title: THE MACHINE LEARNING FRAMEWORK
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
THE MACHINE LEARNING FRAMEWORK

Traditional Programming:
    Rules + Data → Answers
    Example: IF revenue > 1M AND employees > 100 THEN high_value = True

Machine Learning:
    Data + Answers → Rules (Model)
    Example: Given 1000 labeled deals, learn what makes a deal "high value"

Then use those learned rules for new predictions:
    Learned Rules + New Data → Predictions
"""

# Let's see this concretely

# TRADITIONAL APPROACH: Hand-written rules
def traditional_prediction(revenue, employees, years_customer):
    """You write the rules based on your experience."""
    if revenue > 100000 and employees > 50:
        return "High Value"
    elif revenue > 50000 or years_customer > 3:
        return "Medium Value"
    else:
        return "Low Value"


# ML APPROACH: Learn from data
class SimpleMLModel:
    """The model LEARNS rules from labeled data."""
    
    def __init__(self):
        self.weights = None
        self.threshold = None
    
    def fit(self, X, y):
        """
        Learn from data (X=features, y=labels).
        We'll compute weights that best predict y from X.
        """
        # For now, let's use a simple average approach
        # (Real ML uses optimization like gradient descent)
        
        # Find the average feature values for each class
        high_value_mask = np.array(y) == "High Value"
        
        if any(high_value_mask):
            high_value_avg = np.mean(X[high_value_mask], axis=0)
            other_avg = np.mean(X[~high_value_mask], axis=0)
            
            # Weights: difference between high-value and other
            self.weights = high_value_avg - other_avg
            
            # Threshold: midpoint score
            high_scores = X[high_value_mask] @ self.weights
            other_scores = X[~high_value_mask] @ self.weights
            self.threshold = (np.mean(high_scores) + np.mean(other_scores)) / 2
        
        print(f"Learned weights: {self.weights}")
        print(f"Learned threshold: {self.threshold:.2f}")
    
    def predict(self, X):
        """Use learned rules to predict on new data."""
        scores = X @ self.weights
        predictions = np.where(scores > self.threshold, "High Value", "Other")
        return predictions


import numpy as np

# Training data: [revenue/1000, employees, years_customer]
X_train = np.array([
    [500, 200, 5],   # High value company
    [300, 150, 3],   # High value
    [50, 20, 1],     # Low value
    [30, 10, 0.5],   # Low value
    [400, 180, 4],   # High value
    [20, 5, 0.5],    # Low value
    [250, 100, 2],   # High value
    [40, 15, 1],     # Low value
])

# Labels (what we want to predict)
y_train = np.array([
    "High Value", "High Value", "Low Value", "Low Value",
    "High Value", "Low Value", "High Value", "Low Value"
])

# Train the model
print("=" * 50)
print("TRAINING THE MODEL (Learning from Data)")
print("=" * 50)
model = SimpleMLModel()
model.fit(X_train, y_train)

# Predict on new data
print("\n" + "=" * 50)
print("PREDICTING ON NEW DATA")
print("=" * 50)
X_new = np.array([
    [350, 140, 3],   # New company - should be High Value?
    [25, 8, 0.5],    # New company - should be Low Value?
])

predictions = model.predict(X_new)
for features, pred in zip(X_new, predictions):
    print(f"Features {features} → Prediction: {pred}")
