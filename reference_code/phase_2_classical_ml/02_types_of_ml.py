# Source: AI_Learning_Cursor lines 1816-1896
# Original transcript phase: None - None
# Nearest header: #### Types of Machine Learning
# Title: THREE MAIN TYPES OF MACHINE LEARNING
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
THREE MAIN TYPES OF MACHINE LEARNING

1. SUPERVISED LEARNING (Most Common)
   - You have inputs (X) AND correct answers (y)
   - Model learns to map X → y
   - Examples: Classification, Regression
   - Salesforce analogy: Einstein Lead Scoring (learns from won/lost leads)

2. UNSUPERVISED LEARNING
   - You have inputs (X) only, NO labels
   - Model finds patterns/structure in data
   - Examples: Clustering, Dimensionality Reduction
   - Salesforce analogy: Account segmentation based on behavior

3. REINFORCEMENT LEARNING
   - Agent takes actions in an environment
   - Gets rewards/punishments
   - Learns to maximize rewards
   - Example: Game-playing AI, ChatGPT fine-tuning (RLHF)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =================================
# SUPERVISED LEARNING: Regression
# =================================
print("1. SUPERVISED LEARNING: Predicting Deal Value")
print("=" * 50)

# Training data with labels
X_supervised = np.array([
    [10, 50],   # employees, meetings
    [50, 120],
    [30, 80],
    [100, 200],
    [25, 60],
]).reshape(-1, 2)

y_supervised = np.array([50000, 200000, 100000, 400000, 75000])  # deal values

# Train model
model_supervised = LinearRegression()
model_supervised.fit(X_supervised, y_supervised)

# Predict
new_deal = np.array([[40, 100]])  # 40 employees, 100 meetings
predicted_value = model_supervised.predict(new_deal)
print(f"New prospect: 40 employees, 100 meetings")
print(f"Predicted deal value: ${predicted_value[0]:,.0f}")

# =================================
# UNSUPERVISED LEARNING: Clustering
# =================================
print("\n2. UNSUPERVISED LEARNING: Customer Segmentation")
print("=" * 50)

# Customer data (no labels!)
X_unsupervised = np.array([
    [100, 10000],   # revenue, support_tickets
    [90, 9500],
    [95, 10200],    # Cluster 1: High revenue, many tickets
    [20, 100],
    [25, 150],
    [15, 80],       # Cluster 2: Low revenue, few tickets
    [60, 8000],
    [55, 7500],     # Cluster 3: Medium revenue, many tickets
])

# Cluster without labels
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_unsupervised)

print("Discovered segments:")
for i, (x, cluster) in enumerate(zip(X_unsupervised, clusters)):
    print(f"  Customer {i+1}: Revenue=${x[0]}K, Tickets={x[1]} → Segment {cluster}")
