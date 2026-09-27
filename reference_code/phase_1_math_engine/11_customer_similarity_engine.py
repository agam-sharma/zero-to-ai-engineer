# Source: AI_Learning_Cursor lines 924-1040
# Original transcript phase: None - None
# Nearest header: #### 🏗️ Build Assignment: Similarity Calculator
# Title: Build a customer similarity calculator using vectors.
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
Build a customer similarity calculator using vectors.
This is the foundation of recommendation systems!
"""
import numpy as np

class CustomerSimilarity:
    """
    Find similar customers using vector operations.
    
    Salesforce Use Case: "Show me accounts similar to this one"
    This is exactly how AI-powered recommendations work!
    """
    
    def __init__(self):
        self.customers = {}
    
    def add_customer(self, name: str, features: list):
        """
        Add a customer with their feature vector.
        Features: [revenue, employees, industry_score, engagement_score]
        """
        self.customers[name] = np.array(features, dtype=float)
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Formula: cos(θ) = (a · b) / (||a|| × ||b||)
        
        Returns a value between -1 and 1:
        - 1: Identical direction (very similar)
        - 0: Perpendicular (unrelated)
        - -1: Opposite direction (very different)
        """
        dot_product = np.dot(v1, v2)
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if magnitude_product == 0:
            return 0
        
        return dot_product / magnitude_product
    
    def find_similar(self, customer_name: str, top_n: int = 3):
        """Find the most similar customers to a given customer."""
        if customer_name not in self.customers:
            raise ValueError(f"Customer {customer_name} not found")
        
        target = self.customers[customer_name]
        similarities = []
        
        for name, features in self.customers.items():
            if name != customer_name:
                sim = self.cosine_similarity(target, features)
                similarities.append((name, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def visualize_similarity_matrix(self):
        """Create a similarity matrix for all customers."""
        names = list(self.customers.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                matrix[i, j] = self.cosine_similarity(
                    self.customers[name1],
                    self.customers[name2]
                )
        
        return names, matrix


# Test the system
print("=" * 50)
print("Customer Similarity System")
print("=" * 50)

cs = CustomerSimilarity()

# Add sample customers: [revenue($K), employees, support_tickets, nps_score]
cs.add_customer("Acme Corp", [500, 200, 10, 8])
cs.add_customer("TechStart", [50, 10, 5, 9])
cs.add_customer("BigEnterprise", [5000, 2000, 100, 7])
cs.add_customer("MediumBiz", [450, 180, 12, 8])      # Similar to Acme
cs.add_customer("TinyStartup", [40, 8, 3, 9])        # Similar to TechStart
cs.add_customer("MegaCorp", [4500, 1800, 90, 7])     # Similar to BigEnterprise

# Find similar customers
print("\nFinding customers similar to 'Acme Corp':")
similar = cs.find_similar("Acme Corp")
for name, score in similar:
    print(f"  {name}: {score:.4f} similarity")

print("\nFinding customers similar to 'TechStart':")
similar = cs.find_similar("TechStart")
for name, score in similar:
    print(f"  {name}: {score:.4f} similarity")

# Show similarity matrix
print("\nSimilarity Matrix:")
names, matrix = cs.visualize_similarity_matrix()
print(f"{'':15}", end="")
for name in names:
    print(f"{name[:10]:>12}", end="")
print()
for i, name in enumerate(names):
    print(f"{name[:15]:15}", end="")
    for j in range(len(names)):
        print(f"{matrix[i,j]:12.3f}", end="")
    print()
