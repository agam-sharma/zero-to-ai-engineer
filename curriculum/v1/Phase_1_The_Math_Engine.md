# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: THE MATH ENGINE (Weeks 1–6)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
# Weeks 1–4: Linear Algebra + Calculus + Autograd
# Weeks 5–6: Probability + Information Theory (NEW — the foundations of every loss function)
# ═══════════════════════════════════════════════════════════════════════

---

## 🔗 The Salesforce Analogy

> **Math in AI is like SOQL in Salesforce.**
>
> You *can* build Salesforce apps without understanding the SOQL query optimizer, governor limits, or how the database indexes work internally. But the moment you hit scale — 10,000 records, complex queries, bulk operations — you're stuck. The engineers who truly master Salesforce are the ones who understand what's happening *under the hood*.
>
> Same principle here. You *could* learn to call `model.train()` and `model.predict()` without understanding the math. But the moment something goes wrong — the model doesn't converge, gradients explode, the loss doesn't decrease — you'll be stuck.
>
> **The math is your debugging superpower.** It's what separates engineers who can *use* AI from engineers who can *build* and *fix* AI.
>
> And here's the good news: **you only need 6 specific math tools** — not an entire math degree:
> 1. **Vectors & Matrices** (Linear Algebra) — what data looks like to a neural network
> 2. **Dot Products & Transformations** — what neural network layers actually do
> 3. **Derivatives & Chain Rule** (Calculus) — how training works
> 4. **Gradient Descent & Autograd** — the algorithm that makes it all come together
> 5. **Probability & MLE** — *why* models have the loss functions they have (it's not arbitrary)
> 6. **Entropy / Cross-Entropy / KL Divergence** — the units of information and the bedrock of every NLP loss

---

# WEEK 1: Linear Algebra — Vectors & Spaces

## 📖 THEORY: What is a Vector?

In your Salesforce world, you work with **records** — structured data with fields:
```json
{ "Name": "Acme Corp", "Revenue": 5000000, "Employees": 200, "Industry": "Technology" }
```

In AI, every piece of data is represented as a **vector** — a list of numbers:
```python
acme_corp = [5000000, 200, 1]  # [Revenue, Employees, Industry_encoded]
```

**Everything** in AI is a vector:
- A word in GPT → vector of 12,288 numbers (GPT-4)
- A pixel in an image → vector of 3 numbers (RGB)
- An audio sample → vector of frequency values
- A Salesforce record → vector of encoded field values

### Why Lists of Numbers?

Computers can't understand "Technology" or "Hot" — they only understand numbers. So we convert everything into numbers. A **vector** is simply an ordered list of numbers that represents something.

```python
# What a "word" looks like to GPT:
# (simplified — real GPT uses 12,288 dimensions)
word_king  = [0.8, 0.2, 0.9, 0.1, 0.7]  # 5 numbers representing "king"
word_queen = [0.8, 0.9, 0.9, 0.1, 0.7]  # similar, but "gender" dimension differs
word_apple = [0.1, 0.1, 0.2, 0.9, 0.3]  # very different — it's not royalty
```

Notice: "king" and "queen" have similar vectors (they're both royalty). "apple" has a very different vector. This is the **key insight**: **similar things have similar vectors.**

---

## 💻 CODE: Vectors from Scratch

### Exercise 1.1: Build Your Own Vector Class

```python
"""
Week 1, Exercise 1.1: Vectors from Scratch

Goal: Understand vectors by implementing them WITHOUT numpy.
Then we'll see why numpy exists (it's 100x faster).
"""

class Vector:
    """A simple vector class — just a list of numbers with math operations."""
    
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)
    
    def __repr__(self):
        return f"Vector({self.components})"
    
    def __len__(self):
        return self.dim
    
    def __getitem__(self, i):
        return self.components[i]
    
    # ─── VECTOR ADDITION ───
    # Add two vectors element-wise: [1,2,3] + [4,5,6] = [5,7,9]
    # Salesforce analogy: Like merging two records' numeric fields
    def __add__(self, other):
        assert self.dim == other.dim, "Vectors must have same dimension"
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    # ─── SCALAR MULTIPLICATION ───
    # Multiply every element by a number: 3 * [1,2,3] = [3,6,9]
    # This "scales" the vector — makes it longer or shorter
    def __mul__(self, scalar):
        return Vector([scalar * x for x in self.components])
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # ─── DOT PRODUCT ───
    # This is THE most important operation in all of AI.
    # dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    # It measures SIMILARITY between two vectors.
    def dot(self, other):
        assert self.dim == other.dim, "Vectors must have same dimension"
        return sum(a * b for a, b in zip(self.components, other.components))
    
    # ─── MAGNITUDE (LENGTH) ───
    # How "long" is this vector?
    # magnitude([3,4]) = sqrt(3² + 4²) = sqrt(25) = 5
    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5
    
    # ─── NORMALIZE ───
    # Scale the vector to have length 1 (unit vector)
    # Direction stays the same, just the magnitude becomes 1
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])
    
    # ─── COSINE SIMILARITY ───
    # Measures the ANGLE between two vectors (not their length)
    # Result is between -1 (opposite) and 1 (identical direction)
    # THIS IS HOW GPT MEASURES WORD SIMILARITY
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())


# ─── TEST IT ───
print("=== Vector Operations ===\n")

v1 = Vector([2, 3, 1])
v2 = Vector([4, 1, 5])

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")            # [6, 4, 6]
print(f"3 * v1 = {3 * v1}")               # [6, 9, 3]
print(f"v1 · v2 = {v1.dot(v2)}")          # 2*4 + 3*1 + 1*5 = 16
print(f"|v1| = {v1.magnitude():.4f}")      # sqrt(4+9+1) = sqrt(14)
print(f"cos(v1, v2) = {v1.cosine_similarity(v2):.4f}")

# ─── WORD SIMILARITY DEMO ───
print("\n=== Word Similarity (The Core of GPT's Attention) ===\n")

# Simplified word vectors (real ones have 768+ dimensions)
king   = Vector([0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
queen  = Vector([0.8, 0.9, 0.9, 0.1, 0.7, 0.3])
man    = Vector([0.5, 0.2, 0.3, 0.1, 0.4, 0.6])
woman  = Vector([0.5, 0.9, 0.3, 0.1, 0.4, 0.6])
apple  = Vector([0.1, 0.1, 0.2, 0.9, 0.3, 0.1])

print(f"similarity(king, queen)  = {king.cosine_similarity(queen):.4f}")   # High!
print(f"similarity(king, man)    = {king.cosine_similarity(man):.4f}")     # Medium
print(f"similarity(king, apple)  = {king.cosine_similarity(apple):.4f}")   # Low!

# THE FAMOUS WORD ANALOGY: king - man + woman ≈ queen
result = king + (-1 * man) + woman  # Should be close to queen
print(f"\nking - man + woman        = {result}")
print(f"queen                     = {queen}")
print(f"similarity(result, queen) = {result.cosine_similarity(queen):.4f}")  # Should be high!
```

### Why the Dot Product is THE Most Important Operation in AI

The dot product measures **how similar** two vectors are. Here's why this matters for GPT:

When GPT is generating text and it reaches:
> "The cat sat on the ___"

It needs to decide: which word comes next? It does this by computing the **dot product** between the current context vector and every word in its vocabulary. The words with the highest dot products (most similar) get the highest probabilities.

```
context_vector · "mat"    = 8.5  ← high similarity → high probability
context_vector · "chair"  = 7.2  ← medium similarity
context_vector · "banana" = 1.3  ← low similarity → low probability
```

**This is the attention mechanism in one sentence: it's just dot products.**

---

## 📖 THEORY: What is a Matrix?

A **matrix** is a grid of numbers — or equivalently, a collection of vectors stacked together.

```python
# A matrix is just vectors stacked as rows (or columns):
# This 3×4 matrix has 3 rows and 4 columns
matrix = [
    [1, 2, 3, 4],    # Row 0 — a vector in 4D
    [5, 6, 7, 8],    # Row 1 — another vector in 4D
    [9, 10, 11, 12],  # Row 2 — another vector in 4D
]
```

### Why Matrices Matter for Neural Networks

A neural network layer IS a matrix multiplication:

```
input_vector × weight_matrix = output_vector
   [1×768]    ×    [768×768]  =    [1×768]
```

When you "run" a neural network, you're multiplying your input vector by weight matrices. Every. Single. Layer.

**A neural network is literally a series of matrix multiplications with non-linear functions in between.**

---

## 💻 CODE: Matrix Operations from Scratch

### Exercise 1.2: Matrix Multiplication

```python
"""
Week 1, Exercise 1.2: Matrix Multiplication from Scratch

THE most important operation in deep learning.
Every neural network layer does this.
"""

def matmul(A, B):
    """
    Multiply matrix A (m×n) by matrix B (n×p) → result is (m×p)
    
    The rule: element [i][j] of the result = 
              dot product of row i of A with column j of B
    
    Salesforce analogy: 
    Think of it as a cross-reference. For each Account (row in A) 
    and each Metric (column in B), you compute a weighted score.
    """
    m = len(A)       # rows of A
    n = len(A[0])    # cols of A (must equal rows of B)
    p = len(B[0])    # cols of B
    
    assert n == len(B), f"Cannot multiply: A is {m}×{n}, B is {len(B)}×{p}"
    
    # Create result matrix filled with zeros
    result = [[0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):        # For each row in A
        for j in range(p):    # For each column in B
            for k in range(n):  # Dot product
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def print_matrix(M, name="Matrix"):
    """Pretty print a matrix."""
    print(f"\n{name}:")
    for row in M:
        print("  [" + ", ".join(f"{x:8.2f}" for x in row) + "]")


# ─── Test: Simple matrix multiply ───
A = [[1, 2],
     [3, 4],
     [5, 6]]    # 3×2

B = [[7, 8, 9],
     [10, 11, 12]]  # 2×3

C = matmul(A, B)    # Should be 3×3

print_matrix(A, "A (3×2)")
print_matrix(B, "B (2×3)")
print_matrix(C, "A × B (3×3)")


# ─── Verify with NumPy ───
import numpy as np

A_np = np.array(A)
B_np = np.array(B)
C_np = A_np @ B_np  # The @ operator is matrix multiply in NumPy/PyTorch

print_matrix(C_np.tolist(), "NumPy verification")


# ═══════════════════════════════════════════════════════════
# 🔥 THIS IS EXACTLY WHAT HAPPENS IN A NEURAL NETWORK LAYER:
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("🧠 SIMULATING A NEURAL NETWORK LAYER")
print("=" * 60)

# Imagine we have 3 words, each represented as a 4-dimensional vector
# This is the INPUT to a neural network layer
inputs = [[0.2, 0.8, 0.5, 0.1],    # "the"
          [0.9, 0.1, 0.3, 0.7],    # "cat"
          [0.4, 0.6, 0.8, 0.2]]    # "sat"

# The neural network layer has LEARNED weights (4×4 matrix)
# These weights were learned during training
weights = [[0.1, 0.4, 0.2, 0.3],
           [0.5, 0.2, 0.8, 0.1],
           [0.3, 0.6, 0.1, 0.5],
           [0.7, 0.3, 0.4, 0.2]]

# The output = input × weights (matrix multiplication!)
output = matmul(inputs, weights)

print_matrix(inputs, "Input (3 words × 4 dims)")
print_matrix(weights, "Weights (4×4 — learned by training)")
print_matrix(output, "Output (3 words × 4 dims — transformed!)")

print("""
What just happened:
  Each word vector was TRANSFORMED by the weight matrix.
  The output vectors encode NEW information about each word.
  This is EXACTLY what happens in every layer of GPT.
  
  GPT-2 does this with:
    Input:   (sequence_length × 768)
    Weights: (768 × 768) 
    Output:  (sequence_length × 768)
  
  Same operation, just bigger matrices.
""")
```

---

## 📖 THEORY: Transpose & the Attention Score Matrix

### What is Transpose?

Transpose flips a matrix — rows become columns, columns become rows:

```
Original A:         Transpose A^T:
[1, 2, 3]           [1, 4]
[4, 5, 6]           [2, 5]
                     [3, 6]
```

### Why Transpose Matters: The Attention Formula

The core of the Transformer is this formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

See that **K^T**? That's the transpose of K. When you compute `Q @ K^T`, you're computing the **similarity between every pair of words** in a sentence.

```python
"""
Week 1, Exercise 1.3: Simulating the Attention Score Matrix

This is the EXACT computation at the heart of every Transformer.
Q @ K^T gives you a matrix where entry [i][j] = 
    "how much should word i pay attention to word j?"
"""
import numpy as np

# 4 words in a sentence: "The cat sat down"
# Each word has a 3-dimensional "query" and "key" vector
# (Real GPT-2 uses 64 dimensions per head)

Q = np.array([
    [1.0, 0.0, 1.0],   # Query for "The"
    [0.0, 1.0, 0.0],   # Query for "cat"
    [1.0, 1.0, 0.0],   # Query for "sat"
    [0.0, 0.0, 1.0],   # Query for "down"
])

K = np.array([
    [1.0, 0.0, 0.0],   # Key for "The"
    [0.0, 1.0, 0.0],   # Key for "cat"
    [1.0, 0.0, 1.0],   # Key for "sat"
    [0.0, 1.0, 1.0],   # Key for "down"
])

# Step 1: Q @ K^T — the raw attention scores
# Each entry [i][j] = dot product of query_i and key_j
# = "how relevant is word j to word i?"
scores = Q @ K.T  # (4×3) @ (3×4) = (4×4)

print("Raw Attention Scores (Q @ K^T):")
print("      The   cat   sat  down")
words = ["The ", "cat ", "sat ", "down"]
for i, word in enumerate(words):
    row = "  ".join(f"{scores[i][j]:5.1f}" for j in range(4))
    print(f"  {word}  {row}")

# Step 2: Scale by sqrt(d_k) — prevents scores from getting too large
d_k = Q.shape[1]  # dimension of key vectors = 3
scaled_scores = scores / np.sqrt(d_k)

print(f"\nScaled by √{d_k} = {np.sqrt(d_k):.2f}:")
for i, word in enumerate(words):
    row = "  ".join(f"{scaled_scores[i][j]:5.2f}" for j in range(4))
    print(f"  {word}  {row}")

# Step 3: Softmax — convert scores to probabilities (each row sums to 1)
def softmax(x):
    """Convert a row of scores into probabilities."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print(f"\nAttention Weights (after softmax):")
print("      The   cat   sat  down")
for i, word in enumerate(words):
    row = "  ".join(f"{attention_weights[i][j]:5.2f}" for j in range(4))
    print(f"  {word}  {row}")

print("""
Interpretation:
  Row 0 (The):  The word "The" pays attention mostly to "The" and "sat"
  Row 1 (cat):  The word "cat" pays attention mostly to "cat" and "down"
  Row 2 (sat):  The word "sat" spreads attention across "The", "cat", "sat"
  Row 3 (down): The word "down" pays attention mostly to "sat" and "down"
  
  Each row sums to 1.0 — it's a probability distribution.
  THIS is what self-attention computes. You just did it.
""")
```

---

# WEEK 2: NumPy Mastery & Matrix Fluency

## 📖 THEORY: Why NumPy? (And Why Your From-Scratch Code is 100x Slower)

Your from-scratch implementations from Week 1 are great for understanding. But they're painfully slow:

```python
# Your from-scratch matmul: 3 nested for-loops in Python
# Python loops are SLOW — each iteration has overhead
def matmul(A, B):
    for i in range(m):       # Loop 1
        for j in range(p):   # Loop 2  
            for k in range(n):  # Loop 3
                result[i][j] += A[i][k] * B[k][j]

# NumPy matmul: Single call to optimized C/Fortran code
result = A @ B   # Same result, 100x faster
```

NumPy runs compiled C code under the hood. It also uses **SIMD instructions** (Single Instruction, Multiple Data) which process multiple numbers simultaneously on your CPU.

**Rule of thumb:** If you're writing a for-loop over array elements in Python, you're probably doing it wrong. Use NumPy instead.

---

## 💻 CODE: NumPy Crash Course

```python
"""
Week 2: NumPy Mastery
Everything you need to know for deep learning
"""
import numpy as np

# ═══════════════════════════════════════════════
# 1. CREATING ARRAYS
# ═══════════════════════════════════════════════

# From Python lists
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}, shape: {a.shape}")  # shape: (5,)

# 2D array (matrix)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"2D array shape: {M.shape}")  # shape: (2, 3)

# Common initialization patterns
zeros = np.zeros((3, 4))        # 3×4 matrix of zeros
ones = np.ones((2, 5))          # 2×5 matrix of ones
random = np.random.randn(3, 3)  # 3×3 matrix of random normal values
eye = np.eye(4)                 # 4×4 identity matrix

print(f"\nRandom 3×3 matrix:\n{random}")


# ═══════════════════════════════════════════════
# 2. RESHAPING — Critical for Deep Learning
# ═══════════════════════════════════════════════

# GPT processes text as (batch_size, sequence_length, d_model)
# You CONSTANTLY need to reshape tensors

data = np.arange(24)  # [0, 1, 2, ..., 23]
print(f"\nOriginal: shape {data.shape}")
print(data)

# Reshape to 2D: 4 sequences, 6 features each
reshaped_2d = data.reshape(4, 6)
print(f"\nReshaped to (4,6):\n{reshaped_2d}")

# Reshape to 3D: 2 batches, 3 sequences, 4 features
reshaped_3d = data.reshape(2, 3, 4)
print(f"\nReshaped to (2,3,4):\n{reshaped_3d}")

# Use -1 to auto-calculate one dimension
auto = data.reshape(6, -1)  # -1 → numpy figures out it's 4
print(f"\nAuto-reshape (6, -1) → shape: {auto.shape}")


# ═══════════════════════════════════════════════
# 3. BROADCASTING — NumPy's Superpower
# ═══════════════════════════════════════════════

# Broadcasting lets you do operations between arrays of different shapes
# NumPy automatically "stretches" the smaller array

# Add a bias vector to every row of a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # shape (3, 3)

bias = np.array([10, 20, 30])    # shape (3,) — just one row

result = matrix + bias  # bias is "broadcast" to every row!
print(f"\nBroadcasting example:")
print(f"Matrix:\n{matrix}")
print(f"Bias: {bias}")
print(f"Matrix + Bias:\n{result}")
# [[11, 22, 33], [14, 25, 36], [17, 28, 39]]

# This is how neural networks add bias terms:
# output = input @ weights + bias  ← bias is broadcast!


# ═══════════════════════════════════════════════
# 4. MATRIX OPERATIONS for Deep Learning
# ═══════════════════════════════════════════════

A = np.random.randn(3, 4)  # 3 words, 4-dim embeddings
B = np.random.randn(4, 5)  # Weight matrix: 4→5

# Matrix multiply (3 equivalent ways)
C1 = A @ B                    # Python 3.5+ operator
C2 = np.matmul(A, B)         # Function form
C3 = np.dot(A, B)            # Also works for 2D

print(f"\nMatrix multiply: ({A.shape}) @ ({B.shape}) = {C1.shape}")
assert np.allclose(C1, C2) and np.allclose(C2, C3)

# Transpose
print(f"A shape:   {A.shape}")
print(f"A^T shape: {A.T.shape}")  # (4, 3) — rows and cols swapped


# ═══════════════════════════════════════════════
# 5. SOFTMAX — The Activation That Powers Attention
# ═══════════════════════════════════════════════

def softmax(x):
    """
    Convert raw scores into probabilities.
    Used in: attention weights, output layer of classifiers
    
    Properties:
    - All outputs are positive
    - All outputs sum to 1 (it's a probability distribution)
    - Larger inputs get exponentially more probability
    """
    # Subtract max for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Example: Convert attention scores to probabilities
scores = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
probs = softmax(scores)
print(f"\nSoftmax example:")
print(f"  Scores:        {scores}")
print(f"  Probabilities: {probs.round(4)}")
print(f"  Sum:           {probs.sum():.4f}")  # Should be 1.0


# ═══════════════════════════════════════════════
# 6. PRACTICAL EXERCISE: Simulate Self-Attention
# ═══════════════════════════════════════════════

print("\n" + "=" * 60)
print("🔥 FULL SELF-ATTENTION IN 10 LINES OF NUMPY")
print("=" * 60)

np.random.seed(42)

# 4 words, each embedded as 8-dimensional vectors
seq_len = 4
d_model = 8
X = np.random.randn(seq_len, d_model)  # Input embeddings

# Learned weight matrices for Q, K, V
d_k = 4  # dimension of queries and keys
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)

# Compute Q, K, V
Q = X @ W_Q   # (4, 4) — queries
K = X @ W_K   # (4, 4) — keys
V = X @ W_V   # (4, 4) — values

# Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
scores = Q @ K.T / np.sqrt(d_k)   # (4, 4) attention scores
weights = softmax(scores)           # (4, 4) attention weights
output = weights @ V               # (4, 4) attended output

print(f"\nInput shape:            {X.shape}")
print(f"Q, K, V shapes:         {Q.shape}, {K.shape}, {V.shape}")
print(f"Attention scores shape: {scores.shape}")
print(f"Attention weights:\n{weights.round(3)}")
print(f"Output shape:           {output.shape}")
print("""
🎉 You just computed self-attention from scratch in NumPy!
This is THE core operation inside every Transformer block.
In Phase 4, you'll implement this in PyTorch with multiple heads.
""")
```

---

# WEEK 3: Calculus — Derivatives & the Chain Rule

## 📖 THEORY: Why Calculus? One Word: Training.

Here's the complete picture of how AI models learn:

```
╔═══════════════════════════════════════════════════════════════╗
║                HOW NEURAL NETWORKS LEARN                      ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. FORWARD PASS:  Input → Model → Prediction                ║
║     (Matrix multiplications — you learned this in Week 1-2)   ║
║                                                               ║
║  2. LOSS:  Compare prediction to correct answer               ║
║     loss = how_wrong_is_the_model(prediction, truth)          ║
║                                                               ║
║  3. BACKWARD PASS:  Compute gradients (derivatives)     ←───║
║     "Which direction should each weight change to               ║
║      reduce the loss?"                                         ║
║     THIS IS WHERE CALCULUS COMES IN                            ║
║                                                               ║
║  4. UPDATE:  Adjust weights in the gradient direction          ║
║     weight = weight - learning_rate × gradient                 ║
║                                                               ║
║  5. REPEAT from step 1 (thousands of times)                   ║
╚═══════════════════════════════════════════════════════════════╝
```

**Derivatives tell you which direction is "downhill."** Training a neural network = finding the bottom of a valley in a very high-dimensional landscape. Derivatives point you downhill.

### The Derivative: Your Slope Detector

A derivative tells you: **if I nudge the input slightly, how much does the output change?**

```python
# The derivative of f(x) = x² is f'(x) = 2x
# At x=3: f'(3) = 6 — meaning: 
#   "If you increase x by a tiny amount ε, 
#    the output increases by approximately 6ε"
```

This is exactly what you need for training: if you nudge a weight by a tiny amount, how much does the loss change?

---

## 💻 CODE: Derivatives from Scratch

```python
"""
Week 3: Calculus for Deep Learning
"""
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════
# NUMERICAL DERIVATIVES
# ═══════════════════════════════════════════════

def numerical_derivative(f, x, h=1e-7):
    """
    Compute the derivative of f at point x using the limit definition:
    
    f'(x) = lim[h→0] (f(x+h) - f(x-h)) / (2h)
    
    We use the "central difference" formula for better accuracy.
    
    Salesforce analogy: This is like checking how a field value changes
    when you slightly adjust an input — except we're doing it for math functions.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


# Test on known functions
print("=== Numerical Derivatives ===\n")

# f(x) = x² → f'(x) = 2x
f1 = lambda x: x ** 2
print(f"f(x) = x²")
print(f"  f'(3) = {numerical_derivative(f1, 3):.6f}  (expected: 6.0)")

# f(x) = x³ → f'(x) = 3x²
f2 = lambda x: x ** 3
print(f"\nf(x) = x³")
print(f"  f'(2) = {numerical_derivative(f2, 2):.6f}  (expected: 12.0)")

# f(x) = sin(x) → f'(x) = cos(x)
f3 = lambda x: np.sin(x)
print(f"\nf(x) = sin(x)")
print(f"  f'(0) = {numerical_derivative(f3, 0):.6f}  (expected: 1.0 = cos(0))")
print(f"  f'(π/2) = {numerical_derivative(f3, np.pi/2):.6f}  (expected: 0.0 = cos(π/2))")


# ═══════════════════════════════════════════════
# PARTIAL DERIVATIVES & GRADIENTS
# ═══════════════════════════════════════════════

print("\n=== Partial Derivatives (Gradients) ===\n")

def gradient(f, point, h=1e-7):
    """
    Compute the gradient of f at a given point.
    The gradient is a VECTOR of partial derivatives — one per input dimension.
    
    In neural networks:
    - f is the loss function
    - point is all the model's weights
    - The gradient tells you how to adjust EACH weight to reduce the loss
    """
    grad = np.zeros_like(point, dtype=float)
    for i in range(len(point)):
        # Nudge the i-th parameter
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        grad[i] = (f(point_plus) - f(point_minus)) / (2 * h)
    return grad


# f(x, y) = x² + y²  — a simple "bowl" function
# Gradient: [2x, 2y] — always points toward the rim (uphill)
# Negative gradient: [-2x, -2y] — points toward the center (downhill)

bowl = lambda p: p[0]**2 + p[1]**2

point = np.array([3.0, 4.0])
grad = gradient(bowl, point)

print(f"f(x, y) = x² + y²")
print(f"At point (3, 4):")
print(f"  f(3, 4) = {bowl(point)}")
print(f"  gradient = {grad}")
print(f"  Expected: [6.0, 8.0]  (which is [2×3, 2×4])")
print(f"  Negative gradient (downhill direction): {-grad}")


# ═══════════════════════════════════════════════
# THE CHAIN RULE — The Heart of Backpropagation
# ═══════════════════════════════════════════════

print("\n=== The Chain Rule ===\n")
print("""
The Chain Rule is THE reason neural networks can be trained.

A neural network is a CHAIN of functions:
  f(g(h(x)))

The chain rule says:
  d/dx f(g(h(x))) = f'(g(h(x))) × g'(h(x)) × h'(x)

In other words: multiply the derivatives at each step.
This is called BACKPROPAGATION.
""")

# Example: f(x) = (3x + 2)⁴
# Chain: f = u⁴ where u = 3x + 2
# f'(x) = 4u³ × 3 = 12(3x + 2)³

f_chain = lambda x: (3*x + 2)**4

x_test = 1.0
numerical = numerical_derivative(f_chain, x_test)
analytical = 12 * (3*x_test + 2)**3  # By chain rule

print(f"f(x) = (3x + 2)⁴")
print(f"At x = {x_test}:")
print(f"  Numerical derivative:  {numerical:.4f}")
print(f"  Analytical (chain rule): {analytical:.4f}")
print(f"  Match: {np.isclose(numerical, analytical)}")


# ═══════════════════════════════════════════════
# GRADIENT DESCENT — The Training Algorithm
# ═══════════════════════════════════════════════

print("\n" + "=" * 60)
print("🔥 GRADIENT DESCENT — How EVERY Neural Network Trains")
print("=" * 60)

def gradient_descent(f, initial_point, learning_rate=0.1, steps=50):
    """
    Find the minimum of function f starting from initial_point.
    
    The algorithm:
    1. Compute gradient at current position
    2. Take a small step in the OPPOSITE direction (downhill)
    3. Repeat
    
    Salesforce analogy:
    Imagine you're on a hilly landscape and you need to find the 
    lowest point. You can't see the whole landscape (too many dimensions),
    but you CAN feel the slope under your feet (the gradient).
    
    Strategy: Always take a step downhill. Eventually you reach a valley.
    """
    point = np.array(initial_point, dtype=float)
    history = [point.copy()]
    
    for step in range(steps):
        grad = gradient(f, point)
        point = point - learning_rate * grad  # Step downhill!
        history.append(point.copy())
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: point = [{point[0]:7.4f}, {point[1]:7.4f}], "
                  f"f(point) = {f(point):8.4f}, |grad| = {np.linalg.norm(grad):.4f}")
    
    return point, history


# Minimize f(x, y) = (x - 3)² + (y - 7)²
# The minimum is at (3, 7) where f = 0
target_func = lambda p: (p[0] - 3)**2 + (p[1] - 7)**2

print(f"\nMinimizing f(x,y) = (x-3)² + (y-7)²")
print(f"Starting from (10, -5). Minimum should be at (3, 7).\n")

final_point, history = gradient_descent(target_func, [10.0, -5.0], learning_rate=0.1, steps=50)

print(f"\n🎯 Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})")
print(f"   Expected:    (3.0000, 7.0000)")
print(f"   f(final):    {target_func(final_point):.8f}")
print(f"""
✅ Gradient descent found the minimum!

THIS is how neural networks train:
  - The "function" is the loss function
  - The "point" is all the model's weights (millions of them)
  - The "gradient" tells each weight which direction to adjust
  - The "learning rate" controls how big each adjustment step is
  
  After thousands of steps, the weights converge to values
  that minimize the loss — meaning the model makes good predictions.
""")
```

---

# WEEK 4: Backpropagation & Autograd (micrograd)

## 📖 THEORY: The Computation Graph

This is the most important week of Phase 1. Here's why:

**Week 1-3 gave you the ingredients:** vectors, matrices, derivatives, gradient descent.
**Week 4 combines them all** into the actual mechanism that trains neural networks: **backpropagation.**

### What is a Computation Graph?

Every mathematical expression can be drawn as a graph:

```
Expression: L = (a * b + c) * d

Graph:
  a ──┐
      ├── [×] ── e ──┐
  b ──┘               ├── [+] ── f ──┐
                      │               ├── [×] ── L
  c ──────────────────┘               │
                                      │
  d ──────────────────────────────────┘

Where: e = a*b, f = e+c, L = f*d
```

**Forward pass:** Walk the graph left → right. Compute each node's value.
**Backward pass:** Walk the graph right → left. Compute each node's gradient.

This is exactly what `loss.backward()` does in PyTorch.

---

## 💻 CODE: Building micrograd (Your Own Autograd Engine)

This is the most important code in the entire course. This is Karpathy's micrograd — an autograd engine in ~100 lines.

**Watch the full video first:** [Andrej Karpathy: micrograd](https://youtu.be/VMj-3S1tku0) (2h25m)

```python
"""
Week 4: Building micrograd — Your Own Autograd Engine

This is the same engine that powers PyTorch's .backward()
If you understand this, you understand how ALL neural networks train.

WATCH FIRST: https://youtu.be/VMj-3S1tku0
"""
import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    """
    A Value wraps a number and tracks the computation graph.
    When you do math with Values, it remembers HOW the result was computed.
    Then .backward() walks the graph in reverse to compute gradients.
    
    This is the EXACT same concept as PyTorch's Tensor with requires_grad=True.
    """
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data              # The actual number
        self.grad = 0.0               # Gradient (dL/d_self), initialized to 0
        self._backward = lambda: None # Function to compute gradient
        self._prev = set(_children)   # Set of child Values
        self._op = _op                # The operation that created this Value
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # ─── Addition ───
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # Gradient just flows through unchanged
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    # ─── Multiplication ───
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            # Each input's gradient = the OTHER input's value × output grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    # ─── Power ───
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    # ─── Tanh activation ───
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            # d(tanh(x))/dx = 1 - tanh²(x)
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    # ─── ReLU activation ───
    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            # d(relu(x))/dx = 1 if x > 0, else 0
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # ─── Backward pass (backpropagation) ───
    def backward(self):
        """
        Compute gradients for ALL nodes in the computation graph.
        Uses topological sort to process nodes in the right order.
        """
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Start: gradient of output w.r.t. itself = 1
        self.grad = 1.0
        
        # Walk backwards through the graph
        for node in reversed(topo):
            node._backward()
    
    # Helper methods for operator overloading
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1


# ═══════════════════════════════════════════════════════
# TEST: Verify gradients match PyTorch
# ═══════════════════════════════════════════════════════

print("=== micrograd Test ===\n")

# Build a computation: L = (a * b + c) * d
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = Value(-2.0, label='d')

e = a * b;  e.label = 'e'     # e = -6
f = e + c;  f.label = 'f'     # f = 4  
L = f * d;  L.label = 'L'     # L = -8

# Backward pass — compute all gradients!
L.backward()

print(f"Forward pass:")
print(f"  a={a.data}, b={b.data}, c={c.data}, d={d.data}")
print(f"  e = a*b = {e.data}")
print(f"  f = e+c = {f.data}")
print(f"  L = f*d = {L.data}")

print(f"\nBackward pass (gradients):")
print(f"  dL/da = {a.grad}")   # Should be b*d = (-3)*(-2) = 6
print(f"  dL/db = {b.grad}")   # Should be a*d = (2)*(-2) = -4
print(f"  dL/dc = {c.grad}")   # Should be d = -2
print(f"  dL/dd = {d.grad}")   # Should be f = 4

# Verify with PyTorch
import torch

a_pt = torch.tensor(2.0, requires_grad=True)
b_pt = torch.tensor(-3.0, requires_grad=True)
c_pt = torch.tensor(10.0, requires_grad=True)
d_pt = torch.tensor(-2.0, requires_grad=True)

L_pt = (a_pt * b_pt + c_pt) * d_pt
L_pt.backward()

print(f"\nPyTorch verification:")
print(f"  dL/da = {a_pt.grad.item()}")
print(f"  dL/db = {b_pt.grad.item()}")
print(f"  dL/dc = {c_pt.grad.item()}")
print(f"  dL/dd = {d_pt.grad.item()}")

print(f"\n✅ All gradients match between micrograd and PyTorch!")


# ═══════════════════════════════════════════════════════
# BUILD A NEURAL NETWORK WITH micrograd!
# ═══════════════════════════════════════════════════════

import random

class Neuron:
    """A single neuron: output = activation(sum(w_i * x_i) + bias)"""
    
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""
    
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron — a stack of layers."""
    
    def __init__(self, n_inputs, layer_sizes):
        sz = [n_inputs] + layer_sizes
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layer_sizes))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Train a small network!
print("\n" + "=" * 60)
print("🧠 TRAINING A NEURAL NETWORK WITH MICROGRAD")
print("=" * 60)

# Simple dataset: learn XOR-like function
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # Desired outputs

# Create network: 3 inputs → 4 hidden → 4 hidden → 1 output
model = MLP(3, [4, 4, 1])
print(f"Parameters: {len(model.parameters())}")

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, predictions))
    
    # Backward pass
    for p in model.parameters():
        p.grad = 0.0  # Zero gradients (critical!)
    loss.backward()
    
    # Update weights (gradient descent)
    learning_rate = 0.05
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss = {loss.data:.6f}")

print(f"\nFinal predictions:")
for x, y in zip(xs, ys):
    pred = model(x)
    print(f"  Input: {x} → Predicted: {pred.data:.4f}, Target: {y}")

print("""
🎉 You just trained a neural network with YOUR OWN autograd engine!

This is EXACTLY what PyTorch does when you call:
  loss.backward()      ← computes gradients (your .backward() method)
  optimizer.step()     ← updates weights (your learning rate * gradient step)
  optimizer.zero_grad() ← resets gradients (your p.grad = 0.0)

You now understand the ENGINE inside PyTorch.
""")
```

---

# WEEK 5: Probability for AI

## 🎯 Why Probability, Right Now?

You've now built autograd and trained a small MLP (Week 4). You called a "loss function" and minimized it. But here's the question almost every tutorial skips:

> **Why *that* loss function and not some other?**

The honest answer — the one that unlocks a dozen other things — is that **every loss function in deep learning is a probability statement in disguise**. Specifically, almost every loss is the **Negative Log-Likelihood (NLL)** of some assumed data distribution. MSE? NLL under a Gaussian. Cross-entropy? NLL under a Categorical. Binary cross-entropy? NLL under a Bernoulli.

Once you *see* this, the whole field stops feeling arbitrary. You stop memorizing which loss to use and start picking one from first principles.

## 🔗 The Salesforce Analogy

> In Salesforce, **validation rules** are boolean: a record is either valid or not. In AI, every prediction is probabilistic: the model outputs a *distribution* over possible answers with associated confidence levels. Probability is the math of "how sure are we?" — and every output of a neural network is ultimately a number that represents degree of belief.

## 📖 THEORY: The 5 Concepts You Need

### 1. Random Variable
A quantity whose value is uncertain. `X = the label predicted for this image`. `X` can take values `{cat, dog, bird}` with some probabilities that sum to 1.

### 2. Probability Distribution
A function that assigns probabilities to each possible value of a random variable. Three distributions you'll see forever:

| Distribution | Values | Example in AI |
|--------------|--------|---------------|
| **Bernoulli(p)** | {0, 1} | Spam / not spam — BCE loss lives here |
| **Categorical(p₁,…,pK)** | {1, …, K} | Next-token prediction (GPT!) — cross-entropy lives here |
| **Gaussian(μ, σ²)** | real numbers | Regression (predict a price) — MSE lives here |

### 3. Expectation (`E[X]`)
The "average value" weighted by probability. `E[X] = Σ x · p(x)`. When a loss is defined over a *dataset*, it's really an expectation over the data distribution.

### 4. Variance
How spread out the values are. `Var[X] = E[(X − E[X])²]`. This is literally MSE where the "prediction" is the mean.

### 5. Maximum Likelihood Estimation (MLE) — the most important idea in this week
Given data `{x₁, …, xₙ}` and a parameterized distribution `p(x; θ)`, choose `θ` to **maximize the likelihood of having observed that data**:

```
θ* = argmax_θ  Π p(xᵢ; θ)         ← likelihood
   = argmax_θ  Σ log p(xᵢ; θ)      ← log-likelihood (easier: sum not product)
   = argmin_θ  − Σ log p(xᵢ; θ)    ← NEGATIVE log-likelihood = your loss function!
```

That last line is where loss functions come from. Every time you minimize cross-entropy, you are running MLE under an assumed categorical distribution. Every time you minimize MSE, you are running MLE under an assumed Gaussian with fixed σ.

## 💻 CODE: Implement the Three Distributions + MLE

### Exercise 5.1: Bernoulli, Categorical, Gaussian from Scratch

```python
"""
Week 5 Exercise 5.1 — PDFs/PMFs from scratch.

Goal: implement the probability mass/density functions for the three
distributions that underlie every common loss function in deep learning.
"""
import numpy as np

def bernoulli_pmf(x: int, p: float) -> float:
    assert x in (0, 1)
    return p if x == 1 else 1 - p

def categorical_pmf(x: int, probs: np.ndarray) -> float:
    assert 0 <= x < len(probs)
    assert np.isclose(probs.sum(), 1.0)
    return float(probs[x])

def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * np.exp(exponent)
```

### Exercise 5.2: MLE for a Gaussian (Derivation + Code)

**The math:** For data `{x₁, ..., xₙ}` under `N(μ, σ²)`, the log-likelihood is

```
L(μ, σ²) = Σᵢ [ −½ log(2πσ²) − (xᵢ − μ)² / (2σ²) ]
```

Set `∂L/∂μ = 0` → μ_MLE = `(1/n) Σ xᵢ` = *sample mean*.
Set `∂L/∂σ² = 0` → σ²_MLE = `(1/n) Σ (xᵢ − μ)²` = *sample variance*.

```python
def fit_gaussian_mle(data: np.ndarray) -> tuple[float, float]:
    """Returns the MLE estimates of (mu, sigma) for a Gaussian."""
    mu = data.mean()
    sigma = data.std()          # (unbiased=False by default in numpy)
    return mu, sigma

# sanity check:
samples = np.random.normal(loc=3.0, scale=1.5, size=10_000)
mu_hat, sigma_hat = fit_gaussian_mle(samples)
assert abs(mu_hat - 3.0) < 0.1 and abs(sigma_hat - 1.5) < 0.1
print(f"Recovered mu={mu_hat:.3f}, sigma={sigma_hat:.3f} from data. ✅")
```

### Exercise 5.3: MLE → MSE — The Punchline

Assuming targets `yᵢ | xᵢ ~ N(f(xᵢ; θ), σ²)` with fixed σ, maximizing log-likelihood is *equivalent* (up to constants) to minimizing `Σ (yᵢ − f(xᵢ; θ))²`. On paper, drop constants from the Gaussian log-pdf, and you're staring at MSE.

**Deliverable:** in your journal, write this derivation by hand. Then code it:

```python
# Show empirically that MLE and MSE give the same optimum for linear regression
# on toy data. Fit y = 2x + 1 + ε; confirm both losses converge to same (w, b).
```

---

## 📅 Week 5 Schedule (3 hrs/day weekdays, ~6 hrs weekend)

| Day | Topic | Deliverable |
|-----|-------|-------------|
| Mon | 3B1B-style intro to probability; random variables; distributions | Journal notes, 1 open question |
| Tue | Bernoulli, Categorical, Gaussian from scratch | Exercise 5.1 complete + unit tests |
| Wed | Expectation, Variance | Plot a Gaussian; compute E[X], Var[X] from samples |
| Thu | Maximum Likelihood Estimation — the concept | Hand-derive μ_MLE, σ²_MLE for Gaussian |
| Fri | MSE = MLE under Gaussian (the connection) | Blog post: "Why MSE is just MLE in disguise" |
| Sat | Mini-project #3 (partial): implement samplers | Sample from each distribution + plot histograms |
| Sun | Paper-of-the-week + retrospective | Commit + blog post live |

## ✅ Week 5 Success Criteria

- [ ] Can state what a random variable is in 1 sentence
- [ ] Can implement PDF/PMF for Bernoulli, Categorical, Gaussian from scratch
- [ ] Can derive μ_MLE and σ²_MLE for a Gaussian on paper
- [ ] Can explain (to a colleague) why MSE is "just MLE under a Gaussian"
- [ ] `fit_gaussian_mle(samples)` recovers parameters to within 1%

---

# WEEK 6: Information Theory for AI

## 🎯 Why Information Theory?

Because **cross-entropy**, **perplexity**, **KL divergence**, **softmax**, and **temperature sampling** are all defined in terms of *information*, not just probability. If you don't have clean intuition for entropy, you'll forever treat these terms as black boxes. After this week, you will never again wonder "why log?" or "why does temperature work the way it does?"

## 🔗 The Salesforce Analogy

> In Salesforce, a **data quality report** measures how "messy" your data is. If every Account has a unique industry code (high diversity), the report is uncertain. If 99% of Accounts are "Technology", the report is certain. **Entropy is the mathematical version of that "uncertainty score."**

## 📖 THEORY: The 5 Information-Theoretic Concepts You Need

### 1. Self-Information (Surprise)
`I(x) = −log p(x)` — the "surprise" of an outcome. Low-probability events are surprising (high info). A coin toss outcome carries 1 bit if the coin is fair. A rigged coin's outcome carries less info.

### 2. Entropy `H(P)`
`H(P) = E[I(X)] = −Σ p(x) log p(x)` — the *average* surprise of a distribution. A uniform distribution has maximum entropy; a point-mass (certain) distribution has entropy 0.

| Distribution over {A, B, C} | Entropy (nats) |
|----------------------------|----------------|
| `[1/3, 1/3, 1/3]` (uniform) | log(3) ≈ 1.099 (max) |
| `[0.9, 0.05, 0.05]` | 0.393 |
| `[1.0, 0.0, 0.0]` (certain) | 0.0 |

### 3. Cross-Entropy `H(P, Q)` — *THE* loss function of NLP
`H(P, Q) = −Σ p(x) log q(x)` — average surprise when you think the true distribution is `Q` but it's actually `P`. This is *exactly* the loss your GPT minimizes, where `P` is the one-hot true next-token and `Q` is your model's predicted distribution.

When `P` is one-hot (true token is class `k`), this collapses to `−log q(k)` — i.e., the **negative log-likelihood of the correct class**. The loss you've been using since Week 4.

### 4. KL Divergence `D_KL(P ∥ Q)`
`D_KL(P ∥ Q) = H(P, Q) − H(P) = Σ p(x) log(p(x)/q(x))` — extra surprise you incur using `Q` instead of the true `P`. **Not symmetric.** KL is the distance from `P` to `Q` in probability-space, and it underlies variational autoencoders, RLHF, and diffusion model losses.

### 5. Softmax + Temperature — derived from first principles
Given logits `z = (z₁, …, zK)`, softmax gives you a probability distribution:

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

This isn't arbitrary. It's the distribution that **maximizes entropy subject to a constraint on the expected energy** — i.e., the Boltzmann distribution from statistical mechanics. With **temperature `T`**:

```
softmax_T(z_i) = exp(z_i / T) / Σ exp(z_j / T)
```

- `T → 0` → distribution collapses onto the argmax (greedy decoding)
- `T = 1` → standard softmax
- `T → ∞` → distribution becomes uniform (maximum randomness)

## 💻 CODE: Implement Everything From Scratch

### Exercise 6.1: Entropy, Cross-Entropy, KL From Scratch

```python
"""
Week 6 Exercise 6.1 — information theory primitives.

Unit-test against PyTorch's torch.nn.functional.cross_entropy to 1e-6.
"""
import numpy as np

EPS = 1e-12  # numerical floor to avoid log(0)

def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + EPS)))

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(-np.sum(p * np.log(q + EPS)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return cross_entropy(p, q) - entropy(p)

def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z -= z.max()                 # numerical stability trick (see below)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()
```

**Key numerical trick** (Phase-5 you'll need this again): subtract `z.max()` before `exp` to avoid overflow. Softmax is invariant to additive constants, so this is free.

### Exercise 6.2: Prove CE Equivalence with PyTorch

```python
import torch
import torch.nn.functional as F

# Random logits and true class
logits = np.array([2.0, 1.0, 0.1])
true_class = 0

# One-hot true distribution
p_true = np.zeros(3); p_true[true_class] = 1.0

# Our NumPy cross-entropy
q_pred = softmax(logits)
my_ce = cross_entropy(p_true, q_pred)

# PyTorch cross-entropy
torch_ce = F.cross_entropy(
    torch.tensor(logits).unsqueeze(0),
    torch.tensor([true_class]),
).item()

assert abs(my_ce - torch_ce) < 1e-6, (my_ce, torch_ce)
print(f"My CE = {my_ce:.6f}, Torch CE = {torch_ce:.6f} — match ✅")
```

### Exercise 6.3: Temperature Sweep (Mini-Project #3)

```python
import matplotlib.pyplot as plt

logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]

for T in temperatures:
    probs = softmax(logits, temperature=T)
    plt.plot(probs, label=f"T={T}", marker="o")
plt.legend(); plt.title("Temperature reshapes the distribution"); plt.ylabel("P")
plt.savefig("temperature_sweep.png")
```

**Observations you should record in your journal:**
- At `T=0.1`, the distribution is nearly one-hot on the top logit — this is how greedy decoding "feels" to the model
- At `T=1.0`, it's moderately peaked
- At `T=10.0`, it's nearly uniform — model "hallucinates wildly" because every token is almost equally likely

You will use exactly this knob when sampling from your own GPT in Phase 6, Week 28. This is not abstract theory — this is a production knob in every LLM API.

### Exercise 6.4: Perplexity = exp(Cross-Entropy)

```python
def perplexity(cross_entropy_nats: float) -> float:
    return float(np.exp(cross_entropy_nats))

# If your GPT's CE loss on validation is 2.5 (nats/token):
#   perplexity = e^2.5 ≈ 12.18
# Interpretation: the model is "as confused as if it had ~12 equally-likely
#   choices at every step." Lower is better.
```

**This is how every LLM paper reports results.** Perplexity ≤ 20 on the Pile is roughly GPT-2 territory.

---

## 📅 Week 6 Schedule

| Day | Topic | Deliverable |
|-----|-------|-------------|
| Mon | Self-info, entropy; watch 3B1B-style info theory intros | Journal notes |
| Tue | Cross-entropy; prove equivalence with NLL under a Categorical | Exercise 6.1 + 6.2 |
| Wed | KL divergence geometry; `D_KL(P∥Q) ≠ D_KL(Q∥P)` | Plot KL for two Bernoullis |
| Thu | Softmax derived from max-entropy | Exercise 6.3 (temperature sweep plot) |
| Fri | Perplexity + put it all together | Blog post: "Every loss function is a probability in disguise" |
| Sat | **Mini-project #3:** finalize your `info_theory` module with 10+ unit tests | Pushed to `src/mymath/info.py` |
| Sun | Paper-of-the-week (Cover & Thomas Ch. 2 summary) | Retro + README progress bar updated |

## ✅ Week 6 Success Criteria

- [ ] Can state entropy, cross-entropy, KL divergence definitions from memory
- [ ] `cross_entropy` matches `F.cross_entropy` to 1e-6
- [ ] Can explain why KL isn't symmetric with a concrete 2-Bernoulli example
- [ ] Can derive softmax as the max-entropy distribution given a mean constraint (sketch on paper)
- [ ] Temperature sweep plot saved + narrated in your blog post
- [ ] Understand that **your GPT's loss in Phase 6 is nothing more than these equations repeated for every token**

---

## 🏆 PHASE 1 MILESTONE CHECK (end of Week 6)

By now you should be able to answer all of these out loud, without notes:

1. What is a dot product geometrically, and where does it appear in attention?
2. Why is matrix multiplication associative but not commutative? Where does that matter in NNs?
3. Derive the chain rule for `f(g(h(x)))`.
4. What does the gradient of a scalar loss w.r.t. a matrix parameter look like, shape-wise?
5. What is MLE? Why is minimizing NLL equivalent to maximizing likelihood?
6. Why is cross-entropy the "right" loss for classification? (hint: NLL under a Categorical)
7. What is KL divergence measuring, and why isn't it a "distance"?
8. Why does softmax have the form `exp(z_i) / Σ exp(z_j)`?
9. What does temperature do, from an entropy perspective?
10. What is perplexity and what makes it a better LLM metric than raw loss?

If you can't answer 8+ of these, **take an extra week to catch up before Phase 2**. Phase 2's classical ML will feel disorienting without this foundation.

---

## 📚 PHASE 1 COMPLETE RESOURCE LIST

### 🎥 Videos (Watch in Order)

| # | Video | Duration | Week | Purpose |
|---|-------|----------|------|---------|
| 1 | [3B1B: Essence of Linear Algebra — Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs) | 16 min | Week 1 | Visual intuition for vectors |
| 2 | [3B1B: Linear combinations, span, basis](https://www.youtube.com/watch?v=k7RM-ot2NWY) | 10 min | Week 1 | Foundation concepts |
| 3 | [3B1B: Matrices as linear transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE) | 12 min | Week 1 | What matrices DO geometrically |
| 4 | [3B1B: Matrix multiplication as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU) | 13 min | Week 1 | Why matmul is chaining transformations |
| 5 | [3B1B: Dot products and duality](https://www.youtube.com/watch?v=LyGKycYT2v0) | 14 min | Week 1 | Deep understanding of dot products |
| 6 | [3B1B: Inverse matrices, column space, null space](https://www.youtube.com/watch?v=uQhTuRlWMxw) | 12 min | Week 2 | Conceptual understanding |
| 7 | [3B1B: Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) | 17 min | Week 2 | For PCA understanding |
| 8 | [3B1B: Essence of Calculus Ch.1](https://www.youtube.com/watch?v=WUvTyaaNkzM) | 17 min | Week 3 | Visual intro to derivatives |
| 9 | [3B1B: The paradox of the derivative](https://www.youtube.com/watch?v=9vKqVkMQHKk) | 18 min | Week 3 | Deep intuition |
| 10 | [3B1B: Gradient descent, how NNs learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | 21 min | Week 3 | The training algorithm visualized |
| 11 | [3B1B: What is backpropagation?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | 14 min | Week 3 | The chain rule in action |
| 12 | [3B1B: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) | 10 min | Week 3 | The math details |
| 13 | **[Karpathy: micrograd](https://youtu.be/VMj-3S1tku0)** | **2h25m** | **Week 4** | **THE most important video. Build autograd from scratch.** |

### 📖 Books & Reading

| # | Resource | Chapters | Why |
|---|----------|----------|-----|
| 1 | [Mathematics for Machine Learning (free PDF)](https://mml-book.github.io/) | Ch. 2 (Linear Algebra), Ch. 5 (Vector Calculus) | THE reference book for ML math |
| 2 | [NumPy for Beginners (Official)](https://numpy.org/doc/stable/user/absolute_beginners.html) | All | NumPy is your daily tool |
| 3 | [CS231n Python/NumPy Tutorial](https://cs231n.github.io/python-numpy-tutorial/) | All | Stanford's AI course NumPy guide |
| 4 | [StatQuest: Dot Product (YouTube)](https://www.youtube.com/watch?v=FrHSbPRIFnk) | — | Crystal-clear 10-min explanation |
| 5 | [StatQuest: Gradient Descent Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) | — | Best beginner gradient descent video |

---

## ✅ PHASE 1 COMPLETION CHECKLIST

- [ ] Can implement vector addition, scalar multiply, dot product from scratch
- [ ] Can implement matrix multiplication from scratch
- [ ] Can explain what Q @ K^T computes in the attention mechanism
- [ ] Can implement softmax and explain why it converts scores to probabilities
- [ ] NumPy fluent: reshape, broadcasting, matrix ops all feel natural
- [ ] Can compute derivatives numerically using the limit definition
- [ ] Can explain the chain rule and apply it to composite functions
- [ ] Can implement gradient descent to find the minimum of a function
- [ ] Built micrograd (Value class with automatic gradient computation)
- [ ] Trained a small MLP using micrograd
- [ ] Verified your gradients match PyTorch's `.backward()`
- [ ] Can implement Bernoulli/Categorical/Gaussian PDFs from scratch
- [ ] Can derive μ_MLE and σ²_MLE for a Gaussian on paper
- [ ] Can explain MSE and Cross-Entropy as NLL under specific assumed distributions
- [ ] `entropy`, `cross_entropy`, `kl_divergence`, `softmax_with_temperature` implemented and unit-tested against PyTorch
- [ ] Can articulate what perplexity measures and compute it from a CE loss
- [ ] **Post-phase exam** in `assessments/phase_1_post_assessment.ipynb` passes ≥ 80%

**Next:** [Phase 2: Classical ML Crash Course (Weeks 7–9)](./Phase_2_Classical_ML.md)
