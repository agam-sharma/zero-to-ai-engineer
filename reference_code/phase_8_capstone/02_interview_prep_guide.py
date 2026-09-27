# Source: AI_Learning_Cursor lines 30209-30472
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: AI Interview Guide
# Title: AI ENGINEERING INTERVIEW PREPARATION
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
AI ENGINEERING INTERVIEW PREPARATION
====================================
What to expect and how to prepare.
"""

# =================================
# 1. INTERVIEW TYPES
# =================================

print("=" * 60)
print("AI ENGINEERING INTERVIEWS")
print("=" * 60)

interview_types = """
TYPICAL AI ENGINEERING INTERVIEW PROCESS:

1. INITIAL SCREEN (30-60 min)
   - Background and experience
   - Why AI? Your journey
   - High-level technical questions
   - Culture fit

2. TECHNICAL PHONE SCREEN (60 min)
   - Coding problem (Python)
   - ML concepts (theory)
   - System design light
   
3. ONSITE/VIRTUAL LOOP (4-6 hours)
   
   Round 1: CODING
   - LeetCode-style problems
   - Focus: Arrays, strings, trees
   - Python implementation
   
   Round 2: ML FUNDAMENTALS
   - Explain backpropagation
   - Design a classifier
   - Discuss overfitting
   
   Round 3: DEEP LEARNING
   - Transformer architecture
   - Attention mechanism
   - Training LLMs
   
   Round 4: SYSTEM DESIGN
   - Design a recommendation system
   - Scale an ML pipeline
   - Deploy a model
   
   Round 5: BEHAVIORAL
   - Past projects
   - Handling failure
   - Team collaboration
"""

print(interview_types)

# =================================
# 2. COMMON QUESTIONS
# =================================

print("\n" + "=" * 60)
print("COMMON INTERVIEW QUESTIONS")
print("=" * 60)

common_questions = """
MACHINE LEARNING FUNDAMENTALS:

Q: Explain the bias-variance tradeoff.
A: Bias = error from assumptions in model (underfitting)
   Variance = error from sensitivity to training data (overfitting)
   Tradeoff: Simpler models have high bias, low variance
             Complex models have low bias, high variance
   Goal: Find sweet spot that minimizes total error

Q: How does backpropagation work?
A: 1. Forward pass: Compute predictions
   2. Calculate loss: Compare to targets
   3. Backward pass: Compute gradients using chain rule
   4. Update weights: Move opposite to gradient
   Key: Chain rule connects distant layers

Q: What is regularization? Types?
A: Regularization prevents overfitting by adding constraints
   - L1 (Lasso): Adds |w| to loss, promotes sparsity
   - L2 (Ridge): Adds w² to loss, shrinks weights
   - Dropout: Randomly drop neurons during training
   - Early stopping: Stop before overfitting

DEEP LEARNING:

Q: Explain the attention mechanism.
A: Attention computes weighted sum of values based on 
   query-key similarity:
   1. Query asks "what am I looking for?"
   2. Keys say "what do I contain?"
   3. Dot product gives similarity scores
   4. Softmax normalizes to weights
   5. Weighted sum of values gives output
   
Q: Why do transformers scale better than RNNs?
A: 1. Parallelization: Process all positions at once
   2. No sequential bottleneck
   3. Direct connections between any positions
   4. Better gradient flow (shorter paths)

Q: What is the difference between GPT and BERT?
A: GPT: Decoder-only, causal (left-to-right), generation
   BERT: Encoder-only, bidirectional, understanding
   GPT predicts next token
   BERT uses masked language modeling (fill blanks)

LLM-SPECIFIC:

Q: How would you reduce hallucinations?
A: 1. Better training data (factual, diverse)
   2. RLHF to penalize hallucinations
   3. Retrieval augmentation (RAG)
   4. Constitutional AI (self-critique)
   5. Calibration (know what you don't know)

Q: Explain RLHF.
A: Reinforcement Learning from Human Feedback:
   1. Collect human preference data
   2. Train reward model on preferences
   3. Optimize LLM using reward model
   4. KL penalty keeps model close to base
   Makes models helpful, harmless, honest

Q: How do you evaluate an LLM?
A: 1. Perplexity: How well it predicts
   2. Downstream tasks: QA, summarization
   3. Human evaluation: Quality, helpfulness
   4. Safety benchmarks: Toxicity, bias
   5. Capability evals: Reasoning, math
"""

print(common_questions)

# =================================
# 3. CODING PROBLEMS
# =================================

print("\n" + "=" * 60)
print("CODING PROBLEMS TO PRACTICE")
print("=" * 60)

coding_problems = """
ML-SPECIFIC CODING PROBLEMS:

1. Implement softmax from scratch
   def softmax(x):
       exp_x = np.exp(x - np.max(x))  # Numerical stability
       return exp_x / exp_x.sum()

2. Implement cross-entropy loss
   def cross_entropy(y_pred, y_true):
       return -np.sum(y_true * np.log(y_pred + 1e-15))

3. Implement a simple neural network layer
   class Linear:
       def __init__(self, in_features, out_features):
           self.W = np.random.randn(in_features, out_features) * 0.01
           self.b = np.zeros(out_features)
       
       def forward(self, x):
           return x @ self.W + self.b

4. Implement attention
   def attention(Q, K, V):
       d_k = Q.shape[-1]
       scores = Q @ K.T / np.sqrt(d_k)
       weights = softmax(scores)
       return weights @ V

5. Implement top-k sampling
   def top_k_sample(logits, k):
       top_k_idx = np.argsort(logits)[-k:]
       top_k_logits = logits[top_k_idx]
       probs = softmax(top_k_logits)
       return np.random.choice(top_k_idx, p=probs)

GENERAL CODING (LeetCode):
- Two Sum
- Valid Parentheses
- Merge Sorted Lists
- Binary Tree Traversal
- Dynamic Programming basics
"""

print(coding_problems)

# =================================
# 4. SYSTEM DESIGN
# =================================

print("\n" + "=" * 60)
print("SYSTEM DESIGN FOR ML")
print("=" * 60)

system_design = """
ML SYSTEM DESIGN FRAMEWORK:

1. CLARIFY REQUIREMENTS
   - What problem are we solving?
   - Scale: Users, requests/second
   - Latency requirements
   - Accuracy requirements

2. HIGH-LEVEL DESIGN
   - Data ingestion pipeline
   - Feature engineering
   - Model training
   - Model serving
   - Monitoring

3. DATA PIPELINE
   - Sources (databases, APIs, files)
   - ETL process
   - Feature store
   - Data versioning

4. MODEL TRAINING
   - Training infrastructure (GPUs)
   - Experiment tracking (MLflow)
   - Hyperparameter tuning
   - Model registry

5. SERVING
   - REST API / gRPC
   - Batching
   - Caching
   - A/B testing

6. MONITORING
   - Model performance metrics
   - Data drift detection
   - Alerting
   - Retraining triggers

EXAMPLE: Design a Content Recommendation System

Requirements:
- 100M users, 10M items
- < 100ms latency
- Personalized recommendations

Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  User Data  │───>│   Feature   │───>│   Model     │
│  Item Data  │    │   Store     │    │  Training   │
│  Behavior   │    │             │    │             │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│    User     │<───│   Ranking   │<───│   Serving   │
│   Request   │    │   Service   │    │   (Cache)   │
└─────────────┘    └─────────────┘    └─────────────┘
"""

print(system_design)
