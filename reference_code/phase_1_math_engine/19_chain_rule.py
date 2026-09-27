# Source: AI_Learning_Cursor lines 1490-1572
# Original transcript phase: None - None
# Nearest header: #### The Chain Rule: Backpropagation Foundation
# Title: THE CHAIN RULE: How Backpropagation Works
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
THE CHAIN RULE: How Backpropagation Works

When you have nested functions like f(g(x)), the chain rule says:
d/dx [f(g(x))] = f'(g(x)) × g'(x)

In neural networks:
- You have many nested layers
- To update early layer weights, you need derivatives through ALL later layers
- The chain rule lets you "chain" these derivatives together
- This is called BACKPROPAGATION
"""
import numpy as np

# Example: y = (2x + 1)³
# Let's break it down:
# - Inner function: g(x) = 2x + 1
# - Outer function: f(g) = g³
# - Composed: y = f(g(x)) = (2x + 1)³

def g(x):
    """Inner function"""
    return 2 * x + 1

def f(g_val):
    """Outer function"""
    return g_val ** 3

def y(x):
    """Composed function"""
    return f(g(x))

# Derivatives
def g_prime(x):
    """Derivative of inner: dg/dx = 2"""
    return 2

def f_prime(g_val):
    """Derivative of outer: df/dg = 3g²"""
    return 3 * (g_val ** 2)

def y_prime_chain_rule(x):
    """
    Full derivative using chain rule:
    dy/dx = df/dg × dg/dx = 3g² × 2 = 6(2x+1)²
    """
    g_val = g(x)
    return f_prime(g_val) * g_prime(x)

# Verify with numerical derivative
def numerical_derivative(func, x, epsilon=1e-7):
    """Compute derivative numerically (for verification)"""
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)

# Test at x = 2
x_test = 2
print("Chain Rule Verification")
print("=" * 40)
print(f"Function: y = (2x + 1)³")
print(f"At x = {x_test}:")
print(f"  Chain rule derivative: {y_prime_chain_rule(x_test)}")
print(f"  Numerical derivative:  {numerical_derivative(y, x_test):.4f}")
print(f"  Match: {'✅' if abs(y_prime_chain_rule(x_test) - numerical_derivative(y, x_test)) < 0.01 else '❌'}")

# CONNECTION TO NEURAL NETWORKS
print("\n" + "=" * 50)
print("HOW THIS APPLIES TO NEURAL NETWORKS")
print("=" * 50)
print("""
In a 3-layer neural network:
  Layer 1: h1 = activation(x @ W1 + b1)
  Layer 2: h2 = activation(h1 @ W2 + b2)
  Output:  y = h2 @ W3 + b3
  Loss:    L = loss_function(y, target)

To update W1, we need dL/dW1.
Using chain rule:
  dL/dW1 = dL/dy × dy/dh2 × dh2/dh1 × dh1/dW1

This is BACKPROPAGATION - chaining derivatives backwards!
""")
