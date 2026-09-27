# Source: AI_Learning_Cursor lines 1379-1486
# Original transcript phase: None - None
# Nearest header: #### Gradient Descent: How AI Learns
# Title: GRADIENT DESCENT: The Algorithm That Powers All of AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
GRADIENT DESCENT: The Algorithm That Powers All of AI

Goal: Find the minimum of a function (minimize errors/loss)
Method: 
1. Start somewhere random
2. Compute the derivative (which way is "downhill"?)
3. Take a small step downhill
4. Repeat until you reach the bottom

This is EXACTLY what happens when you train a neural network!
"""
import numpy as np
import matplotlib.pyplot as plt

# Let's minimize f(x) = (x - 3)² + 1
# The minimum is at x = 3 (where derivative = 0)

def loss_function(x):
    """The function we want to minimize."""
    return (x - 3) ** 2 + 1

def loss_derivative(x):
    """The derivative tells us the slope/direction."""
    return 2 * (x - 3)

def gradient_descent(starting_x, learning_rate, num_steps):
    """
    Find the minimum of loss_function using gradient descent.
    
    Parameters:
    - starting_x: Where to start searching
    - learning_rate: How big of a step to take (α or eta)
    - num_steps: How many steps to take
    """
    x = starting_x
    history = [(x, loss_function(x))]
    
    for step in range(num_steps):
        # 1. Compute derivative (gradient)
        gradient = loss_derivative(x)
        
        # 2. Update x by moving OPPOSITE to the gradient
        # If gradient is positive (uphill to the right), move left
        # If gradient is negative (uphill to the left), move right
        x = x - learning_rate * gradient
        
        # Record history
        history.append((x, loss_function(x)))
        
        if step < 5 or step % 10 == 0:
            print(f"Step {step+1}: x = {x:.4f}, loss = {loss_function(x):.4f}, gradient = {gradient:.4f}")
    
    return x, history


# Run gradient descent
print("=" * 60)
print("GRADIENT DESCENT DEMO")
print("=" * 60)
print("Goal: Find x that minimizes f(x) = (x - 3)² + 1")
print("(The answer should be x = 3)\n")

final_x, history = gradient_descent(
    starting_x=-5,      # Start far from the answer
    learning_rate=0.1,  # Step size
    num_steps=50
)

print(f"\n✅ Final answer: x = {final_x:.4f}")
print(f"   Expected: x = 3.0000")

# Visualize the journey
plt.figure(figsize=(12, 5))

# Plot 1: The loss function and the path
plt.subplot(1, 2, 1)
x = np.linspace(-6, 8, 100)
plt.plot(x, loss_function(x), 'b-', linewidth=2, label='Loss function')

# Plot the gradient descent path
history_x = [h[0] for h in history]
history_loss = [h[1] for h in history]
plt.scatter(history_x, history_loss, c=range(len(history)), cmap='Reds', s=50, zorder=5)
plt.plot(history_x, history_loss, 'r--', alpha=0.5)
plt.scatter([history_x[0]], [history_loss[0]], color='green', s=200, marker='o', label='Start', zorder=6)
plt.scatter([history_x[-1]], [history_loss[-1]], color='red', s=200, marker='*', label='End', zorder=6)

plt.xlabel('x')
plt.ylabel('Loss')
plt.title('Gradient Descent Finding the Minimum')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss over time
plt.subplot(1, 2, 2)
plt.plot(range(len(history_loss)), history_loss, 'b-', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Decreasing Over Time\n(This is the "learning curve")')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=100)
print("\nSaved gradient_descent.png")
plt.show()
