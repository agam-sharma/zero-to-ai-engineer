# Source: AI_Learning_Cursor lines 1585-1671
# Original transcript phase: None - None
# Nearest header: #### 🏗️ Build Assignment: Gradient Descent for 2D
# Title: Extend gradient descent to 2D - finding the minimum of a surface.
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
Extend gradient descent to 2D - finding the minimum of a surface.
This is closer to real neural network training!
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D loss function: L(w1, w2) = (w1 - 2)² + (w2 - 3)² + 1
# Minimum is at (w1=2, w2=3)

def loss_2d(w1, w2):
    return (w1 - 2) ** 2 + (w2 - 3) ** 2 + 1

def gradient_2d(w1, w2):
    """Return gradient: [dL/dw1, dL/dw2]"""
    dL_dw1 = 2 * (w1 - 2)
    dL_dw2 = 2 * (w2 - 3)
    return np.array([dL_dw1, dL_dw2])

def gradient_descent_2d(start_w1, start_w2, learning_rate, num_steps):
    """Gradient descent in 2D parameter space."""
    w = np.array([start_w1, start_w2], dtype=float)
    history = [w.copy()]
    
    for step in range(num_steps):
        grad = gradient_2d(w[0], w[1])
        w = w - learning_rate * grad
        history.append(w.copy())
        
        if step < 3 or step == num_steps - 1:
            print(f"Step {step+1}: w1={w[0]:.4f}, w2={w[1]:.4f}, loss={loss_2d(w[0], w[1]):.4f}")
    
    return w, np.array(history)


# Run 2D gradient descent
print("2D GRADIENT DESCENT")
print("=" * 50)
print("Finding minimum of L(w1, w2) = (w1-2)² + (w2-3)² + 1")
print("Expected minimum: (w1=2, w2=3)\n")

final_w, history = gradient_descent_2d(
    start_w1=-3,
    start_w2=-1,
    learning_rate=0.1,
    num_steps=50
)

print(f"\n✅ Final: w1={final_w[0]:.4f}, w2={final_w[1]:.4f}")

# Visualize
fig = plt.figure(figsize=(14, 5))

# 3D surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
w1_range = np.linspace(-5, 7, 50)
w2_range = np.linspace(-3, 9, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)
L = loss_2d(W1, W2)
ax1.plot_surface(W1, W2, L, alpha=0.7, cmap='viridis')
ax1.scatter(history[:, 0], history[:, 1], 
            [loss_2d(h[0], h[1]) for h in history],
            color='red', s=20, marker='o')
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Loss')
ax1.set_title('Gradient Descent on 2D Loss Surface')

# 2D contour plot (bird's eye view)
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(W1, W2, L, levels=20, cmap='viridis')
ax2.plot(history[:, 0], history[:, 1], 'r.-', linewidth=1, markersize=5)
ax2.scatter([history[0, 0]], [history[0, 1]], color='green', s=100, label='Start', zorder=5)
ax2.scatter([history[-1, 0]], [history[-1, 1]], color='red', s=100, marker='*', label='End', zorder=5)
ax2.scatter([2], [3], color='blue', s=100, marker='x', label='True Minimum', zorder=5)
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_title('Contour Plot with Gradient Descent Path')
ax2.legend()

plt.tight_layout()
plt.savefig('gradient_descent_2d.png', dpi=100)
print("\nSaved gradient_descent_2d.png")
plt.show()
