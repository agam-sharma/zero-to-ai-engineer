# Source: AI_Learning_Cursor lines 1319-1375
# Original transcript phase: None - None
# Nearest header: #### Derivatives: The Direction of Change
# Title: Derivatives: The Direction of Change
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np
import matplotlib.pyplot as plt

# A derivative tells you: "How fast is y changing as x changes?"
# This is the SLOPE at any point

# Example: Simple function y = x²
def f(x):
    return x ** 2

# Its derivative: dy/dx = 2x
def f_derivative(x):
    return 2 * x

# Visualize
x = np.linspace(-3, 3, 100)
y = f(x)

plt.figure(figsize=(12, 5))

# Plot 1: The function and its slope at different points
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')

# Show tangent lines at a few points
for x_point in [-2, 0, 1, 2]:
    slope = f_derivative(x_point)
    y_point = f(x_point)
    # Tangent line: y = slope * (x - x_point) + y_point
    x_tangent = np.linspace(x_point - 1, x_point + 1, 10)
    y_tangent = slope * (x_tangent - x_point) + y_point
    plt.plot(x_tangent, y_tangent, '--', alpha=0.7, 
             label=f'slope at x={x_point}: {slope}')
    plt.scatter([x_point], [y_point], s=100, zorder=5)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and Tangent Lines (Derivatives)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: The derivative function
plt.subplot(1, 2, 2)
plt.plot(x, f_derivative(x), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative: f\'(x) = 2x\n(The slope at each point)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('derivatives.png', dpi=100)
print("Saved derivatives.png")
plt.show()
