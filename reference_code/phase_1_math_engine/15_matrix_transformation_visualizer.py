# Source: AI_Learning_Cursor lines 1207-1299
# Original transcript phase: None - None
# Nearest header: #### 🏗️ Build Assignment: Transformation Visualizer
# Title: Build a visual tool to understand how matrices transform space.
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
Build a visual tool to understand how matrices transform space.
This will cement your understanding of what neural network layers DO.
"""
import numpy as np
import matplotlib.pyplot as plt

def visualize_transformation(matrix, title="Matrix Transformation"):
    """
    Show how a 2x2 matrix transforms the unit square.
    This is what's happening inside neural networks!
    """
    # Original unit square corners
    square = np.array([
        [0, 0],  # Origin
        [1, 0],  # Right
        [1, 1],  # Top-right
        [0, 1],  # Top
        [0, 0],  # Close the square
    ]).T  # Transpose to (2, 5) for matrix multiplication
    
    # Transform the square
    transformed = matrix @ square
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original square
    axes[0].fill(square[0], square[1], alpha=0.3, color='blue')
    axes[0].plot(square[0], square[1], 'b-', linewidth=2, label='Original')
    axes[0].scatter([0], [0], color='red', s=100, zorder=5)
    axes[0].annotate('Origin', (0.05, -0.15))
    axes[0].set_xlim(-2, 3)
    axes[0].set_ylim(-2, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)
    axes[0].set_title('Original Unit Square')
    axes[0].legend()
    
    # Transformed square
    axes[1].fill(transformed[0], transformed[1], alpha=0.3, color='red')
    axes[1].plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
    axes[1].scatter([0], [0], color='red', s=100, zorder=5)
    
    # Find appropriate limits
    all_coords = np.concatenate([transformed, square], axis=1)
    max_val = np.abs(all_coords).max() + 0.5
    axes[1].set_xlim(-max_val, max_val)
    axes[1].set_ylim(-max_val, max_val)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    axes[1].set_title(f'{title}\nMatrix: {matrix.tolist()}')
    axes[1].legend()
    
    plt.tight_layout()
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved to {filename}")
    plt.show()


# Visualize different transformations
print("1. SCALING (2x in both directions)")
scale = np.array([[2, 0], [0, 2]])
visualize_transformation(scale, "Scaling 2x")

print("\n2. ROTATION (45 degrees)")
angle = np.pi / 4  # 45 degrees in radians
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
visualize_transformation(rotation, "Rotation 45 degrees")

print("\n3. SHEARING (horizontal)")
shear = np.array([[1, 0.5], [0, 1]])
visualize_transformation(shear, "Horizontal Shear")

print("\n4. REFLECTION (over x-axis)")
reflect = np.array([[1, 0], [0, -1]])
visualize_transformation(reflect, "Reflection over X-axis")

print("\n5. NEURAL NETWORK-LIKE (random weights)")
# This is similar to what a neural network layer does
# It warps space in complex ways!
nn_weights = np.array([[0.8, -0.6], [0.3, 1.2]])
visualize_transformation(nn_weights, "Neural Network Transformation")
