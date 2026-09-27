# Source: AI_Learning_Cursor lines 880-920
# Original transcript phase: None - None
# Nearest header: #### Visualizing Vectors
# Title: Visualizing Vectors
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, labels, title="Vector Visualization"):
    """Visualize 2D vectors as arrows from origin."""
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, (v, label) in enumerate(zip(vectors, labels)):
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                   color=colors[i % len(colors)], label=label)
    
    # Set up the plot
    all_vals = np.array(vectors).flatten()
    max_val = max(abs(all_vals)) + 1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.gca().set_aspect('equal')
    plt.savefig('vectors.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Saved to vectors.png")


# Visualize some vectors
vectors = [
    np.array([3, 2]),
    np.array([-1, 4]),
    np.array([2, -3]),
]
labels = ['v1: [3, 2]', 'v2: [-1, 4]', 'v3: [2, -3]']

plot_vectors(vectors, labels, "Vectors in 2D Space")
