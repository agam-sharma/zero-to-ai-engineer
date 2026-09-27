# Source: AI_Learning_Cursor lines 484-504
# Original transcript phase: None - None
# Nearest header: #### Core NumPy Concepts You MUST Master
# Title: Core NumPy Concepts You MUST Master
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# Often you need to reshape data for different layers

# A "flattened" image (like 28x28 MNIST digit flattened to 784 pixels)
flat_image = np.random.randn(784)
print(f"Flat shape: {flat_image.shape}")  # (784,)

# Reshape to 2D image
image_2d = flat_image.reshape(28, 28)
print(f"2D shape: {image_2d.shape}")  # (28, 28)

# Reshape for neural network input (batch_size, features)
nn_input = flat_image.reshape(1, 784)  # 1 sample, 784 features
print(f"NN input shape: {nn_input.shape}")  # (1, 784)

# Common patterns:
# -1 means "figure out this dimension automatically"
batch = np.random.randn(32, 28, 28)  # 32 images
flattened_batch = batch.reshape(32, -1)  # (32, 784)
print(f"Flattened batch: {flattened_batch.shape}")
