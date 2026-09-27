# Source: AI_Learning_Cursor lines 689-711
# Original transcript phase: None - None
# Nearest header: #### Key Python Patterns Used in AI
# Title: Key Python Patterns Used in AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# AI datasets often need preprocessing

# Example: Normalize pixel values from 0-255 to 0-1
raw_pixels = [0, 128, 255, 64, 192]
normalized = [p / 255.0 for p in raw_pixels]
print(f"Normalized: {normalized}")

# Example: Filter valid training samples
samples = [
    {"text": "Hello", "label": 1},
    {"text": "", "label": 0},        # Empty - invalid
    {"text": "World", "label": 1},
    {"text": None, "label": 0},      # None - invalid
]
valid_samples = [s for s in samples if s["text"]]
print(f"Valid samples: {len(valid_samples)}")  # 2

# Example: Apply transformation to all samples
texts = [s["text"] for s in valid_samples]
lowercased = [t.lower() for t in texts]
print(f"Processed: {lowercased}")
