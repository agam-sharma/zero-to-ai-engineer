# Source: AI_Learning_Cursor lines 714-733
# Original transcript phase: None - None
# Nearest header: #### Key Python Patterns Used in AI
# Title: Key Python Patterns Used in AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

def data_generator(data, batch_size):
    """
    Yield batches of data instead of loading everything into memory.
    
    Salesforce Analogy: Like a QueryLocator in Batch Apex.
    You don't load all 10 million records at once - you process in batches.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        yield batch


# Example usage
large_dataset = list(range(1000))  # Imagine this is millions of samples

for batch_num, batch in enumerate(data_generator(large_dataset, batch_size=100)):
    if batch_num < 3:  # Show first 3 batches
        print(f"Batch {batch_num}: {len(batch)} samples, first={batch[0]}, last={batch[-1]}")
