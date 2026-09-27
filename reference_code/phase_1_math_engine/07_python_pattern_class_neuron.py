# Source: AI_Learning_Cursor lines 642-686
# Original transcript phase: None - None
# Nearest header: #### Key Python Patterns Used in AI
# Title: Key Python Patterns Used in AI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

class Neuron:
    """
    A single neuron - the building block of neural networks.
    
    Salesforce Analogy: Think of this like a Formula Field.
    - Inputs are like field references
    - Weights are like the multipliers in your formula
    - Bias is like a constant you add
    - Activation is like a conditional statement (IF/THEN)
    """
    
    def __init__(self, num_inputs):
        """Initialize with random weights (like random formula coefficients)"""
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0.0
    
    def forward(self, inputs):
        """
        Compute the neuron's output.
        This is like evaluating a formula field.
        """
        # Step 1: Weighted sum (dot product)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Step 2: Activation function (we'll learn why later)
        # ReLU: if negative, output 0; if positive, output as-is
        output = max(0, weighted_sum)
        
        return output
    
    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"


# Create and test a neuron
neuron = Neuron(num_inputs=3)
print(neuron)

# Test with sample input
sample_input = np.array([1.0, 2.0, 3.0])
output = neuron.forward(sample_input)
print(f"Input: {sample_input}")
print(f"Output: {output}")
