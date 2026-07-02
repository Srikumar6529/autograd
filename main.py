from tensor.tensor import Tensor
from tensor.layers import *
# Create a 2-layer network matching your exact structure
model = Sequential([
    Dense(n_in=4, n_out=3, activation='relu'), # Hidden layer
    Dense(n_in=3, n_out=2, activation=None)    # Output layer emitting raw logits
])

# Create a sample batch of 2 elements with 4 features each
X_batch = Tensor([
    [0.5, 1.2, -0.3, 0.8],
    [-1.1, 0.4, 2.1, -0.5]
])

# Create true one-hot encoded labels (e.g., sample 1 is class 0, sample 2 is class 1)
Y_true = Tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

# 1. Fire Forward Pass to collect raw model logits
logits = model(X_batch)

# 2. Convert raw logits into clean probability metrics
probabilities = logits.softmax()

# 3. Calculate Categorical Cross Entropy Loss
loss = probabilities.categorical_crossentropy(Y_true)

print("--- Forward Pipeline Evaluation ---")
print("Raw Logits:\n", logits.data)
print("\nSoftmax Probabilities:\n", probabilities.data)
print("\nFinal Categorical Cross Entropy Loss Scalar:", loss.data)


