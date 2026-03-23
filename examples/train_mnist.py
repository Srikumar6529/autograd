import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from core.tensor import Tensor
from core.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from core.optim import SGD

def accuracy(logits, targets):
    # logits: (B, C)
    preds = np.argmax(logits.data, axis=1)
    return np.mean(preds == targets)
def evaluate(model, X, y, batch_size=64):
    total_acc = 0
    total = 0

    for i in range(0, len(X), batch_size):
        x = Tensor(X[i:i+batch_size], requires_grad=False)
        y_batch = y[i:i+batch_size]

        logits = model(x)
        preds = np.argmax(logits.data, axis=1)

        total_acc += np.sum(preds == y_batch)
        total += len(y_batch)

    return total_acc / total

# -----------------------------
# Load MNIST (no PyTorch)
# -----------------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(float) / 255.0
y = mnist.target.astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# -----------------------------
# Convert to Tensor
# -----------------------------
X_train = Tensor(X_train, requires_grad=False)
y_train = y_train  # keep as numpy for indexing

# -----------------------------
# Model
# -----------------------------
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

criterion = CrossEntropyLoss()

# -----------------------------
# Training params
# -----------------------------
lr = 0.01
epochs = 10
batch_size = 32
optimizer = SGD(model.parameters(), lr)
# -----------------------------
# Training loop
# -----------------------------
losses = []
accuracies = []
for epoch in range(epochs):
    perm = np.random.permutation(len(X_train.data))

    total_loss = 0
    total_acc = 0
    num_batches = 0

    for i in range(0, len(X_train.data), batch_size):
        idx = perm[i:i+batch_size]

        x = Tensor(X_train.data[idx], requires_grad=False)
        y_batch = y_train[idx]

        # Forward
        logits = model(x)
        loss = criterion(logits, y_batch)

        # Backward
        loss.backward()

        # SGD
        optimizer.step()
        optimizer.zero_grad()
        '''for param in model.parameters():
            param.data -= lr * param.grad
            param.grad = np.zeros_like(param.data)'''

        # Metrics
        total_loss += loss.data
        total_acc += accuracy(logits, y_batch)
        num_batches += 1
    losses.append(total_loss / num_batches)
    accuracies.append(total_acc / num_batches)
    print(f"Epoch {epoch}, Loss: {total_loss/num_batches:.4f}, Acc: {total_acc/num_batches:.4f}")

test_acc = evaluate(model, X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("results/loss.png")

plt.clf()

plt.plot(accuracies)
plt.title("Training Accuracy")
plt.savefig("results/accuracy.png")