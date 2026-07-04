# Autograd Engine

A lightweight deep learning framework built from scratch using **NumPy**.

This project implements reverse-mode automatic differentiation, tensor operations, neural network layers, SGD optimization, gradient accumulation, and complete training loops without relying on PyTorch or TensorFlow for the actual learning engine.

The goal of this project is to understand what happens inside frameworks like PyTorch when you call `loss.backward()` and `optimizer.step()`.

![Demo](assets/demo.gif)

---

## What this project demonstrates

- Reverse-mode automatic differentiation
- Dynamic computation graph construction
- Tensor operations with gradient propagation
- Broadcasting-aware backpropagation
- Matrix multiplication gradients
- Activation functions
- Softmax and categorical cross-entropy loss
- Dense neural network layers
- SGD optimizer
- End-to-end neural network training
- MNIST training example
- PyTorch-style API design at a small educational scale

---

## Project structure

```text
autograd/
├── demo.py                    # Small XOR demo runnable from the root
├── requirements.txt
├── tensor/
│   ├── tensor.py              # Core Tensor class and autograd engine
│   ├── layers.py              # Layer, Dense, Sequential
│   ├── optim.py               # SGD optimizer and accuracy helper
│   └── __init__.py
├── scripts/
│   ├── data_loader.py         # MNIST download/preprocessing helpers
│   └── train_and_export.py    # MNIST training example
└── tests/
    └── test_tensor.py         # Add gradient/unit tests here
```

---

## Quick start

```bash
git clone https://github.com/Srikumar6529/autograd.git
cd autograd
pip install -r requirements.txt
python demo.py
```

Expected output:

```text
AUTOGRAD ENGINE DEMO
Training a tiny neural network on XOR using only NumPy + this engine
Epoch 001 | loss=...
Epoch 800 | loss=... | accuracy=100.00%
```

---

## Demo: training XOR from scratch

`demo.py` trains a small neural network on the XOR problem using only this custom autograd engine.

```python
from tensor import Dense, SGD, Sequential, Tensor

model = Sequential([
    Dense(n_in=2, n_out=8, activation="relu"),
    Dense(n_in=8, n_out=2, activation=None),
])

optimizer = SGD(model.layers, lr=0.1)

logits = model(X)
probs = logits.softmax()
loss = probs.categorical_crossentropy(y)
loss.backward()
optimizer.step()
```

This shows the full deep learning loop:

```text
Forward pass -> Loss computation -> Backward pass -> Parameter update
```

---

## MNIST training example

The `scripts/train_and_export.py` file trains a fully connected neural network on MNIST.

```bash
python scripts/train_and_export.py
```

The script downloads MNIST, normalizes the images, one-hot encodes labels, trains a multilayer neural network, and evaluates final accuracy.

Architecture:

```text
784 input features -> Dense(64, ReLU) -> Dense(16, ReLU) -> Dense(10) -> Softmax
```

---

## Core components

### Tensor

The `Tensor` class stores:

- raw NumPy data
- gradient values
- parent tensors
- backward functions
- graph traversal logic

Calling `backward()` builds a topological ordering of the computation graph and applies the chain rule in reverse.

### Layers

Implemented neural network abstractions:

- `Layer`
- `Dense`
- `Sequential`

### Optimizer

Implemented optimizer:

- `SGD`

The optimizer updates parameters using:

```text
parameter = parameter - learning_rate * gradient
```

---

## Feature comparison

| Feature | This Engine | PyTorch |
|---|---:|---:|
| Tensor object | ✅ | ✅ |
| Reverse-mode autograd | ✅ | ✅ |
| Dynamic computation graph | ✅ | ✅ |
| Broadcasting gradients | ✅ | ✅ |
| Matrix multiplication | ✅ | ✅ |
| ReLU | ✅ | ✅ |
| Softmax | ✅ | ✅ |
| Cross-entropy loss | ✅ | ✅ |
| Dense layers | ✅ | ✅ |
| SGD optimizer | ✅ | ✅ |
| CNN layers | ❌ | ✅ |
| GPU acceleration | ❌ | ✅ |

---

## Resume bullet

> Built a NumPy-based automatic differentiation engine with 15+ tensor operations, broadcasting-aware backpropagation, dense neural network layers, SGD optimization, and end-to-end MNIST/XOR training demos, enabling framework-level understanding of reverse-mode autodiff and neural network training internals.

---

## Why I built this

I wanted to move beyond using deep learning libraries as black boxes and understand the mechanics behind automatic differentiation, gradient flow, neural network layers, and optimization. Building this project helped me understand how modern deep learning frameworks connect mathematical operations into a computation graph and use backpropagation to train models.

---

## Future work

- Add PyTorch gradient parity tests for every operation
- Add Adam optimizer
- Add Conv2D and MaxPool layers
- Add model save/load utilities
- Add type hints and docstrings across the full engine
- Add more examples beyond MNIST and XOR

