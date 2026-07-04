# Autograd

> A lightweight deep learning framework built entirely from scratch using **NumPy**.

Autograd is a custom deep learning framework that implements **reverse-mode automatic differentiation** for tensors and uses it to train neural networks without relying on machine learning frameworks such as PyTorch or TensorFlow.

The goal of this project is to understand and implement the fundamental building blocks behind modern deep learning libraries, including computational graphs, automatic differentiation, neural network layers, optimization algorithms, and end-to-end model training.

---

## Features

* Reverse-mode automatic differentiation
* Dynamic computational graph construction
* Broadcasting-aware gradient propagation
* Matrix multiplication and tensor operations
* Tensor reshaping and reduction operations
* ReLU activation
* Softmax activation
* Cross Entropy Loss
* Dense (Fully Connected) layers
* Sequential model API
* Stochastic Gradient Descent (SGD) optimizer
* Gradient validation against PyTorch
* End-to-end MNIST handwritten digit classification example

---

# Project Structure

```text
autograd/
│
├── tensor/
│   ├── tensor.py          # Tensor implementation & autograd engine
│   ├── layers.py          # Neural network layers
│   ├── loss.py            # Loss functions
│   ├── optim.py           # SGD optimizer
│   └── ...
│
├── tests/                 # Gradient validation tests
│
├── scripts/
│   └── mnist.py           # MNIST training example
│
├── demo.py                # Quick project demonstration
│
└── README.md
```

---

# Architecture

```text
Input Tensor
      │
      ▼
Tensor Operations
      │
      ▼
Dynamic Computational Graph
      │
      ▼
Backward Pass
      │
      ▼
Automatic Gradient Computation
      │
      ▼
SGD Optimizer
      │
      ▼
Updated Parameters
```

---

# Supported Components

| Category                               | Status |
| -------------------------------------- | :----: |
| Tensor Operations                      |    ✅   |
| Broadcasting                           |    ✅   |
| Matrix Multiplication                  |    ✅   |
| Reverse-Mode Automatic Differentiation |    ✅   |
| ReLU                                   |    ✅   |
| Softmax                                |    ✅   |
| Cross Entropy Loss                     |    ✅   |
| Dense Layers                           |    ✅   |
| Sequential Model                       |    ✅   |
| SGD Optimizer                          |    ✅   |
| MNIST Training                         |    ✅   |

---

# Quick Start

Clone the repository

```bash
git clone https://github.com/Srikumar6529/autograd.git
cd autograd
```

Install dependencies

```bash
pip install numpy
```

Run the demo

```bash
python demo.py
```

---

# Demo

The demo trains a fully connected neural network on a subset of the **MNIST handwritten digit dataset** using only this custom automatic differentiation engine.

The demonstration includes:

* Forward propagation
* Backward propagation
* Automatic gradient computation
* SGD optimization
* Training loop
* Accuracy evaluation
* Sample predictions
* Gradient statistics

Example output:

```text
========================================================================
AUTOGRAD ENGINE DEMO: HANDWRITTEN DIGIT CLASSIFICATION
Training a neural network using only NumPy + this custom autograd engine
========================================================================

Dataset: MNIST

Train shape: (2000, 784)
Test shape : (500, 784)

Model
784 → 64 → ReLU → 10 → Softmax

Epoch 01/5 | loss=1.7750 | train_acc=78.20% | test_acc=66.20%
Epoch 02/5 | loss=0.9951 | train_acc=85.60% | test_acc=72.40%
Epoch 03/5 | loss=0.6916 | train_acc=88.00% | test_acc=78.20%
Epoch 04/5 | loss=0.5525 | train_acc=89.40% | test_acc=80.80%
Epoch 05/5 | loss=0.4767 | train_acc=90.20% | test_acc=82.60%

Sample Predictions

7 ✓
2 ✓
1 ✓
0 ✓
4 ✓
1 ✓
4 ✓
9 ✓

Gradient Statistics

First Layer Weight Shape : (784, 64)
Gradient Shape           : (784, 64)

Gradient Mean            : -0.00031
Gradient Std             : 0.01428
Gradient Max             : 0.13754
Gradient Min             : -0.12218
Gradient L2 Norm         : 3.48
```

> **Note:** The exact numerical values may vary between runs due to random weight initialization.

---

# Demo GIF

<p align="center">
  <img src="assets/demo.gif" width="850">
</p>

---

# Testing

Gradient correctness is validated by comparing gradients produced by this engine against **PyTorch**.

Run all tests:

```bash
python -m unittest discover tests
```

The test suite verifies:

* Tensor operations
* Broadcasting behavior
* Matrix multiplication
* Automatic differentiation
* Gradient correctness

---

# Motivation

Deep learning frameworks provide powerful abstractions, but they also hide many of the algorithms that make them work.

This project rebuilds those components from first principles to develop a deeper understanding of:

* Computational graphs
* Reverse-mode automatic differentiation
* Backpropagation
* Neural network training
* Gradient-based optimization

Rather than treating frameworks like PyTorch as black boxes, this project demonstrates how they work internally.

---

# Future Improvements

* [ ] Adam Optimizer
* [ ] Batch Normalization
* [ ] Dropout
* [ ] Convolution Layers (Conv2D)
* [ ] Max Pooling
* [ ] CUDA Backend
* [ ] Mixed Precision Training
* [ ] Transformer Layers

---

# License

This project is released under the MIT License.
