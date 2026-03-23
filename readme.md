# Mini Deep Learning Framework from Scratch (NumPy)

Built a PyTorch-like deep learning framework from scratch with:
- Tensor class
- Autograd engine
- Neural networks
- Stable cross entropy
- SGD optimizer

Trained on MNIST → **94.9% accuracy**

## 📊 Results

| Metric | Value |
|------|------|
| Train Accuracy | 95% |
| Test Accuracy | 94.9% |

Trained using custom autograd engine (no PyTorch)

## ⚙️ Autograd Engine

Each Tensor stores:
- data
- gradient
- parent nodes
- backward function

During backward():
- traverse graph
- apply chain rule
- accumulate gradients

results/loss.png
results/accuracy.png