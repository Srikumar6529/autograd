import numpy as np
from core.tensor import Tensor

# -----------------------
# Base Module
# -----------------------
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args):
        return self.forward(*args)


# -----------------------
# Linear Layer
# -----------------------
class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


# -----------------------
# ReLU
# -----------------------
class ReLU(Module):
    def forward(self, x):
        return x.relu()   # MUST exist in your Tensor


# -----------------------
# Sequential Model
# -----------------------
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params


# -----------------------
# Cross Entropy Loss
# -----------------------
class CrossEntropyLoss:
    def __call__(self, logits, targets):
        # logits: (B, C)
        B, C = logits.data.shape

        # Step 1: subtract max (stability trick)
        max_logits = np.max(logits.data, axis=1, keepdims=True)
        shifted = logits.data - max_logits

        # Step 2: log-sum-exp trick
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

        # Step 3: log softmax
        log_probs = shifted - log_sum_exp  # (B, C)

        # Step 4: pick correct class
        targets = targets.astype(int)
        selected = log_probs[np.arange(B), targets]

        # Step 5: loss
        loss = -np.mean(selected)

        out = Tensor(loss, requires_grad=True)

        # -----------------------------
        # BACKWARD (CRITICAL)
        # -----------------------------
        def _backward():
            if logits.requires_grad:
                softmax = np.exp(log_probs)  # stable softmax
                one_hot = np.zeros_like(softmax)
                one_hot[np.arange(B), targets] = 1

                grad = (softmax - one_hot) / B

                logits.grad = grad if logits.grad is None else logits.grad + grad

        out._backward = _backward
        out._prev = {logits}

        return out


# -----------------------
# SGD Optimizer
# -----------------------
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0