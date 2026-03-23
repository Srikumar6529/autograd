import numpy as np
from core.tensor import Tensor
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

