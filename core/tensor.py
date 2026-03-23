import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            new_idx = []
            for i in idx:
                if isinstance(i, Tensor):
                    new_idx.append(i.data.astype(np.int32))
                else:
                    new_idx.append(i)
            idx = tuple(new_idx)

        elif isinstance(idx, Tensor):
            idx = idx.data.astype(np.int32)

        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in getattr(v, "_prev", []):
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            if hasattr(node, "_backward"):
                node._backward()


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, True, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                # FIX broadcasting
                grad = out.grad
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(len(other.data.shape)):
                    if other.data.shape[i] == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                self.grad = grad_self if self.grad is None else self.grad + grad_self

            if other.requires_grad:
                grad_other = -out.grad
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        out._prev = [self, other]
        return out
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, True, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = (1 / other.data) * out.grad
                self.grad = grad_self if self.grad is None else self.grad + grad_self

            if other.requires_grad:
                grad_other = (-self.data / (other.data ** 2)) * out.grad
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        out._prev = [self, other]
        return out
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, True, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(np.float32) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out


    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.data * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out


    def log(self):
        eps = 1e-8
        out = Tensor(np.log(self.data + eps), requires_grad=self.requires_grad)
        #out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = (1 / self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad

                # reshape grad to broadcast properly
                if axis is not None and not keepdims:
                    shape = list(self.data.shape)
                    if isinstance(axis, int):
                        shape[axis] = 1
                    else:
                        for ax in axis:
                            shape[ax] = 1
                    grad = grad.reshape(shape)

                grad = np.broadcast_to(grad, self.data.shape)

                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out
    def max(self, axis=None, keepdims=False):
        out = Tensor(
            np.max(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                mask = (self.data == out.data)

                if axis is not None and not keepdims:
                    shape = list(self.data.shape)
                    if isinstance(axis, int):
                        shape[axis] = 1
                    else:
                        for ax in axis:
                            shape[ax] = 1
                    grad = grad.reshape(shape)

                grad = np.broadcast_to(grad, self.data.shape)
                grad = grad * mask

                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * np.ones_like(self.data) / self.data.size
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = [self]
        return out