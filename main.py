from tensor.tensor import Tensor
from tensor.layers import *
# Matrix Shape: (2, 3)
matrix = Tensor([[1., 1., 1.],
                 [1., 1., 1.]], requires_grad=True)

# Vector Shape: (1, 3) -> Broadcasted to (2, 3) in forward pass
bias = Tensor([[10., 20., 30.]], requires_grad=True)

# Forward pass runs smoothly
loss = matrix + bias

# We simulate a backprop gradient coming from a loss function
loss.grad = np.array([[1., 1., 1.],
                      [1., 1., 1.]], dtype=np.float32)

# Trigger your custom backward routine
loss._backward()

print("Matrix Grad Shape:", matrix.grad.shape) # Output: (2, 3)
print("Bias Grad Shape:",   bias.grad.shape)   # Output: (1, 3)
print("Bias Grad Values:",   bias.grad)         # Output: [[2., 2., 2.]]

