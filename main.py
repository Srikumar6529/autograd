from tensor.tensor import Tensor
from tensor.layers import *
# Create your model cleanly
model = Sequential([
    Dense(n_in=4, n_out=8, activation='relu'),
    Dense(n_in=8, n_out=2, activation=None)
])

# Create 3 sample points with 4 features each
X_data = Tensor([
    [1.5, -2.0,  0.5, -1.2],
    [-0.5, 3.0, -2.5,  1.1],
    [0.0,  1.0,  2.0, -3.0]
])

# Execute!
outputs = model(X_data)
print(outputs)

