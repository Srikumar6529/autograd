from tensor.tensor import Tensor
import numpy as np
class Layer:
    def __init__(self):
        self.params = []
    def forward(self,inputs):
        raise NotImplementedError
    def __call__(self, inputs):
        return self.forward(inputs)
    
class Dense(Layer):
    def __init__(self,n_in,n_out,activation):
        super().__init__()
        self.weights = Tensor(np.random.randn(n_in, n_out)*0.01, requires_grad=True)
        self.bias = Tensor(np.zeros((1,n_out)),requires_grad=True)
        self.act_fn = activation
        self.params = [self.weights,self.bias]

    def forward(self, inputs):
        x1 = inputs @ self.weights
        x1 = x1 + self.bias
        if self.act_fn == "relu":
            return x1.relu()
        return x1
    
class Sequential():
    def __init__(self,layers):
        self.layers = layers
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    def __call__(self, inputs):
        return self.forward(inputs)
    
