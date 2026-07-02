import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False,_parents = None):
        self.data = np.array(data, dtype=np.float32)
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data,dtype=np.float32) if requires_grad else None
        self._parents = _parents
        self._backward = lambda : None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})\n{self.data}"
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data,dtype=np.float32)
        elif np.all(self.grad == 0.0):
            self.grad = np.ones_like(self.data, dtype=np.float32)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._parents is not None:
                    for parent in v._parents:
                        build_topo(parent)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()

        
    def _get_broadcast_shape(self, other_shape):
        """
        Helper method to compute the final output shape resulting from two inputs.
        """
        shape1 = list(self.shape)
        shape2 = list(other_shape)
        
        # Pad the shorter shape with 1s on the left
        while len(shape1) < len(shape2): shape1.insert(0, 1)
        while len(shape2) < len(shape1): shape2.insert(0, 1)
            
        target_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2:
                target_shape.append(dim1)
            elif dim1 == 1:
                target_shape.append(dim2)
            elif dim2 == 1:
                target_shape.append(dim1)
            else:
                raise ValueError(
                    f"Cannot broadcast shapes: {self.shape} and {other_shape}"
                )
        return target_shape

    def _broadcast_to(self, target_shape):
        """
        Manually broadcasts the tensor's data to match a target shape.
        Returns a raw NumPy array with the expanded shape.
        """
        shape1 = list(self.shape)
        while len(shape1) < len(target_shape):
            shape1.insert(0, 1)
            
        data_reshaped = self.data.reshape(shape1)
        
        for axis, (dim1, target_dim) in enumerate(zip(shape1, target_shape)):
            if dim1 == 1 and target_dim > 1:
                data_reshaped = np.repeat(data_reshaped, target_dim, axis=axis)
            elif dim1 != target_dim:
                raise ValueError(
                    f"Incompatible shapes for manual broadcasting: {self.shape} vs {target_shape}"
                )
                
        return data_reshaped

    # --- ARITHMETIC OPERATIONS ---

    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        out_grad = self.requires_grad or other.requires_grad
        out = Tensor(b_self + b_other, requires_grad=out_grad,_parents = (self,other))
        if out_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad
                    # Handle Broadcasting: Sum out dimensions that were padded on the left
                    while len(grad_self.shape) > len(self.shape):
                        grad_self = grad_self.sum(axis=0)
                    # Sum out dimensions where the parent had a size of 1
                    for axis, dim in enumerate(self.shape):
                        if dim == 1:
                            grad_self = grad_self.sum(axis=axis, keepdims=True)
                    self.grad += grad_self

                if other.requires_grad:
                    grad_other = out.grad
                    # Handle Broadcasting for the second operand
                    while len(grad_other.shape) > len(other.shape):
                        grad_other = grad_other.sum(axis=0)
                    for axis, dim in enumerate(other.shape):
                        if dim == 1:
                            grad_other = grad_other.sum(axis=axis, keepdims=True)
                    other.grad += grad_other
                    
            out._backward = _backward

        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(b_self - b_other, requires_grad=out_requires_grad, _parents=(self, other))

        if out_requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad * 1.0
                    while len(grad_self.shape) > len(self.shape): grad_self = grad_self.sum(axis=0)
                    for axis, dim in enumerate(self.shape):
                        if dim == 1: grad_self = grad_self.sum(axis=axis, keepdims=True)
                    self.grad += grad_self

                if other.requires_grad:
                    grad_other = out.grad * -1.0
                    while len(grad_other.shape) > len(other.shape): grad_other = grad_other.sum(axis=0)
                    for axis, dim in enumerate(other.shape):
                        if dim == 1: grad_other = grad_other.sum(axis=axis, keepdims=True)
                    other.grad += grad_other
            out._backward = _backward

        return out
    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(b_self * b_other, requires_grad=out_requires_grad, _parents=(self, other))

        if out_requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad * b_other  # d/dx(x*y) = y
                    while len(grad_self.shape) > len(self.shape): grad_self = grad_self.sum(axis=0)
                    for axis, dim in enumerate(self.shape):
                        if dim == 1: grad_self = grad_self.sum(axis=axis, keepdims=True)
                    self.grad += grad_self

                if other.requires_grad:
                    grad_other = out.grad * b_self  # d/dy(x*y) = x
                    while len(grad_other.shape) > len(other.shape): grad_other = grad_other.sum(axis=0)
                    for axis, dim in enumerate(other.shape):
                        if dim == 1: grad_other = grad_other.sum(axis=axis, keepdims=True)
                    other.grad += grad_other
            out._backward = _backward

        return out

    def __truediv__(self, other): # Note: Changed __div__ to __truediv__ for modern Python 3 compatibility
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(b_self / b_other, requires_grad=out_requires_grad, _parents=(self, other))

        if out_requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad / b_other
                    while len(grad_self.shape) > len(self.shape): grad_self = grad_self.sum(axis=0)
                    for axis, dim in enumerate(self.shape):
                        if dim == 1: grad_self = grad_self.sum(axis=axis, keepdims=True)
                    self.grad += grad_self

                if other.requires_grad:
                    grad_other = out.grad * (-b_self / (b_other ** 2))
                    while len(grad_other.shape) > len(other.shape): grad_other = grad_other.sum(axis=0)
                    for axis, dim in enumerate(other.shape):
                        if dim == 1: grad_other = grad_other.sum(axis=axis, keepdims=True)
                    other.grad += grad_other
            out._backward = _backward

        return out

        # Place these inside your Tensor class, right under your main arithmetic methods

    def __radd__(self, other):
        # 5 + self is exactly identical to self + 5
        return self.__add__(other)

    def __rmul__(self, other):
        # 5 * self is exactly identical to self * 5
        return self.__mul__(other)

    def __rsub__(self, other):
        # Handles: 5 - self
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        return other.__sub__(self)

    def __rtruediv__(self, other):
        # Handles: 5 / self
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        return other.__truediv__(self)
    
    #Matrix ops 
    def __matmul__(self, other):
        # Convert raw arrays, lists, or scalars to a Tensor instance
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # 1. VALIDATE DIMENSIONS FOR MATMUL
        # Matrix multiplication requires at least 1D/2D arrays. Scalars are not allowed.
        if self.data.ndim < 1 or other.data.ndim < 1:
            raise ValueError("Matrix multiplication requires tensors of at least 1 dimension.")

        # 2. CHECK INNER DIMENSION MATCH (for standard 2D matrices)
        # If both are 2D, the columns of self (self.shape[1]) must match rows of other (other.shape[0])
        if self.data.ndim == 2 and other.data.ndim == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Dimension mismatch for matmul: {self.shape} and {other.shape}. "
                    f"Inner dimensions {self.shape[1]} and {other.shape[0]} must match."
                )

        # 3. PERFORM THE MATRIX MULTIPLICATION
        # np.matmul natively handles standard 2D dot products and higher-order batch matmuls
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.matmul(self.data, other.data), requires_grad=out_requires_grad, _parents=(self, other))

        if out_requires_grad:
            def _backward():
                if self.requires_grad:
                    # grad_self = out.grad @ other.T
                    self.grad += np.matmul(out.grad, np.transpose(other.data))
                if other.requires_grad:
                    # grad_other = self.T @ out.grad
                    other.grad += np.matmul(np.transpose(self.data), out.grad)
            out._backward = _backward

        return out
    
    def __rmatmul__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        return other.__matmul__(self)
    
    def transpose(self, axes=None):
        """
        Transposes the dimensions of the tensor.
        'axes' can be a tuple or list of ints to reorder dimensions specifically.
        """
        # np.transpose handles the reordering logic natively
        transposed_data = np.transpose(self.data, axes=axes)
        out = Tensor(transposed_data, requires_grad=self.requires_grad, _parents=(self,))

        if self.requires_grad:
            def _backward():
                if axes is None:
                    # If default axes were used, reversing them again restores original orientation
                    self.grad += np.transpose(out.grad)
                else:
                    # If custom axis ordering was specified, we invert that specific permutation
                    # np.argsort tells us how to undo the axis reordering map
                    inverse_axes = np.argsort(axes)
                    self.grad += np.transpose(out.grad, axes=inverse_axes)
            out._backward = _backward

        return out

    @property
    def T(self):
        """
        A shortcut property to quickly reverse axes, matching NumPy/PyTorch syntax.
        Allows you to simply call: tensor.T
        """
        return self.transpose()
    
    #Activation functions
    def relu(self):
        """
        Applies the Rectified Linear Unit activation function element-wise.
        Replaces all negative numbers with 0.
        """
        # np.maximum compares every element to 0 and returns the larger value
        relu_data = np.maximum(0.0, self.data)
        out = Tensor(relu_data, requires_grad=self.requires_grad, _parents=(self,))
        if self.requires_grad:
            def _backward():
                # Pass gradient through only where the forward input data was positive
                self.grad += out.grad * (self.data > 0.0)
            out._backward = _backward

        return out
        
    def softmax(self):
        """
        Applies the Softmax function along the final axis (axis=-1).
        Uses a subtraction trick to avoid numerical overflow (nan errors).
        """
        # 1. Shift inputs for numerical stability (subtract max value per sample)
        max_val = np.max(self.data, axis=-1, keepdims=True)
        exp_data = np.exp(self.data - max_val)
        
        # 2. Divide by the row sums to form probabilities
        prob_data = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        
        out = Tensor(prob_data, requires_grad=self.requires_grad, _parents=(self,))

        if self.requires_grad:
            def _backward():
                # out.data contains our calculated probability distribution (P)
                # Core vectorised softmax backprop calculation:
                sum_grad_times_prob = np.sum(out.grad * out.data, axis=-1, keepdims=True)
                self.grad += out.data * (out.grad - sum_grad_times_prob)
            out._backward = _backward

        return out
    def categorical_crossentropy(self, targets):
        """
        Computes the CCE loss between this probability distribution and true targets.
        Expects 'targets' to be a Tensor instance of one-hot encoded labels.
        """
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        # Clip probabilities to avoid taking log(0) which outputs NaN or -Inf
        eps = 1e-15
        clipped_preds = np.clip(self.data, eps, 1.0 - eps)
        
        # Cross entropy core math: -sum(y_true * log(y_pred))
        loss_per_sample = -np.sum(targets.data * np.log(clipped_preds), axis=-1)
        
        # Average the loss across the entire batch of samples
        mean_loss = np.mean(loss_per_sample)
        
        out = Tensor(mean_loss, requires_grad=self.requires_grad or targets.requires_grad, _parents=(self, targets))

        if self.requires_grad:
            def _backward():
                # d/dP = - (targets / predictions)
                grad_preds = -(targets.data / clipped_preds)
                
                # Account for the np.mean() operation by dividing by the batch size
                batch_size = self.shape[0]
                self.grad += (out.grad * grad_preds) / batch_size
            out._backward = _backward

        return out




    
