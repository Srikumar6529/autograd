import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape})\n{self.data}"
    
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
        return Tensor(b_self + b_other, requires_grad=self.requires_grad or other.requires_grad)

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        return Tensor(b_self - b_other, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        return Tensor(b_self * b_other, requires_grad=self.requires_grad or other.requires_grad)

    def __truediv__(self, other): # Note: Changed __div__ to __truediv__ for modern Python 3 compatibility
        if not isinstance(other, Tensor): other = Tensor(other)
        target_shape = self._get_broadcast_shape(other.shape)
        
        b_self = self._broadcast_to(target_shape)
        b_other = other._broadcast_to(target_shape)
        return Tensor(b_self / b_other, requires_grad=self.requires_grad or other.requires_grad)

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
        res_data = np.matmul(self.data, other.data)
        
        # 4. TRACK GRADIENTS
        out_requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(res_data, requires_grad=out_requires_grad)
    
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
        
        # Keep track of gradients across the transformation
        return Tensor(transposed_data, requires_grad=self.requires_grad)

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
        
        # Pass forward the gradient tracking configuration
        return Tensor(data=relu_data, requires_grad=self.requires_grad)



    
