import unittest
import numpy as np
import torch

# Import your custom tensor and mockup framework structures
# Adjust these imports if your file structure is different
from tensor.tensor import Tensor
from tensor.layers import Sequential, Dense

class TestCustomTensorEngine(unittest.TestCase):
    
    def assert_arrays_close(self, custom_arr, torch_tensor, rtol=1e-4, atol=1e-4):
        """Helper function to compare custom numpy data with PyTorch tensors."""
        torch_arr = torch_tensor.detach().cpu().numpy()
        np.testing.assert_allclose(custom_arr, torch_arr, rtol=rtol, atol=atol)

    def test_initialization_and_properties(self):
        """1. Verify basic tensor instantiation, datatypes, and gradients initialization."""
        data = [[1, 2], [3, 4]]
        t = Tensor(data, requires_grad=True)
        
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.dtype, np.float32)
        self.assertTrue(t.requires_grad)
        self.assertIsNotNone(t.grad)
        np.testing.assert_equal(t.grad, np.zeros((2, 2), dtype=np.float32))

    def test_broadcasting_rules_and_addition(self):
        """2. Validate standard addition, automatic broadcasting alignment, and gradient tracking."""
        x_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # (2, 3)
        b_raw = [[10.0, 20.0, 30.0]]                # (1, 3) -> Broadcasted downward
        
        # Custom
        x = Tensor(x_raw, requires_grad=True)
        b = Tensor(b_raw, requires_grad=True)
        out = x + b
        out.backward()
        
        # PyTorch Reference
        tx = torch.tensor(x_raw, requires_grad=True, dtype=torch.float32)
        tb = torch.tensor(b_raw, requires_grad=True, dtype=torch.float32)
        tout = tx + tb
        tout.backward(torch.ones_like(tout))
        
        # Assert Forward
        self.assert_arrays_close(out.data, tout)
        # Assert Backward (Vector accumulation test)
        self.assert_arrays_close(x.grad, tx.grad)
        self.assert_arrays_close(b.grad, tb.grad)  # Verifies the broadcasting sum-out loop

    def test_subtraction_and_right_sided_ops(self):
        """3. Check subtraction, non-commutative structures, and scalar right-sided triggers (e.g., 5 - X)."""
        x_raw = [[2.0, 4.0], [6.0, 8.0]]
        
        # Custom (5.0 - X)
        x = Tensor(x_raw, requires_grad=True)
        out = 5.0 - x
        out.backward()
        
        # PyTorch Reference
        tx = torch.tensor(x_raw, requires_grad=True, dtype=torch.float32)
        tout = 5.0 - tx
        tout.backward(torch.ones_like(tout))
        
        self.assert_arrays_close(out.data, tout)
        self.assert_arrays_close(x.grad, tx.grad)

    def test_elementwise_multiplication_and_division(self):
        """4. Test multiplication and division forward passes alongside complex gradient functions."""
        x_raw = [[1.0, 2.0], [3.0, 4.0]]
        y_raw = [[5.0, 6.0], [7.0, 8.0]]
        
        # Custom Mul & Div
        x = Tensor(x_raw, requires_grad=True)
        y = Tensor(y_raw, requires_grad=True)
        out_mul = x * y
        out_div = out_mul / y
        out_div.backward()
        
        # PyTorch Reference
        tx = torch.tensor(x_raw, requires_grad=True, dtype=torch.float32)
        ty = torch.tensor(y_raw, requires_grad=True, dtype=torch.float32)
        tout_mul = tx * ty
        tout_div = tout_mul / ty
        tout_div.backward(torch.ones_like(tout_div))
        
        self.assert_arrays_close(out_div.data, tout_div)
        self.assert_arrays_close(x.grad, tx.grad)
        self.assert_arrays_close(y.grad, ty.grad)

    def test_matrix_multiplication(self):
        """5. Check structural 2D dot products and corresponding transposed gradient calculations."""
        x_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # (2, 3)
        w_raw = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # (3, 2)
        
        # Custom
        x = Tensor(x_raw, requires_grad=True)
        w = Tensor(w_raw, requires_grad=True)
        out = x @ w
        out.backward()
        
        # PyTorch Reference
        tx = torch.tensor(x_raw, requires_grad=True, dtype=torch.float32)
        tw = torch.tensor(w_raw, requires_grad=True, dtype=torch.float32)
        tout = tx @ tw
        tout.backward(torch.ones_like(tout))
        
        self.assert_arrays_close(out.data, tout)
        self.assert_arrays_close(x.grad, tx.grad)
        self.assert_arrays_close(w.grad, tw.grad)

    def test_transpose_and_relu(self):
        """6. Validate axis permutations and the boundary conditions of the ReLU activation gradient filter."""
        x_raw = [[-2.0, 3.0], [0.0, -5.0]]
        
        # Custom
        x = Tensor(x_raw, requires_grad=True)
        out = x.T.relu()
        out.backward()
        
        # PyTorch Reference
        tx = torch.tensor(x_raw, requires_grad=True, dtype=torch.float32)
        tout = tx.t().relu()
        tout.backward(torch.ones_like(tout))
        
        self.assert_arrays_close(out.data, tout)
        self.assert_arrays_close(x.grad, tx.grad)

    def test_softmax_and_categorical_crossentropy(self):
        """7. Verify probability transformations, clipping boundaries, and multi-class loss calculations."""
        logits_raw = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
        targets_raw = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        
        # Custom Softmax + CCE
        logits = Tensor(logits_raw, requires_grad=True)
        targets = Tensor(targets_raw)
        probs = logits.softmax()
        loss = probs.categorical_crossentropy(targets)
        loss.backward()
        
        # PyTorch Reference (PyTorch combines Softmax+CrossEntropy into one module)
        t_logits = torch.tensor(logits_raw, requires_grad=True, dtype=torch.float32)
        t_targets = torch.tensor(targets_raw, dtype=torch.float32)
        
        # Note: PyTorch cross_entropy expects soft probabilities or targets and averages them across dim=0
        t_loss = torch.nn.functional.cross_entropy(t_logits, t_targets)
        t_loss.backward()
        
        self.assert_arrays_close(loss.data, t_loss)
        self.assert_arrays_close(logits.grad, t_logits.grad)

    def test_full_sequential_keras_mockup_pipeline(self):
        """8. Run an entire multi-layer network integration pass to ensure everything communicates without errors."""
        np.random.seed(42) # Lock data seed
        
        X_data = np.random.randn(3, 4).astype(np.float32)
        Y_data = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        
        # 1. Build Model Instance
        model = Sequential([
            Dense(4, 8, activation='relu'),
            Dense(8, 2, activation=None)
        ])
        
        # 2. Fire complete network cycle
        inputs = Tensor(X_data)
        targets = Tensor(Y_data)
        
        logits = model(inputs)
        probs = logits.softmax()
        loss = probs.categorical_crossentropy(targets)
        
        # 3. Assert graph execution compiles without a single runtime exception
        try:
            loss.backward()
            success = True
        except Exception as e:
            success = False
            print(f"Pipeline crashed with error: {e}")
            
        self.assertTrue(success, "The high level forward-backward network architecture failed execution.")
        self.assertEqual(loss.data.ndim, 0) # Loss must be a scalar

if __name__ == '__main__':
    unittest.main()
