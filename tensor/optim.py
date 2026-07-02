import numpy as np
class SGD:
    def __init__(self, layers, lr=0.01):
        """
        Stochastic Gradient Descent Optimizer.
        Accepts a list of layers (like model.layers) and a learning rate (lr).
        """
        self.layers = layers
        self.lr = lr

    def step(self):
        """
        Updates the weights and biases using standard gradient descent:
        W = W - (lr * gradient)
        """
        for layer in self.layers:
            # Check if the layer actually has trainable params (like Dense layers)
            if hasattr(layer, 'params'):
                for param in layer.params:
                    if param.requires_grad and param.grad is not None:
                        # 1. Update the underlying numpy array data
                        param.data -= self.lr * param.grad

    def zero_grad(self):
        """
        Clears out old accumulated gradients before starting a new training step.
        If we don't clear them, new gradients will += add on top of old ones.
        """
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param in layer.params:
                    if param.grad is not None:
                        # Reset the gradient canvas back to zero
                        param.grad.fill(0.0)
def evaluate_accuracy(model, inputs, targets):
    """
    Computes the accuracy percentage of the model on a given dataset.
    """
    # 1. Fire a forward pass (No need to compute gradients during testing)
    logits = model(inputs)
    probabilities = logits.softmax()
    
    # 2. Find the predicted class indices (The column index with the highest probability)
    # np.argmax converts [[0.1, 0.9], [0.8, 0.2]] into [1, 0]
    predictions = np.argmax(probabilities.data, axis=-1)
    
    # 3. Find the true class indices from the one-hot targets
    true_classes = np.argmax(targets.data, axis=-1)
    
    # 4. Compare element-by-element and calculate the mean score
    correct_matches = (predictions == true_classes)
    accuracy_percentage = np.mean(correct_matches) * 100.0
    
    return accuracy_percentage
