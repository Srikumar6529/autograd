import os
import urllib.request
import numpy as np

def load_mnist():
    """
    Downloads and prepares the MNIST dataset using pure NumPy storage.
    Flattens 28x28 images to 784 features and one-hot encodes labels.
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    file_path = "mnist.npz"
    
    if not os.path.exists(file_path):
        print("Downloading MNIST dataset (approx 11MB)... Please wait...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
        
    with np.load(file_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        
    # 1. FLATTEN AND NORMALIZE: 28x28 matrices become 784 element input features scaled 0-1
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
    
    # 2. ONE-HOT ENCODING: Convert integer targets (0-9) to 10-dimensional arrays
    def one_hot(labels, num_classes=10):
        out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        out[np.arange(labels.shape[0]), labels] = 1.0
        return out
        
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    
    return x_train, y_train, x_test, y_test

def get_mini_batches(X, Y, batch_size):
    """Generates randomized mini-batches out of the entire dataset."""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]
