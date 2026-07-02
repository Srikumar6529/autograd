import numpy as np
from tensor.tensor import Tensor
from tensor.layers import Sequential, Dense
from tensor.optim import SGD, evaluate_accuracy
from scripts.data_loader import load_mnist, get_mini_batches

# 1. LOAD THE ENTIRE MNIST ENVIRONMENT
x_train, y_train, x_test, y_test = load_mnist()
print("=" * 60)
print(f"MNIST DATASET CONFIGURED")
print(f"-> Full Training Samples: {x_train.shape}")
print(f"-> Full Validation Samples: {x_test.shape}")
print("=" * 60)

# 2. DESIGN THE NEURAL NETWORK
# 784 inputs -> 64 Hidden Units (He initialized) -> 10 Class Outputs
# We pass an explicit 'relu' string flag to use your internal tensor activation method
scale = np.sqrt(2.0 / 784)
model = Sequential([
    Dense(n_in=784, n_out=64, activation='relu'),
    Dense(n_in=64,  n_out=16, activation="relu"),
    Dense(n_in=16,n_out=10,activation=None)
])

# 3. SET TRAINING HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10

optimizer = SGD(model.layers, lr=LEARNING_RATE)

print(f"\nHyperparameters: Batch Size={BATCH_SIZE} | Learning Rate={LEARNING_RATE} | Epochs={EPOCHS}")
print("\n--- Starting Formal Training Loop ---")

# 4. EXECUTE COUPLING STEPS OVER THE TRAINING DATA
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    batch_count = 0
    
    # Iterate dynamically through mini-batches
    for batch_x, batch_y in get_mini_batches(x_train, y_train, BATCH_SIZE):
        # A. Clear previous gradient memory canvas completely
        optimizer.zero_grad()
        
        # B. Load training arrays with active gradient tracking enabled
        inputs = Tensor(batch_x, requires_grad=False) 
        targets = Tensor(batch_y, requires_grad=False)
        
        # C. Forward Pass Execution
        logits = model(inputs)
        probabilities = logits.softmax()
        loss = probabilities.categorical_crossentropy(targets)
        
        # D. Backward Pass Chain Rule
        loss.backward()
        
        # E. Weight updates across parameters
        optimizer.step()
        
        # Track total rolling training loss metric
        epoch_loss += loss.data
        batch_count += 1
        
    # Calculate exact average loss over the training dataset for this epoch
    avg_train_loss = epoch_loss / batch_count
    
    # F. Run a quick intermediate validation pass on the training data 
    # to measure how well the network is learning the specific set
    train_acc = evaluate_accuracy(model, Tensor(x_train), Tensor(y_train))
    
    print(f"Epoch {epoch + 1:02d}/{EPOCHS:02d} -> Train Loss: {avg_train_loss:.4f} | Training Set Accuracy: {train_acc:.2f}%")

print("\n--- Training Complete! Starting Ultimate Test Evaluation ---")

# 5. FINAL TEST DATASET ACCURACY SCORE
# We evaluate across the complete testing data matrix to judge actual generalization
test_inputs = Tensor(x_test, requires_grad=False) # explicitly false to save computer memory
test_targets = Tensor(y_test, requires_grad=False)

final_test_accuracy = evaluate_accuracy(model, test_inputs, test_targets)

# Final calculation pass to determine final testing cross-entropy loss value
test_logits = model(test_inputs)
test_probs = test_logits.softmax()
final_test_loss = test_probs.categorical_crossentropy(test_targets).data

print("=" * 60)
print(f"FINAL BENCHMARK EVALUATION RESULTS")
print(f"-> Final Evaluation Test Loss: {final_test_loss:.4f}")
print(f"-> Final Generalization Test Accuracy Score: {final_test_accuracy:.2f}%")
print("=" * 60)

def visualize_and_predict(model, test_images, test_labels, sample_index=0):
    """
    Renders an MNIST handwritten digit as ASCII art in the terminal 
    and displays the model's live prediction against the true label.
    """
    # 1. Isolate the target test image and label row
    raw_img = test_images[sample_index]
    true_label = np.argmax(test_labels[sample_index])
    
    # 2. Re-shape the flat 784 array back into a 28x28 grid for visualization
    grid_28x28 = raw_img.reshape(28, 28)
    
    print("\n" + "="*30)
    print(f" RENDERED TARGET IMAGE (Index: {sample_index})")
    print("="*30)
    
    # 3. Print the pixel values as visual ASCII characters based on intensity
    for row in grid_28x28:
        line = ""
        for pixel in row:
            if pixel > 0.7:
                line += "██"  # Solid dark fill
            elif pixel > 0.3:
                line += "▒▒"  # Medium gray shading
            elif pixel > 0.05:
                line += ".." # Light trace dot
            else:
                line += "  "  # Empty background space
        print(line)
        
    print("="*30)
    
    # 4. Feed the raw image array through your model instance for inference
    img_tensor = Tensor(raw_img.reshape(1, -1), requires_grad=False)
    output_probabilities = model(img_tensor).softmax().data[0]
    predicted_class = np.argmax(output_probabilities)
    confidence = output_probabilities[predicted_class] * 100.0
    
    print(f"-> True Dataset Label:   {true_label}")
    print(f"-> AI Engine Prediction: {predicted_class} ({confidence:.2f}% Confidence)")
    print("="*30 + "\n")

# Run the visualizer on a couple of test images to see it think in real time!
visualize_and_predict(model, x_test, y_test, sample_index=0)
visualize_and_predict(model, x_test, y_test, sample_index=1)
