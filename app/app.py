import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import your custom autograd framework structures
from tensor.tensor import Tensor
from tensor.layers import Sequential, Dense

app = FastAPI(title="Custom Autograd Engine Inference API")

# Enable CORS so your frontend can communicate with it smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize the exact network structure from your training script
model = Sequential([
    Dense(n_in=784, n_out=64, activation='relu'),
    Dense(n_in=64,  n_out=16, activation="relu"),
    Dense(n_in=16,  n_out=10, activation=None)
])

# 2. Hot-load your pre-trained weights from the root folder
weights_path = "mnist_weights.npz"
if os.path.exists(weights_path):
    try:
        weights = np.load(weights_path)
        model.layers[0].weights.data = weights['w1']
        model.layers[0].bias.data = weights['b1']
        model.layers[1].weights.data = weights['w2']
        model.layers[1].bias.data = weights['b2']
        model.layers[2].weights.data = weights['w3']
        model.layers[2].bias.data = weights['b3']
        print("🎉 Successfully loaded pre-trained autograd weights into the model graph!")
    except Exception as e:
        print(f"⚠️ Warning: Found weights file but failed to parse parameters ({e}).")
else:
    print("⚠️ Warning: 'mnist_weights.npz' not found. App running with random initialization. Please run training script first!")

class PredictRequest(BaseModel):
    pixels: list[float]  # Expecting a flat array of 784 normalized floats (0.0 to 1.0)

@app.post("/predict")
def predict(payload: PredictRequest):
    if len(payload.pixels) != 784:
        raise HTTPException(status_code=400, detail="Payload must contain exactly 784 pixels.")
    
    # Format into a batch of 1 row vector
    input_array = np.array(payload.pixels, dtype=np.float32).reshape(1, -1)
    input_tensor = Tensor(input_array, requires_grad=False)
    
    # Execute forward pass through your custom framework code
    logits = model(input_tensor)
    probabilities = logits.softmax().data[0]
    
    predicted_digit = int(np.argmax(probabilities))
    
    # Map out the confidence scores for the UI display
    confidences = {str(i): float(probabilities[i]) for i in range(10)}
    
    return {
        "prediction": predicted_digit,
        "confidence": float(probabilities[predicted_digit]),
        "all_confidences": confidences
    }

# 3. Mount the static directory so files are accessible at /static/*
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 4. Catch-all route to serve your index.html at the root URL "/"
@app.get("/")
def read_root():
    return FileResponse(os.path.join("app/static", "index.html"))

@app.get("/health")
def health():
    return {"status": "healthy"}