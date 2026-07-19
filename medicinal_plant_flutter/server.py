#!/usr/bin/env python3
"""FastAPI server for medicinal plant inference - LOCAL VERSION"""

import os
import numpy as np
from PIL import Image
from io import BytesIO

# Import TensorFlow
import tensorflow as tf

# FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Medicinal Plant Classifier API")

# Get paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

tflite_path = os.path.join(ASSETS_DIR, "vectvmixer_float32.tflite")
labels_path = os.path.join(ASSETS_DIR, "labels.txt")

print("\n" + "=" * 70)
print("🌿 MEDICINAL PLANT CLASSIFIER API - LOCAL SERVER")
print("=" * 70)
print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Assets directory: {ASSETS_DIR}")
print(f"TFLite model: {tflite_path}")
print(f"Labels file: {labels_path}")

# Check if files exist
if not os.path.exists(tflite_path):
    print(f"❌ ERROR: Model file not found at {tflite_path}")
    exit(1)

if not os.path.exists(labels_path):
    print(f"❌ ERROR: Labels file not found at {labels_path}")
    exit(1)

print("\n✅ Files found! Loading model...")

# Load model
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
try:
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ Labels loaded: {len(labels)} classes")
    print(f"   Classes: {labels}")
except Exception as e:
    print(f"❌ Failed to load labels: {e}")
    exit(1)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to [1, 224, 224, 3] and normalize to [-1, 1]"""
    img = image.convert('RGB')
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # [0, 1]
    img_array = (img_array - 0.5) / 0.5  # [-1, 1]
    img_array = np.expand_dims(img_array, 0)  # Add batch [1, 224, 224, 3]

    return img_array


def softmax(x):
    """Compute softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "loaded",
        "classes": len(labels),
        "version": "1.0"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant class from image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        print(f"\n📸 Processing image: {file.filename}")

        # Preprocess
        img_array = preprocess_image(image)
        print(f"✅ Preprocessed: shape={img_array.shape}, range=[{img_array.min():.2f}, {img_array.max():.2f}]")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get predictions
        output_flat = output_data.flatten()
        pred_idx = np.argmax(output_flat)
        probs = softmax(output_flat)

        print(f"✅ Prediction: {labels[pred_idx]} (confidence: {probs[pred_idx]:.2%})")

        # Return results
        return {
            "success": True,
            "predicted_class": labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_predictions": {
                labels[i]: float(probs[i]) for i in range(len(labels))
            }
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("✅ SERVER READY!")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Predict from image")
    print("\nStarting server on http://0.0.0.0:8000")
    print("API docs: http://localhost:8000/docs")
    print("Remote URL: http://192.168.29.48:8000")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
