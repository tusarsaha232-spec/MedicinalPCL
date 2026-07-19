#!/usr/bin/env python3
"""FastAPI server - Uses TFLite model (working infrastructure)"""

import os
import sys
import numpy as np
from PIL import Image
from io import BytesIO

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Medicinal Plant Classifier API")

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
labels_path = os.path.join(ASSETS_DIR, "labels.txt")
tflite_path = os.path.join(ASSETS_DIR, "vectvmixer_float32.tflite")

print("\n" + "=" * 70)
print("🌿 MEDICINAL PLANT CLASSIFIER API - TFLITE")
print("=" * 70)
print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Assets directory: {ASSETS_DIR}")

# Check files
if not os.path.exists(labels_path):
    print(f"❌ ERROR: Labels file not found")
    sys.exit(1)

if not os.path.exists(tflite_path):
    print(f"❌ ERROR: TFLite model not found")
    sys.exit(1)

print("✅ Files found! Loading...")

# Load labels
try:
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ Labels: {len(labels)} classes - {labels}")
except Exception as e:
    print(f"❌ Failed to load labels: {e}")
    sys.exit(1)

# Load TFLite model
print("\n🔄 Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ TFLite model loaded!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Failed to load TFLite: {e}")
    sys.exit(1)

print(f"\n✅✅✅ MODEL READY! ✅✅✅")


def softmax(x):
    """Compute softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image"""
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_array = np.expand_dims(img_array, 0)
    return img_array.astype(np.float32)


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "classes": len(labels), "version": "1.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict using TFLite model"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        print(f"\n📸 Image: {file.filename} ({len(contents)} bytes)")

        # Preprocess
        img_array = preprocess_image(image)
        print(f"   Preprocessed: shape={img_array.shape}")

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        logits = output.flatten()
        pred_idx = np.argmax(logits)
        probs = softmax(logits)

        print(f"✅ Prediction: {labels[pred_idx]} ({probs[pred_idx]:.2%})")

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
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("✅ SERVER READY!")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Predict from image")
    print("\nStarting server on http://0.0.0.0:3000")
    print("Remote URL: http://192.168.29.48:3000")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=3000)
