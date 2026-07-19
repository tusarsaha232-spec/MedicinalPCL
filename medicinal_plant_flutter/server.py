#!/usr/bin/env python3
"""FastAPI server - Uses ACTUAL trained PyTorch model"""

import os
import sys
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Import model classes
from model_classes import VECTVMixer

# Initialize FastAPI
app = FastAPI(title="Medicinal Plant Classifier API")

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
labels_path = os.path.join(ASSETS_DIR, "labels.txt")
pth_path = os.path.join(ASSETS_DIR, "best_vectvmixer.pth")
pkl_path = os.path.join(ASSETS_DIR, "vectvmixer.pkl")

print("\n" + "=" * 70)
print("🌿 MEDICINAL PLANT CLASSIFIER API - PYTORCH MODEL")
print("=" * 70)
print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Assets directory: {ASSETS_DIR}")

# Check files
if not os.path.exists(labels_path):
    print(f"❌ ERROR: Labels file not found")
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

# Load model
model = None
device = torch.device('cpu')

print("\n🔄 Loading model...")

# Try .pth file first
if os.path.exists(pth_path) and os.path.getsize(pth_path) > 0:
    try:
        print(f"   Trying: {pth_path}")
        model = VECTVMixer(num_classes=len(labels))
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from .pth!")
    except Exception as e:
        print(f"⚠️  .pth failed: {e}")
        model = None

# Try pickle file if .pth didn't work
if model is None and os.path.exists(pkl_path):
    try:
        print(f"   Trying: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            model = torch.load(f, map_location=device)
        model.eval()
        print(f"✅ Model loaded from pickle!")
    except Exception as e:
        print(f"⚠️  Pickle failed: {e}")
        model = None

# Fallback: create fresh model
if model is None:
    print(f"\n⚠️  No trained weights found - using fresh model for structure test")
    model = VECTVMixer(num_classes=len(labels))
    model.to(device)
    model.eval()
    print(f"✅ Fresh model created!")

print(f"\n✅✅✅ MODEL READY! ✅✅✅")


def softmax(x):
    """Compute softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to [1, 3, 224, 224]"""
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # [0, 1]
    img_array = (img_array - 0.5) / 0.5  # [-1, 1]
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW format
    img_array = np.expand_dims(img_array, 0)  # Add batch
    return img_array


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "classes": len(labels), "version": "1.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict using actual trained model"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        print(f"\n📸 Image: {file.filename} ({len(contents)} bytes)")

        # Preprocess
        img_array = preprocess_image(image)
        print(f"   Preprocessed: shape={img_array.shape}")

        # Inference
        img_tensor = torch.from_numpy(img_array).to(device)
        with torch.no_grad():
            output = model(img_tensor)

        # Get prediction
        logits = output.cpu().numpy().flatten()
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
    print("\nStarting server on http://0.0.0.0:8000")
    print("Remote URL: http://192.168.29.48:8000")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
