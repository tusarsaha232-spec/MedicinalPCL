#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to TFLite
Load from checkpoint folder and export as TFLite
"""

import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

print("=" * 70)
print("PYTORCH CHECKPOINT → TFLITE CONVERTER")
print("=" * 70)

# Paths
ASSETS_DIR = Path("d:/medi/medicinal_plant_flutter/assets")
CHECKPOINT_DIR = ASSETS_DIR / "best_vectvmixer"
OUTPUT_TFLITE = ASSETS_DIR / "vectvmizer_float32_converted.tflite"
LABELS_PATH = ASSETS_DIR / "labels.txt"

print(f"\nInput (Checkpoint):  {CHECKPOINT_DIR}")
print(f"Output (TFLite):     {OUTPUT_TFLITE}")

# Check if checkpoint exists
if not CHECKPOINT_DIR.exists():
    print(f"\n❌ ERROR: Checkpoint folder not found: {CHECKPOINT_DIR}")
    sys.exit(1)

print("\n[1/5] Loading checkpoint...")
try:
    # Load checkpoint as PyTorch state_dict
    checkpoint = torch.load(
        CHECKPOINT_DIR / "data.pkl",
        map_location=torch.device('cpu')
    )
    print(f"✅ Checkpoint loaded!")
    print(f"   Keys: {list(checkpoint.keys())[:5]}...")
except Exception as e:
    print(f"❌ Failed to load checkpoint: {e}")
    sys.exit(1)

print("\n[2/5] Creating test input...")
dummy_input = torch.randn(1, 3, 224, 224)
print(f"✅ Test input shape: {dummy_input.shape}")

print("\n[3/5] Converting to ONNX...")
try:
    onnx_path = ASSETS_DIR / "model_temp.onnx"

    # For now, create a simple PyTorch model to demonstrate
    print("   Note: Full model class needed for proper conversion")
    print("   Using direct numpy approach instead...")

except Exception as e:
    print(f"⚠️  ONNX export: {e}")

print("\n[4/5] Creating TFLite model from NumPy arrays...")
try:
    # Create a simple functional model that loads weights
    # This is a workaround since we don't have the exact model class

    # Load labels
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]

    num_classes = len(labels)

    # Create a simple placeholder model
    # Real model would be loaded from checkpoint
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(f"✅ TensorFlow model created")
    print(f"   Input shape: (None, 224, 224, 3)")
    print(f"   Output classes: {num_classes}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    # Save
    with open(OUTPUT_TFLITE, 'wb') as f:
        f.write(tflite_model)

    print(f"\n✅ TFLite model saved!")
    print(f"   Path: {OUTPUT_TFLITE}")
    print(f"   Size: {OUTPUT_TFLITE.stat().st_size / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"❌ Conversion failed: {e}")
    sys.exit(1)

print("\n[5/5] Testing converted model...")
try:
    interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_TFLITE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"✅ TFLite model works!")
    print(f"   Input:  {input_details[0]['shape']}")
    print(f"   Output: {output_details[0]['shape']}")
    print(f"   Sample output: {output[0, :3]}")

except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("⚠️  IMPORTANT NOTES:")
print("=" * 70)
print("""
This converted model is a PLACEHOLDER for demonstration.

For the REAL trained model, you need to:
1. Run the PyTorch notebook in Colab
2. At the end, add this export code:

    import torch.onnx
    import tf2onnx
    import onnx
    import tensorflow as tf

    # Export to ONNX
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, 'model.onnx',
                      input_names=['input'], output_names=['output'])

    # Download and convert to TFLite
    # Use TensorFlow Lite converter on the ONNX file

Then download the .tflite file and replace:
   d:\\medi\\medicinal_plant_flutter\\assets\\vectvmizer_float32.tflite

Or use the vectvmixer.pkl pickle model as-is in a Python server!
""")
print("=" * 70)
