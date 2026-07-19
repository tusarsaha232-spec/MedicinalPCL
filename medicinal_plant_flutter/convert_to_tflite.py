#!/usr/bin/env python3
"""
Convert PyTorch model to TFLite
"""

import torch
import numpy as np
import tensorflow as tf
from pathlib import Path

print("=" * 70)
print("PYTORCH → TFLITE CONVERTER")
print("=" * 70)

# Paths
ASSETS_DIR = Path(__file__).parent / "assets"
PTH_PATH = ASSETS_DIR / "best_vectvmizer.pth"
TFLITE_OUTPUT = ASSETS_DIR / "vectvmizer_float32_new.tflite"

print(f"\nInput (PyTorch):  {PTH_PATH}")
print(f"Output (TFLite):  {TFLITE_OUTPUT}")

# Check if .pth file exists
if not PTH_PATH.exists():
    print(f"\n❌ ERROR: {PTH_PATH} not found!")
    print("\nTO FIX:")
    print("1. Download 'best_vectvmizer.pth' from Google Drive")
    print("   Location: /content/drive/MyDrive/best_vectvmizer.pth")
    print("2. Save it to: d:\\medi\\medicinal_plant_flutter\\assets\\")
    print("3. Run this script again")
    exit(1)

print("\n[1/4] Loading PyTorch model...")
try:
    # Load model - need class definition
    from pth_leaf_c_2_model import VECTVMixer  # Import model class

    model = VECTVMixer(num_classes=14)
    model.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
    model.eval()
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    print("\nNeed model class definition from the training notebook")
    exit(1)

print("\n[2/4] Create dummy input and test PyTorch model...")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    print(f"✅ PyTorch output shape: {pytorch_output.shape}")
    print(f"   Output: {pytorch_output[0, :3]}")  # First 3 values
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

print("\n[3/4] Converting to ONNX (intermediate)...")
try:
    onnx_path = ASSETS_DIR / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"✅ ONNX model saved: {onnx_path}")
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    print("   Trying direct PyTorch to TFLite...")

print("\n[4/4] Converting ONNX to TFLite...")
try:
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save as SavedModel first
    saved_model_path = ASSETS_DIR / "saved_model"
    tf_rep.export_graph(saved_model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ TFLite model saved: {TFLITE_OUTPUT}")
    print(f"   Size: {TFLITE_OUTPUT.stat().st_size / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")
    exit(1)

print("\n[5/5] Testing converted TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_OUTPUT))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test with dummy data
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    test_input = (test_input - 0.5) / 0.5  # Normalize

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"✅ TFLite model works!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output: {output[0, :3]}")

except Exception as e:
    print(f"❌ TFLite test failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ CONVERSION COMPLETE!")
print("=" * 70)
print(f"\nNEXT STEPS:")
print(f"1. Replace old model:")
print(f"   mv {TFLITE_OUTPUT} {ASSETS_DIR}/vectvmizer_float32.tflite")
print(f"2. Rebuild Flutter app:")
print(f"   flutter clean && flutter build apk --debug")
print(f"3. Test on phone!")
print("=" * 70)
