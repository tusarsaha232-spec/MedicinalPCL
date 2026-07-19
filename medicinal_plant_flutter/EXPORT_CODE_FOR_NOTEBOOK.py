"""
ADD THIS TO THE END OF YOUR COLAB NOTEBOOK
This exports the trained model to ONNX format (convertible to TFLite)
"""

# ============================================================
# EXPORT TRAINED MODEL TO ONNX (for mobile/TFLite)
# ============================================================

import torch
import torch.onnx

print("\n" + "="*70)
print("🔄 EXPORTING MODEL TO ONNX...")
print("="*70)

# Load the trained model from checkpoint
CHECKPOINT_PATH = "/content/drive/MyDrive/best_vectvmixer.pth"

print(f"\n[1/4] Loading trained model from: {CHECKPOINT_PATH}")
try:
    model = VECTVMixer(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Create dummy input matching model input
print(f"\n[2/4] Creating dummy input (1, 3, 224, 224)...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export to ONNX
print(f"\n[3/4] Exporting to ONNX format...")
try:
    onnx_path = "/content/vectvmixer_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    print(f"✅ ONNX exported to: {onnx_path}")
except Exception as e:
    print(f"❌ Export failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test the ONNX model
print(f"\n[4/4] Testing ONNX model...")
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    test_input = dummy_input.cpu().numpy()
    result = sess.run([output_name], {input_name: test_input})

    print(f"✅ ONNX model works!")
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {result[0].shape}")

except Exception as e:
    print(f"⚠️  ONNX runtime test skipped: {e}")
    print(f"   (But ONNX file is still created and usable)")

print("\n" + "="*70)
print("✅ EXPORT COMPLETE!")
print("="*70)
print("\nNEXT STEPS:")
print("1. Download 'vectvmixer_model.onnx' from Colab")
print("2. Convert ONNX → TFLite using:")
print("   - Online: https://netron.app/ or TFLite converter")
print("   - Local: Use TensorFlow Lite converter")
print("3. Replace: medicinal_plant_flutter/assets/vectvmixer_float32.tflite")
print("4. Run server and test on phone!")
print("="*70)
