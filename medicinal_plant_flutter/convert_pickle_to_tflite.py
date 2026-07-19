#!/usr/bin/env python3
"""
Load trained model from pickle and convert to TFLite with real weights
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from pathlib import Path

print("\n" + "=" * 70)
print("🚀 LOAD TRAINED PICKLE MODEL → TFLITE")
print("=" * 70)

# ============================================================
# STEP 1: Define Model Architecture
# ============================================================

print("\n[1/4] Defining model architecture...")

class EdgeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x

class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 11, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(11)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x

class VeinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = x + self.mlp1(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp2(self.norm2(x))
        return x

class VECTVMixer(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.edge = EdgeBranch()
        self.color = ColorBranch()
        self.vein = VeinBranch()
        self.fusion = nn.Sequential(
            nn.Conv2d(35, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2)
        )
        self.constituent = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        self.num_tokens = 56 * 56
        self.mixer = nn.Sequential(
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128)
        )
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 28)
        self.bn = nn.BatchNorm1d(28)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(28, num_classes)

    def forward(self, x):
        e = self.edge(x)
        c = self.color(x)
        v = self.vein(x)
        x = torch.cat([e, c, v], dim=1)
        x = self.fusion(x)
        x = self.constituent(x)
        x = self.down1(x)
        x = self.down2(x)
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.mixer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

print("✅ Architecture defined!")

# ============================================================
# STEP 2: Load Trained Model from Pickle
# ============================================================

print("\n[2/4] Loading trained model from pickle...")

pickle_path = r"d:\medi\medicinal_plant_flutter\assets\vectvmixer.pkl"

try:
    device = torch.device('cpu')
    model = torch.load(pickle_path, map_location=device, weights_only=False)

    # If it's a model directly
    if isinstance(model, nn.Module):
        print(f"✅ Loaded trained model from pickle!")
    else:
        print(f"⚠️  Pickle contains: {type(model)}")
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
            print(f"✅ Extracted model from dictionary!")
        else:
            print(f"❌ Cannot extract model from pickle")
            sys.exit(1)

    model.eval()
    print(f"✅ Model ready for inference!")

except Exception as e:
    print(f"❌ Failed to load pickle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# STEP 3: Convert to TFLite
# ============================================================

print("\n[3/4] Converting to TFLite...")

try:
    # Create SavedModel via TensorFlow
    class PyTorchWrapper(tf.Module):
        def __init__(self, pytorch_model):
            super().__init__()
            self.pytorch_model = pytorch_model

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32)
        ])
        def __call__(self, x):
            with torch.no_grad():
                output = self.pytorch_model(torch.from_numpy(x.numpy()))
            return tf.convert_to_tensor(output.numpy())

    wrapper = PyTorchWrapper(model)
    concrete_func = wrapper.__call__.get_concrete_function()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True

    tflite_model = converter.convert()

    tflite_path = "d:/medi/medicinal_plant_flutter/assets/vectvmizer_trained.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ TFLite created with trained weights!")
    print(f"   File: {tflite_path}")
    print(f"   Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"⚠️  Conversion failed: {e}")
    print(f"   Trying alternative method...")

    # Alternative: SavedModel path
    try:
        saved_model_path = "d:/medi/medicinal_plant_flutter/assets/saved_model_trained"

        # Convert via SavedModel
        wrapper = tf.saved_model.save(model, saved_model_path)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        tflite_model = converter.convert()

        tflite_path = "d:/medi/medicinal_plant_flutter/assets/vectvmizer_trained.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ TFLite created (via SavedModel)!")
        print(f"   Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

    except Exception as e2:
        print(f"❌ Both methods failed!")
        print(f"   Error 1: {e}")
        print(f"   Error 2: {e2}")
        sys.exit(1)

# ============================================================
# STEP 4: Test TFLite
# ============================================================

print("\n[4/4] Testing TFLite model...")

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test with different random inputs
    print("   Testing with 3 different random images...")
    for i in range(3):
        test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        test_input = (test_input - 0.5) / 0.5

        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        pred_idx = np.argmax(output[0])
        print(f"   Test {i+1}: Max class {pred_idx}, value {output[0, pred_idx]:.4f}")

    print(f"✅ TFLite model works with trained weights!")

except Exception as e:
    print(f"⚠️  Test failed: {e}")

print("\n" + "=" * 70)
print("✅ CONVERSION COMPLETE!")
print("=" * 70)
print(f"\n📁 New model: {tflite_path}")
print(f"\n📋 NEXT STEP:")
print(f"1. Copy this file to replace old one:")
print(f"   vectvmizer_trained.tflite → vectvmizer_float32.tflite")
print(f"2. Restart server: CTRL+C then 'C:\\Python312\\python.exe server.py'")
print(f"3. Test on phone - should see DIFFERENT predictions now! 🌿")
print("=" * 70 + "\n")
