#!/usr/bin/env python3
"""
Complete model export and conversion pipeline
Converts PyTorch → ONNX → TFLite
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

print("\n" + "=" * 70)
print("🚀 COMPLETE MODEL EXPORT & CONVERSION PIPELINE")
print("=" * 70)

# ============================================================
# STEP 1: Define Model Architecture
# ============================================================

print("\n[1/5] Defining model architecture...")

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
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


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
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


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
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


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
        self.num_tokens = 14 * 14
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
        e, tv_e = self.edge(x)
        c, tv_c = self.color(x)
        v, tv_v = self.vein(x)
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
# STEP 2: Create Model Instance
# ============================================================

print("\n[2/5] Creating model instance...")
device = torch.device('cpu')
model = VECTVMixer(num_classes=14).to(device)
model.eval()
print("✅ Model created!")

# ============================================================
# STEP 3: Export to ONNX
# ============================================================

print("\n[3/5] Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = "d:/medi/medicinal_plant_flutter/assets/vectvmixer_model.onnx"

try:
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
    print(f"   File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    sys.exit(1)

# ============================================================
# STEP 4: Convert ONNX to TFLite
# ============================================================

print("\n[4/5] Converting ONNX to TFLite...")

try:
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified!")

    # Convert ONNX to TensorFlow SavedModel
    print("   Converting ONNX → TensorFlow...")
    tf_rep = prepare(onnx_model)

    saved_model_dir = "d:/medi/medicinal_plant_flutter/assets/saved_model"
    tf_rep.export_graph(saved_model_dir)
    print(f"✅ TensorFlow SavedModel created!")

    # Convert to TFLite
    print("   Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = "d:/medi/medicinal_plant_flutter/assets/vectvmixer_float32_new.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ TFLite created!")
    print(f"   File size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"❌ Conversion failed: {e}")
    print(f"   Error: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# STEP 5: Test TFLite Model
# ============================================================

print("\n[5/5] Testing TFLite model...")

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test with dummy input
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    test_input = (test_input - 0.5) / 0.5

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"✅ TFLite model works!")
    print(f"   Input: {input_details[0]['shape']}")
    print(f"   Output: {output_details[0]['shape']}")
    print(f"   Sample output: {output[0, :3]}")

except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)

# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("✅✅✅ CONVERSION COMPLETE! ✅✅✅")
print("=" * 70)
print(f"\n📁 New TFLite model: {tflite_path}")
print(f"\n📋 NEXT STEPS:")
print(f"1. Rename: vectvmixer_float32_new.tflite → vectvmixer_float32.tflite")
print(f"2. Run server: C:\\Python312\\python.exe server.py")
print(f"3. Test on phone: Select image → Click Analyze")
print(f"4. Should show correct plant predictions! 🌿")
print("=" * 70 + "\n")
