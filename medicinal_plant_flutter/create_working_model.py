#!/usr/bin/env python3
"""
Create a working TFLite model with deterministic output that varies by input
This will show that the app works correctly - you can then replace it with
your trained model from Colab later
"""

import numpy as np
import tensorflow as tf

print("\n" + "=" * 70)
print("🌿 CREATING WORKING TFLITE MODEL")
print("=" * 70)

# Create a simple CNN model
inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Output 14 classes
outputs = tf.keras.layers.Dense(14, activation='softmax', name='output')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("\n✅ Model created")
print(f"   Input: {model.input_shape}")
print(f"   Output: {model.output_shape}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = "d:/medi/medicinal_plant_flutter/assets/vectvmizer_float32.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ TFLite saved: {tflite_path}")
print(f"   Size: {len(tflite_model) / 1024:.2f} KB")

# Test with different images
print("\n[Test] Running predictions on different images...")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['Aloe', 'Neem', 'Ashoka', 'Centella', 'Hibiscus', 'Justicia',
          'Kalanchoe', 'Mint', 'Mikania', 'Moringa', 'Tulsi', 'Amla', 'Arjun', 'Weed']

print("\nPredictions for different images:")
for test_num in range(5):
    # Create different test images
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    test_input = (test_input - 0.5) / 0.5

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_idx = np.argmax(output[0])
    confidence = output[0, pred_idx]

    print(f"   Image {test_num+1}: {labels[pred_idx]} ({confidence:.1%})")

print("\n" + "=" * 70)
print("✅ DONE!")
print("=" * 70)
print(f"""
This model shows that the app works correctly.

IMPORTANT:
To use your actual trained model:
1. Export your trained model from Colab as ONNX or SavedModel
2. Convert to TFLite
3. Replace vectvmizer_float32.tflite with your trained model
4. Restart server and test

For now, restart server: CTRL+C then run again
""")
print("=" * 70 + "\n")
