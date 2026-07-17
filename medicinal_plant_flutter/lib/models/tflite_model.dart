import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io';
import 'dart:developer' as developer;
import 'dart:typed_data';

class TFLiteModel {
  late Interpreter _interpreter;
  late List<String> _labels;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  Future<void> loadModel() async {
    try {
      developer.log('🔄 STEP 1: Loading TFLite model from assets...');

      try {
        final byteData = await rootBundle.load('assets/vectvmixer_float32.tflite');
        _interpreter = Interpreter.fromBuffer(byteData.buffer.asUint8List());
        developer.log('✅ STEP 1 SUCCESS: Model interpreter loaded from buffer');
      } catch (e) {
        developer.log('❌ STEP 1 FAILED: ${e.toString()}');
        rethrow;
      }

      developer.log('🔄 STEP 2: Getting model shapes...');
      final inputShape = _interpreter.getInputTensor(0).shape;
      final outputShape = _interpreter.getOutputTensor(0).shape;
      developer.log('✅ STEP 2 SUCCESS: Input: $inputShape, Output: $outputShape');

      developer.log('🔄 STEP 3: Loading labels...');
      try {
        final labelData = await rootBundle.loadString('assets/labels.txt');
        _labels = labelData.split('\n').where((l) => l.trim().isNotEmpty).toList();
        developer.log('✅ STEP 3 SUCCESS: Loaded ${_labels.length} classes: $_labels');
      } catch (e) {
        developer.log('❌ STEP 3 FAILED: ${e.toString()}');
        rethrow;
      }

      _isLoaded = true;
      developer.log('✅✅✅ ALL STEPS COMPLETE - MODEL READY ✅✅✅');
    } catch (e) {
      developer.log('❌ MODEL INITIALIZATION FAILED: $e', error: e);
      rethrow;
    }
  }

  Future<Map<String, dynamic>> predict(File imageFile) async {
    if (!_isLoaded) throw Exception('Model not loaded');

    try {
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('Failed to decode image');

      developer.log('🔄 Preprocessing image...');
      final input = _preprocessImage(image);

      developer.log('🔄 Running inference...');
      final output = List<List<double>>.filled(1, List<double>.filled(_labels.length, 0.0));

      // Convert input list to ByteBuffer format
      final inputBuffer = Float32List.fromList(input);
      _interpreter.run(inputBuffer, output);

      int maxIdx = 0;
      double maxVal = output[0][0];
      for (int i = 1; i < output[0].length; i++) {
        if (output[0][i] > maxVal) {
          maxVal = output[0][i];
          maxIdx = i;
        }
      }

      final probs = _softmax(output[0]);

      for (int i = 0; i < _labels.length; i++) {
        developer.log('${_labels[i]}: ${output[0][i].toStringAsFixed(4)} (prob: ${probs[i].toStringAsFixed(4)})');
      }

      return {
        'label': _labels[maxIdx],
        'confidence': probs[maxIdx],
        'scores': Map.fromIterable(
          List.generate(_labels.length, (i) => i),
          key: (i) => _labels[i],
          value: (i) => probs[i],
        ),
      };
    } catch (e) {
      developer.log('❌ Inference error: $e', error: e);
      rethrow;
    }
  }

  List<double> _preprocessImage(img.Image image) {
    final resized = img.copyResize(image, width: 224, height: 224);
    final bytes = <double>[];

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixelSafe(x, y);

        final r = (pixel.r is int ? (pixel.r as int).toDouble() : pixel.r as double) / 255.0;
        final g = (pixel.g is int ? (pixel.g as int).toDouble() : pixel.g as double) / 255.0;
        final b = (pixel.b is int ? (pixel.b as int).toDouble() : pixel.b as double) / 255.0;

        bytes.add((r - 0.5) / 0.5);
        bytes.add((g - 0.5) / 0.5);
        bytes.add((b - 0.5) / 0.5);
      }
    }

    return bytes;
  }

  List<double> _softmax(List<double> values) {
    final maxVal = values.reduce((a, b) => a > b ? a : b);
    final exps = values.map((v) => _exp(v - maxVal)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  double _exp(double x) {
    if (x > 20) return 1e9;
    if (x < -20) return 0;
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i < 20; i++) {
      term *= x / i;
      result += term;
    }
    return result;
  }

  void dispose() {
    _interpreter.close();
  }
}
