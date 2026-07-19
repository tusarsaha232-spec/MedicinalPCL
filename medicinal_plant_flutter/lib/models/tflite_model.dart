import 'package:flutter/services.dart';
import 'dart:io';
import 'dart:developer' as developer;
import 'dart:convert';
import 'package:http/http.dart' as http;

class TFLiteModel {
  late List<String> _labels;
  bool _isLoaded = false;
  static const String _serverUrl = 'http://192.168.x.x:8000'; // UPDATE WITH YOUR PC IP

  bool get isLoaded => _isLoaded;

  Future<void> loadModel() async {
    try {
      developer.log('🔄 Loading labels from assets...');

      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.trim().isNotEmpty).toList();
      developer.log('✅ Loaded ${_labels.length} classes: $_labels');

      // Test connection to server
      developer.log('🔄 Testing connection to API server...');
      try {
        final response = await http.get(Uri.parse('$_serverUrl/health')).timeout(
          const Duration(seconds: 5),
          onTimeout: () => throw Exception('Server timeout'),
        );

        if (response.statusCode == 200) {
          developer.log('✅ API server is ready!');
          _isLoaded = true;
          developer.log('✅✅✅ MODEL READY (Using FastAPI Server) ✅✅✅');
        } else {
          throw Exception('Server returned ${response.statusCode}');
        }
      } catch (e) {
        developer.log('❌ Cannot connect to API server at $_serverUrl');
        developer.log('   Make sure server.py is running on your PC');
        developer.log('   Error: $e');
        rethrow;
      }
    } catch (e) {
      developer.log('❌ MODEL INITIALIZATION FAILED: $e', error: e);
      rethrow;
    }
  }

  Future<Map<String, dynamic>> predict(File imageFile) async {
    if (!_isLoaded) throw Exception('Model not loaded');

    try {
      developer.log('========== INFERENCE START (API SERVER) ==========');

      final imageBytes = await imageFile.readAsBytes();
      developer.log('✅ Image bytes read: ${imageBytes.length}');

      // Send image to server
      developer.log('🔄 Sending to $_serverUrl/predict...');
      final request = http.MultipartRequest('POST', Uri.parse('$_serverUrl/predict'))
        ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: 'image.jpg'));

      final response = await request.send().timeout(const Duration(seconds: 30));
      developer.log('✅ Response: ${response.statusCode}');

      if (response.statusCode != 200) {
        throw Exception('Server error: ${response.statusCode}');
      }

      final responseBody = await response.stream.bytesToString();
      final result = jsonDecode(responseBody) as Map<String, dynamic>;

      developer.log('✅ Prediction: ${result['predicted_class']}');
      developer.log('   Confidence: ${result['confidence']}');

      return {
        'label': result['predicted_class'] as String,
        'confidence': (result['confidence'] as num).toDouble(),
        'scores': result['all_predictions'] as Map<String, dynamic>,
      };
    } catch (e) {
      developer.log('❌ Error: $e', error: e);
      rethrow;
    }
  }

  void dispose() {
    // No cleanup needed for HTTP client
  }
}
