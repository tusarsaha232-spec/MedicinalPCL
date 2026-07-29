import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/tflite_model.dart';

final modelProvider = FutureProvider<TFLiteModel>((ref) async {
  final model = TFLiteModel();
  await model.loadModel();
  ref.onDispose(() => model.dispose());
  return model;
});

final predictionProvider = StateProvider<Map<String, dynamic>?>((ref) => null);
final predictionTimeProvider = StateProvider<DateTime?>((ref) => null);
final isProcessingProvider = StateProvider<bool>((ref) => false);
final errorMessageProvider = StateProvider<String?>((ref) => null);
final selectedImagePathProvider = StateProvider<String?>((ref) => null);
