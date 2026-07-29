import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../constants/asset_images.dart';
import '../providers/model_provider.dart';
import '../widgets/image_preview.dart';
import '../widgets/result_card.dart';
import '../widgets/test_image_gallery.dart';

class PlantClassifierScreen extends ConsumerStatefulWidget {
  const PlantClassifierScreen({Key? key}) : super(key: key);

  @override
  ConsumerState<PlantClassifierScreen> createState() => _PlantClassifierScreenState();
}

class _PlantClassifierScreenState extends ConsumerState<PlantClassifierScreen> with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late AnimationController _slideController;
  late AnimationController _scaleController;

  List<String> get _sampleImages => PlantAssets.plantNames
      .map((plant) => PlantAssets.getDisplayName(plant))
      .toList();

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(duration: const Duration(milliseconds: 800), vsync: this);
    _slideController = AnimationController(duration: const Duration(milliseconds: 600), vsync: this);
    _scaleController = AnimationController(duration: const Duration(milliseconds: 500), vsync: this);
  }

  Future<void> _runInference() async {
    final selectedImagePath = ref.read(selectedImagePathProvider);
    if (selectedImagePath == null) {
      ref.read(errorMessageProvider.notifier).state = 'Please select an image first';
      return;
    }

    ref.read(isProcessingProvider.notifier).state = true;
    ref.read(errorMessageProvider.notifier).state = null;

    try {
      final model = await ref.read(modelProvider.future);
      final result = await model.predict(File(selectedImagePath));
      ref.read(predictionProvider.notifier).state = result;
      ref.read(predictionTimeProvider.notifier).state = DateTime.now();
    } catch (e) {
      debugPrint('Full error: $e');
      ref.read(errorMessageProvider.notifier).state = 'Inference failed: $e';
    } finally {
      ref.read(isProcessingProvider.notifier).state = false;
    }
  }

  void _refreshModelStatus() {
    ref.read(errorMessageProvider.notifier).state = null;
    ref.invalidate(modelProvider);
  }

  void _clearImage() {
    _fadeController.reset();
    _slideController.reset();
    _scaleController.reset();
    ref.read(selectedImagePathProvider.notifier).state = null;
    ref.read(predictionProvider.notifier).state = null;
    ref.read(predictionTimeProvider.notifier).state = null;
    ref.read(errorMessageProvider.notifier).state = null;
  }

  Future<void> _selectImageFromGallery() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    ref.read(selectedImagePathProvider.notifier).state = pickedFile.path;
    ref.read(predictionProvider.notifier).state = null;
    ref.read(predictionTimeProvider.notifier).state = null;
    ref.read(errorMessageProvider.notifier).state = null;
  }

  Future<void> _takePhotoWithCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile == null) return;

    ref.read(selectedImagePathProvider.notifier).state = pickedFile.path;
    ref.read(predictionProvider.notifier).state = null;
    ref.read(predictionTimeProvider.notifier).state = null;
    ref.read(errorMessageProvider.notifier).state = null;
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _scaleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final modelState = ref.watch(modelProvider);
    final selectedImagePath = ref.watch(selectedImagePathProvider);
    final selectedImage = selectedImagePath == null ? null : File(selectedImagePath);
    final prediction = ref.watch(predictionProvider);
    final predictionTime = ref.watch(predictionTimeProvider);
    final isProcessing = ref.watch(isProcessingProvider);
    final providerErrorMessage = ref.watch(errorMessageProvider);
    final modelErrorMessage = modelState.hasError
        ? 'Server connection failed. Use refresh after the hosted service wakes up.'
        : null;
    final errorMessage = providerErrorMessage ?? modelErrorMessage;
    final modelLoaded = modelState.hasValue;
    final modelLoading = modelState.isLoading;
    final modelFailed = modelState.hasError;

    ref.listen<Map<String, dynamic>?>(predictionProvider, (previous, next) {
      if (next != null && previous != next) {
        _fadeController.forward(from: 0);
        _slideController.forward(from: 0);
        _scaleController.forward(from: 0);
      }
    });

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.green.shade50,
              Colors.teal.shade50,
            ],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.green.shade100,
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(color: Colors.green.shade200),
                              ),
                              child: Text(
                                'University Research Project',
                                style: TextStyle(
                                  color: Colors.green.shade800,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                            const SizedBox(height: 10),
                            Text(
                              'Medicinal Plant Recognition',
                              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                    fontWeight: FontWeight.w900,
                                    color: Colors.green.shade900,
                                    height: 1.08,
                                  ),
                            ),
                            const SizedBox(height: 5),
                            Text(
                              'CNN-based leaf image classification for research demonstration',
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.green.shade700,
                                fontWeight: FontWeight.w600,
                                height: 1.25,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 12),
                      IconButton(
                        icon: const Icon(Icons.info_outline),
                        onPressed: () => Navigator.of(context).pushNamed('/about'),
                        tooltip: 'About App',
                        style: IconButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.green.shade700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: _buildMetricChip(
                          icon: Icons.category_outlined,
                          value: '${PlantAssets.plantNames.length}',
                          label: 'Classes',
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: _buildMetricChip(
                          icon: Icons.image_search_outlined,
                          value: 'Samples',
                          label: 'Dataset',
                          color: Colors.teal,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  _buildServerStatusCard(
                    isReady: modelLoaded,
                    isLoading: modelLoading,
                    hasError: modelFailed,
                  ),
                  if (errorMessage != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: AnimatedOpacity(
                        opacity: errorMessage != null ? 1 : 0,
                        duration: const Duration(milliseconds: 300),
                        child: Container(
                          decoration: BoxDecoration(
                            color: Colors.red.shade50,
                            border: Border.all(color: Colors.red.shade300),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            children: [
                              Icon(Icons.error_outline, color: Colors.red.shade600),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  errorMessage,
                                  style: TextStyle(
                                    color: Colors.red.shade700,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  const SizedBox(height: 24),
                  Text(
                    'Input Specimen',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade800,
                        ),
                  ),
                  const SizedBox(height: 12),
                  AnimatedScale(
                    scale: selectedImage != null ? 1.0 : 0.95,
                    duration: const Duration(milliseconds: 300),
                    child: ImagePreview(image: selectedImage),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    'Image Source',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade800,
                        ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: _buildModernButton(
                          icon: Icons.photo_library,
                          label: 'Gallery',
                          onPressed: _selectImageFromGallery,
                          color: Colors.blue,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildModernButton(
                          icon: Icons.camera_alt,
                          label: 'Camera',
                          onPressed: _takePhotoWithCamera,
                          color: Colors.purple,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: _buildModernButton(
                      icon: Icons.collections_bookmark,
                      label: 'Sample Dataset',
                      onPressed: () => showModalBottomSheet(
                        context: context,
                        isScrollControlled: true,
                        useSafeArea: true,
                        backgroundColor: Colors.white,
                        shape: const RoundedRectangleBorder(
                          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                        ),
                        builder: (context) => Container(
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                          ),
                          child: Column(
                            children: [
                              Container(
                                width: 50,
                                height: 5,
                                margin: const EdgeInsets.symmetric(vertical: 12),
                                decoration: BoxDecoration(
                                  color: Colors.grey.shade300,
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                              const Expanded(child: TestImageGallery()),
                            ],
                          ),
                        ),
                      ),
                      color: Colors.teal,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    children: [
                      Expanded(
                        flex: 2,
                        child: Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [Colors.green.shade400, Colors.teal.shade400],
                            ),
                            borderRadius: BorderRadius.circular(12),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.green.withValues(alpha: 0.3),
                                blurRadius: 8,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: ElevatedButton.icon(
                            onPressed: isProcessing || selectedImage == null || !modelLoaded ? null : _runInference,
                            icon: isProcessing
                                ? SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      valueColor: AlwaysStoppedAnimation<Color>(
                                        Colors.white.withValues(alpha: 0.8),
                                      ),
                                    ),
                                  )
                                : const Icon(Icons.search, size: 20),
                            label: Text(isProcessing ? 'Evaluating...' : 'Run Classification'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              padding: const EdgeInsets.symmetric(vertical: 14),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.grey.shade300,
                          borderRadius: BorderRadius.circular(12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withValues(alpha: 0.2),
                              blurRadius: 8,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ElevatedButton.icon(
                          onPressed: _clearImage,
                          icon: const Icon(Icons.close, size: 20),
                          label: const Text('Clear'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            foregroundColor: Colors.grey.shade700,
                            padding: const EdgeInsets.symmetric(
                              vertical: 14,
                              horizontal: 16,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  if (prediction != null)
                    FadeTransition(
                      opacity: Tween<double>(begin: 0, end: 1).animate(_fadeController),
                      child: SlideTransition(
                        position: Tween<Offset>(
                          begin: const Offset(0, 0.3),
                          end: Offset.zero,
                        ).animate(_slideController),
                        child: ResultCard(
                          plantName: PlantAssets.getDisplayName(prediction['label'] as String),
                          scientificName: prediction['label'] as String,
                          confidence: (prediction['confidence'] as double) * 100,
                          predictionTime: predictionTime,
                        ),
                      ),
                    ),
                  const SizedBox(height: 20),
                  Text(
                    'Research Classes',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade800,
                        ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: _sampleImages.asMap().entries.map((entry) {
                      final index = entry.key;
                      final plant = entry.value;
                      return AnimatedOpacity(
                        opacity: 1,
                        duration: Duration(milliseconds: 200 + (index * 50)),
                        child: Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                Colors.green.shade100,
                                Colors.teal.shade100,
                              ],
                            ),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.green.withValues(alpha: 0.1),
                                blurRadius: 4,
                              ),
                            ],
                          ),
                          child: Chip(
                            label: Text(
                              plant,
                              style: TextStyle(
                                color: Colors.green.shade800,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                            backgroundColor: Colors.transparent,
                            side: BorderSide(
                              color: Colors.green.shade200,
                              width: 0.5,
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildModernButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
    required Color color,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.22)),
        boxShadow: [
          BoxShadow(
            color: color.withValues(alpha: 0.1),
            blurRadius: 14,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 18),
        label: Text(label),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          foregroundColor: color,
          padding: const EdgeInsets.symmetric(vertical: 12),
        ),
      ),
    );
  }

  Widget _buildServerStatusCard({
    required bool isReady,
    required bool isLoading,
    required bool hasError,
  }) {
    final Color accent = hasError
        ? const Color(0xFFC2413A)
        : isReady
            ? const Color(0xFF16804A)
            : const Color(0xFFB7791F);
    final IconData icon = hasError
        ? Icons.cloud_off_outlined
        : isReady
            ? Icons.verified_outlined
            : Icons.sync_outlined;
    final String title = hasError
        ? 'Server Not Ready'
        : isReady
            ? 'CNN Server Ready'
            : 'Checking Server';
    final String subtitle = hasError
        ? 'Render may be waking up. Tap refresh to check again.'
        : isReady
            ? 'Health check passed. Classification is available.'
            : 'Connecting to the hosted classifier endpoint...';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: accent.withValues(alpha: 0.22)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: accent.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(14),
            ),
            child: isLoading
                ? Padding(
                    padding: const EdgeInsets.all(12),
                    child: CircularProgressIndicator(
                      strokeWidth: 2.4,
                      valueColor: AlwaysStoppedAnimation<Color>(accent),
                    ),
                  )
                : Icon(icon, color: accent, size: 25),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Color(0xFF173527),
                    fontSize: 16,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: Color(0xFF64766C),
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    height: 1.25,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          IconButton.filledTonal(
            onPressed: isLoading ? null : _refreshModelStatus,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh server status',
            style: IconButton.styleFrom(
              foregroundColor: accent,
              backgroundColor: accent.withValues(alpha: 0.1),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricChip({
    required IconData icon,
    required String value,
    required String label,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.88),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withValues(alpha: 0.22)),
        boxShadow: [
          BoxShadow(
            color: color.withValues(alpha: 0.08),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(icon, size: 19, color: color),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.green.shade900,
                    fontSize: 14,
                    fontWeight: FontWeight.w900,
                    height: 1,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.grey.shade600,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    height: 1.1,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
