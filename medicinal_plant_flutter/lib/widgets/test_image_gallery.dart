import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../constants/asset_images.dart';
import '../providers/model_provider.dart';

final selectedSamplePlantProvider = StateProvider.autoDispose<String?>((ref) => null);

class TestImageGallery extends ConsumerWidget {
  const TestImageGallery({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedPlant = ref.watch(selectedSamplePlantProvider);

    if (selectedPlant == null) {
      return Container(
        color: Colors.white,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Select Sample Images',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${PlantAssets.plantNames.length} medicinal plants - Tap to browse images',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                  ),
                ],
              ),
            ),
            Expanded(
              child: GridView.builder(
                padding: const EdgeInsets.all(8),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  childAspectRatio: 1,
                ),
                itemCount: PlantAssets.plantNames.length,
                itemBuilder: (context, index) {
                  final plant = PlantAssets.plantNames[index];
                  final count = PlantAssets.getImageCount(plant);
                  final displayName = PlantAssets.getDisplayName(plant);

                  return GestureDetector(
                    onTap: () => ref.read(selectedSamplePlantProvider.notifier).state = plant,
                    child: Card(
                      elevation: 2,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [Colors.green.shade300, Colors.green.shade700],
                          ),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.eco, size: 48, color: Colors.white),
                            const SizedBox(height: 12),
                            Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 8),
                              child: Text(
                                displayName,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              '$count images',
                              style: const TextStyle(fontSize: 11, color: Colors.white70),
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      );
    }

    final images = PlantAssets.getImages(selectedPlant);

    return Container(
      color: Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => ref.read(selectedSamplePlantProvider.notifier).state = null,
                ),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        PlantAssets.getDisplayName(selectedPlant),
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      Text(
                        '${images.length} sample images - Tap to select',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.all(8),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                mainAxisSpacing: 8,
                crossAxisSpacing: 8,
                childAspectRatio: 1,
              ),
              itemCount: images.length,
              itemBuilder: (context, index) {
                final imagePath = images[index];
                return GestureDetector(
                  onTap: () => _loadAndSelect(context, ref, imagePath),
                  child: Card(
                    elevation: 1,
                    clipBehavior: Clip.antiAlias,
                    child: _ImageThumbnail(imagePath: imagePath, index: index),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _loadAndSelect(BuildContext context, WidgetRef ref, String assetPath) async {
    try {
      final data = await rootBundle.load(assetPath);
      final tempDir = Directory.systemTemp;
      final tempFile = File('${tempDir.path}/test_plant_${DateTime.now().millisecondsSinceEpoch}.jpg');

      await tempFile.writeAsBytes(data.buffer.asUint8List());
      ref.read(selectedImagePathProvider.notifier).state = tempFile.path;
      ref.read(predictionProvider.notifier).state = null;
      ref.read(predictionTimeProvider.notifier).state = null;
      ref.read(errorMessageProvider.notifier).state = null;

      if (context.mounted) Navigator.of(context).pop();
    } catch (e) {
      debugPrint('Error loading image: $e');
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }
}

class _ImageThumbnail extends StatelessWidget {
  final String imagePath;
  final int index;

  const _ImageThumbnail({required this.imagePath, required this.index});

  @override
  Widget build(BuildContext context) {
    return Image.asset(
      imagePath,
      fit: BoxFit.cover,
      cacheHeight: 200,
      cacheWidth: 200,
      filterQuality: FilterQuality.low,
      frameBuilder: (context, child, frame, wasSynchronouslyLoaded) {
        if (wasSynchronouslyLoaded || frame != null) return child;
        return Container(
          color: Colors.grey.shade200,
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(height: 4),
                Text('${index + 1}', style: const TextStyle(fontSize: 10)),
              ],
            ),
          ),
        );
      },
      errorBuilder: (context, error, stackTrace) {
        debugPrint('Failed to load: $imagePath - $error');
        return Container(
          color: Colors.grey.shade300,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.broken_image, color: Colors.red.shade400, size: 28),
              const SizedBox(height: 4),
              Text('${index + 1}', style: const TextStyle(fontSize: 10)),
            ],
          ),
        );
      },
    );
  }
}
