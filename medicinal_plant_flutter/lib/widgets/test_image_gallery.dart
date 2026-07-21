import 'package:flutter/material.dart';
import 'dart:io';
import 'package:flutter/services.dart';
import '../constants/asset_images.dart';

class TestImageGallery extends StatefulWidget {
  final Function(File) onImageSelected;

  const TestImageGallery({Key? key, required this.onImageSelected}) : super(key: key);

  @override
  State<TestImageGallery> createState() => _TestImageGalleryState();
}

class _TestImageGalleryState extends State<TestImageGallery> {
  String? _selectedPlant;

  @override
  Widget build(BuildContext context) {
    if (_selectedPlant == null) {
      // Show plant categories
      final plants = testPlantImages.entries
          .where((e) => e.value.isNotEmpty)
          .map((e) => e.key)
          .toList();

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(
              'Test Plants (${plants.length})',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
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
              itemCount: plants.length,
              itemBuilder: (context, index) {
                final plant = plants[index];
                final count = testPlantImages[plant]?.length ?? 0;
                final displayName = plant.replaceAll('_', ' ');

                return GestureDetector(
                  onTap: () => setState(() => _selectedPlant = plant),
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
                          Icon(Icons.eco, size: 48, color: Colors.white),
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
      );
    } else {
      // Show images for selected plant
      final images = testPlantImages[_selectedPlant] ?? [];

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => setState(() => _selectedPlant = null),
                ),
                Expanded(
                  child: Text(
                    '${_selectedPlant?.replaceAll('_', ' ')} (${images.length} images)',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
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
                  onTap: () => _loadAndSelect(imagePath),
                  child: Card(
                    elevation: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8),
                        image: DecorationImage(
                          image: AssetImage(imagePath),
                          fit: BoxFit.cover,
                        ),
                      ),
                      child: Stack(
                        alignment: Alignment.bottomRight,
                        children: [
                          Container(
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(8),
                              gradient: LinearGradient(
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                                colors: [Colors.transparent, Colors.black26],
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(4),
                            child: Text(
                              '${index + 1}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
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
      );
    }
  }

  Future<void> _loadAndSelect(String assetPath) async {
    try {
      final ByteData data = await rootBundle.load(assetPath);
      final Directory tempDir = Directory.systemTemp;
      final File tempFile = File('${tempDir.path}/test_plant_image.jpg');
      await tempFile.writeAsBytes(data.buffer.asUint8List());
      widget.onImageSelected(tempFile);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      debugPrint('Error loading image: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }
}
