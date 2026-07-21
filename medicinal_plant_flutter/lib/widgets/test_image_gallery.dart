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
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(
              'Test Plants (${PlantAssets.plantNames.length})',
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
              itemCount: PlantAssets.plantNames.length,
              itemBuilder: (context, index) {
                final plant = PlantAssets.plantNames[index];
                final count = PlantAssets.getImageCount(plant);
                final displayName = PlantAssets.getDisplayName(plant);

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
      final images = PlantAssets.getImages(_selectedPlant!);

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
                    '${PlantAssets.getDisplayName(_selectedPlant!)} (${images.length} images)',
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
                    clipBehavior: Clip.antiAlias,
                    child: _ImageThumbnail(imagePath: imagePath, index: index),
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
      final File tempFile = File('${tempDir.path}/test_plant_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await tempFile.writeAsBytes(data.buffer.asUint8List());
      widget.onImageSelected(tempFile);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      debugPrint('❌ Error loading image: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }
}

class _ImageThumbnail extends StatefulWidget {
  final String imagePath;
  final int index;

  const _ImageThumbnail({required this.imagePath, required this.index});

  @override
  State<_ImageThumbnail> createState() => _ImageThumbnailState();
}

class _ImageThumbnailState extends State<_ImageThumbnail> {
  bool _loaded = false;
  bool _error = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _loadImage();
  }

  void _loadImage() {
    final AssetImage assetImage = AssetImage(widget.imagePath);
    precacheImage(assetImage, context).then((_) {
      if (mounted) {
        setState(() {
          _loaded = true;
          _error = false;
        });
      }
    }).catchError((e) {
      debugPrint('❌ Failed to load: ${widget.imagePath} - $e');
      if (mounted) {
        setState(() {
          _loaded = true;
          _error = true;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_loaded) {
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
              Text('${widget.index + 1}', style: const TextStyle(fontSize: 10)),
            ],
          ),
        ),
      );
    }

    if (_error) {
      return Container(
        color: Colors.grey.shade300,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.broken_image, color: Colors.red.shade400, size: 28),
            const SizedBox(height: 4),
            Text('${widget.index + 1}', style: const TextStyle(fontSize: 10)),
          ],
        ),
      );
    }

    return Image.asset(
      widget.imagePath,
      fit: BoxFit.cover,
      cacheHeight: 200,
      cacheWidth: 200,
      filterQuality: FilterQuality.low,
    );
  }
}
