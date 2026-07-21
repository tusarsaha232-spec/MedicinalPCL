import 'package:flutter/material.dart';
import 'dart:io';
import 'package:flutter/services.dart';
import 'dart:async';

class AssetGallery extends StatefulWidget {
  final Function(File) onImageSelected;

  const AssetGallery({
    Key? key,
    required this.onImageSelected,
  }) : super(key: key);

  @override
  State<AssetGallery> createState() => _AssetGalleryState();
}

class _AssetGalleryState extends State<AssetGallery> {
  // Plant categories with their asset paths
  final Map<String, List<String>> plantImages = {
    'Aloe': [
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Aloe_barbadensis_miller/100.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Aloe_barbadensis_miller/102.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Aloe_barbadensis_miller/140.jpg',
    ],
    'Neem': [
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Azadirachta_indica/1.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Azadirachta_indica/10.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Azadirachta_indica/100.jpg',
    ],
    'Hibiscus': [
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Hibiscus_rosa_sinensis/1.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Hibiscus_rosa_sinensis/10.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Hibiscus_rosa_sinensis/100.jpg',
    ],
    'Mint': [
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Mentha_arvensis/1.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Mentha_arvensis/10.jpg',
      'assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/Mentha_arvensis/100.jpg',
    ],
  };

  String? _selectedPlant;

  @override
  Widget build(BuildContext context) {
    if (_selectedPlant == null) {
      // Show plant categories
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: Text(
              'Select a Plant',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
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
              itemCount: plantImages.keys.length,
              itemBuilder: (context, index) {
                final plant = plantImages.keys.toList()[index];
                return GestureDetector(
                  onTap: () {
                    setState(() => _selectedPlant = plant);
                  },
                  child: Card(
                    elevation: 2,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Colors.green.shade300,
                            Colors.green.shade700,
                          ],
                        ),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.eco,
                            size: 48,
                            color: Colors.white,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            plant,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
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
    } else {
      // Show images for selected plant
      final images = plantImages[_selectedPlant]!;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => setState(() => _selectedPlant = null),
                ),
                Expanded(
                  child: Text(
                    _selectedPlant!,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
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
                return GestureDetector(
                  onTap: () => _loadAndSelect(images[index]),
                  child: Card(
                    elevation: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8),
                        image: DecorationImage(
                          image: AssetImage(images[index]),
                          fit: BoxFit.cover,
                        ),
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
      // Load asset and save to temp directory
      final ByteData data = await rootBundle.load(assetPath);
      final Directory tempDir = Directory.systemTemp;
      final File tempFile = File('${tempDir.path}/test_image.jpg');
      await tempFile.writeAsBytes(data.buffer.asUint8List());

      widget.onImageSelected(tempFile);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      debugPrint('Error loading image: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }
}
