import 'package:flutter/material.dart';
import 'dart:io';

class AssetImageBrowser extends StatefulWidget {
  final Function(File) onImageSelected;

  const AssetImageBrowser({
    Key? key,
    required this.onImageSelected,
  }) : super(key: key);

  @override
  State<AssetImageBrowser> createState() => _AssetImageBrowserState();
}

class _AssetImageBrowserState extends State<AssetImageBrowser> {
  List<String> _plantFolders = [];
  String? _selectedPlant;
  List<File> _plantImages = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPlantFolders();
  }

  Future<void> _loadPlantFolders() async {
    try {
      // Load from app directory
      final appDir = Directory.current;
      final assetsDir = Directory('${appDir.path}/assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test');

      if (await assetsDir.exists()) {
        final folders = await assetsDir.list().toList();
        final plantNames = folders
            .whereType<Directory>()
            .map((d) => d.path.split(Platform.pathSeparator).last)
            .toList()
            ..sort();

        setState(() {
          _plantFolders = plantNames;
          _isLoading = false;
        });
      } else {
        debugPrint('Assets dir not found: ${assetsDir.path}');
        setState(() => _isLoading = false);
      }
    } catch (e) {
      debugPrint('Error loading plant folders: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<void> _loadPlantImages(String plant) async {
    try {
      final appDir = Directory.current;
      final plantDir = Directory('${appDir.path}/assets/PhytoLeaf-14-20260721T201046Z-1-001/PhytoLeaf-14/test/$plant');

      if (await plantDir.exists()) {
        final files = await plantDir.list().toList();
        final images = files
            .whereType<File>()
            .where((f) => f.path.endsWith('.jpg') || f.path.endsWith('.png'))
            .toList()
            ..sort((a, b) => a.path.compareTo(b.path));

        setState(() {
          _selectedPlant = plant;
          _plantImages = images;
        });
      }
    } catch (e) {
      debugPrint('Error loading images: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_selectedPlant == null) {
      // Show plant list
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              'Select a Plant Category',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 1,
                mainAxisSpacing: 8,
                crossAxisSpacing: 8,
              ),
              itemCount: _plantFolders.length,
              itemBuilder: (context, index) {
                final plant = _plantFolders[index];
                return GestureDetector(
                  onTap: () => _loadPlantImages(plant),
                  child: Card(
                    elevation: 2,
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8),
                        color: Colors.green.shade50,
                      ),
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.spa, size: 40, color: Colors.green.shade700),
                            const SizedBox(height: 8),
                            Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Text(
                                plant.replaceAll('_', ' '),
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.green.shade700,
                                ),
                              ),
                            ),
                          ],
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
    } else {
      // Show images for selected plant
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => setState(() => _selectedPlant = null),
                ),
                Expanded(
                  child: Text(
                    _selectedPlant!.replaceAll('_', ' '),
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                mainAxisSpacing: 8,
                crossAxisSpacing: 8,
              ),
              itemCount: _plantImages.length,
              itemBuilder: (context, index) {
                final image = _plantImages[index];
                return GestureDetector(
                  onTap: () {
                    widget.onImageSelected(image);
                    Navigator.pop(context);
                  },
                  child: Card(
                    elevation: 1,
                    child: Image.file(
                      image,
                      fit: BoxFit.cover,
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
}
