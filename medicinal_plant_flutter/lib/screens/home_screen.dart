import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/tflite_model.dart';
import '../widgets/image_preview.dart';
import '../widgets/action_buttons.dart';
import '../widgets/result_card.dart';
import '../widgets/asset_gallery.dart';
import 'dart:io';

class PlantClassifierScreen extends StatefulWidget {
  const PlantClassifierScreen({Key? key}) : super(key: key);

  @override
  State<PlantClassifierScreen> createState() => _PlantClassifierScreenState();
}

class _PlantClassifierScreenState extends State<PlantClassifierScreen> {
  final TFLiteModel _model = TFLiteModel();
  File? _selectedImage;
  Map<String, dynamic>? _prediction;
  bool _isProcessing = false;
  bool _modelLoaded = false;
  String? _errorMessage;

  final List<String> _sampleImages = [
    'Aloe', 'Neem', 'Ashoka', 'Centella', 'Hibiscus', 'Justicia',
    'Kalanchoe', 'Mint', 'Mikania', 'Moringa', 'Tulsi', 'Amla', 'Arjun', 'Weed'
  ];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      debugPrint('🔄 Initializing model...');
      debugPrint('📍 About to call _model.loadModel()...');
      await _model.loadModel();
      setState(() => _modelLoaded = true);
      debugPrint('✅ Model loaded successfully');
    } catch (e) {
      debugPrint('❌ CAUGHT EXCEPTION: $e');
      setState(() => _errorMessage = e.toString());
    }
  }

  Future<void> _runInference() async {
    if (_selectedImage == null) {
      setState(() => _errorMessage = 'Please select an image first');
      return;
    }

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });
    try {
      final result = await _model.predict(_selectedImage!);
      setState(() => _prediction = result);
    } catch (e) {
      debugPrint('Full error: $e');
      setState(() => _errorMessage = 'Inference failed: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  void _clearImage() {
    setState(() {
      _selectedImage = null;
      _prediction = null;
      _errorMessage = null;
    });
  }

  Future<void> _selectImageFromGallery() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _prediction = null;
        _errorMessage = null;
      });
    }
  }

  Future<void> _takePhotoWithCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _prediction = null;
        _errorMessage = null;
      });
    }
  }

  @override
  void dispose() {
    _model.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plant Classifier'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => Navigator.of(context).pushNamed('/about'),
            tooltip: 'About App',
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                margin: const EdgeInsets.symmetric(vertical: 8),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Row(
                    children: [
                      Icon(
                        _modelLoaded ? Icons.check_circle : Icons.hourglass_empty,
                        color: _modelLoaded ? Colors.green : Colors.orange,
                        size: 24,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          _modelLoaded ? '✅ Model Ready' : '⏳ Loading Model...',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                            color: _modelLoaded ? Colors.green : Colors.orange,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              if (_errorMessage != null)
                Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Text(
                      '❌ Error: $_errorMessage',
                      style: TextStyle(color: Colors.red.shade700, fontSize: 12),
                    ),
                  ),
                ),
              const SizedBox(height: 16),
              ImagePreview(image: _selectedImage),
              const SizedBox(height: 16),
              ActionButtons(
                onGallery: _selectImageFromGallery,
                onCamera: _takePhotoWithCamera,
                isProcessing: _isProcessing,
              ),
              const SizedBox(height: 8),
              ElevatedButton.icon(
                onPressed: () => showModalBottomSheet(
                  context: context,
                  isScrollControlled: true,
                  builder: (context) => SizedBox(
                    height: MediaQuery.of(context).size.height * 0.8,
                    child: AssetGallery(
                      onImageSelected: (file) {
                        setState(() {
                          _selectedImage = file;
                          _prediction = null;
                          _errorMessage = null;
                        });
                      },
                    ),
                  ),
                ),
                icon: const Icon(Icons.photo_library),
                label: const Text('Test Images'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade600,
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing || _selectedImage == null ? null : _runInference,
                      icon: const Icon(Icons.check),
                      label: const Text('Analyze'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  ElevatedButton.icon(
                    onPressed: _clearImage,
                    icon: const Icon(Icons.clear),
                    label: const Text('Clear'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              if (_prediction != null)
                ResultCard(
                  plantName: _prediction!['label'] as String,
                  confidence: (_prediction!['confidence'] as double) * 100,
                ),
              if (_isProcessing)
                const Center(
                  child: Padding(
                    padding: EdgeInsets.all(16.0),
                    child: CircularProgressIndicator(),
                  ),
                ),
              const SizedBox(height: 16),
              Text(
                'Sample Plants',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _sampleImages.map((plant) {
                  return Chip(
                    label: Text(plant),
                    backgroundColor: Colors.green.shade100,
                    labelStyle: TextStyle(color: Colors.green.shade800),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
