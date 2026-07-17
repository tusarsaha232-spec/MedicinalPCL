import 'package:flutter/material.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Column(
                  children: [
                    Icon(Icons.nature_people, size: 80, color: Colors.green.shade600),
                    const SizedBox(height: 16),
                    Text(
                      'Medicinal Plant Classifier',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Version 1.0.0',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: Colors.grey,
                          ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 32),
              Text(
                'About This App',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 12),
              Text(
                'This application uses advanced AI and machine learning to identify medicinal plants from photos. Simply take a picture or upload an image of a plant leaf, and the app will identify the plant and provide information about its medicinal properties.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              Text(
                'Supported Plants',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 12),
              _buildPlantList(context),
              const SizedBox(height: 24),
              Text(
                'Technology',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 12),
              _buildTechInfo(context),
              const SizedBox(height: 32),
              Center(
                child: Text(
                  'Built with ❤️ using Flutter & TensorFlow Lite',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey,
                      ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPlantList(BuildContext context) {
    final plants = [
      'Aloe', 'Amla', 'Arjun', 'Ashoka', 'Bael', 'Guava',
      'Jamun', 'Mint', 'Neem', 'Pepper', 'Tulsi', 'Hibiscus', 'Curry', 'Weed'
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: plants.map((plant) {
        return Chip(
          label: Text(plant),
          backgroundColor: Colors.green.shade100,
          labelStyle: TextStyle(color: Colors.green.shade800),
        );
      }).toList(),
    );
  }

  Widget _buildTechInfo(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildTechItem('Framework', 'Flutter'),
        _buildTechItem('ML Model', 'TensorFlow Lite'),
        _buildTechItem('Model Type', 'VECTVMixer CNN'),
        _buildTechItem('Input Size', '224×224 RGB'),
        _buildTechItem('Classes', '14 medicinal plants'),
      ],
    );
  }

  Widget _buildTechItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value, style: const TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}
