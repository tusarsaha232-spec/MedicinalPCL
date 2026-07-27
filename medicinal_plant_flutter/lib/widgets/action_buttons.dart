import 'package:flutter/material.dart';

class ActionButtons extends StatelessWidget {
  final VoidCallback onGallery;
  final VoidCallback onCamera;
  final bool isProcessing;

  const ActionButtons({
    Key? key,
    required this.onGallery,
    required this.onCamera,
    required this.isProcessing,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: isProcessing ? null : onGallery,
            icon: const Icon(Icons.image),
            label: const Text('Gallery'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: isProcessing ? null : onCamera,
            icon: const Icon(Icons.camera_alt),
            label: const Text('Camera'),
          ),
        ),
      ],
    );
  }
}
