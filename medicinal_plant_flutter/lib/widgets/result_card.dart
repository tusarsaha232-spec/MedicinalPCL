import 'package:flutter/material.dart';

class ResultCard extends StatelessWidget {
  final String plantName;
  final String scientificName;
  final double confidence;
  final DateTime? predictionTime;

  const ResultCard({
    Key? key,
    required this.plantName,
    required this.scientificName,
    required this.confidence,
    this.predictionTime,
  }) : super(key: key);

  Color _confidenceColor() {
    if (confidence >= 80) return const Color(0xFF16804A);
    if (confidence >= 60) return const Color(0xFFB7791F);
    return const Color(0xFFC2413A);
  }

  String _confidenceLabel() {
    if (confidence >= 80) return 'Strong match';
    if (confidence >= 60) return 'Moderate match';
    return 'Low match';
  }

  String _formatTime(DateTime? dateTime) {
    if (dateTime == null) return '';
    final hour = dateTime.hour.toString().padLeft(2, '0');
    final minute = dateTime.minute.toString().padLeft(2, '0');
    final second = dateTime.second.toString().padLeft(2, '0');
    return '$hour:$minute:$second';
  }

  @override
  Widget build(BuildContext context) {
    final confidenceColor = _confidenceColor();

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFD7E7DA)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: const Color(0xFFE7F5EC),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.analytics_outlined, color: Color(0xFF1F7A4D)),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Classification Report',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w800,
                              color: const Color(0xFF183A2A),
                            ),
                      ),
                      Text(
                        'CNN inference result',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: const Color(0xFF5E7467),
                              fontWeight: FontWeight.w500,
                            ),
                      ),
                    ],
                  ),
                ),
                if (predictionTime != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF3F6F4),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.schedule, size: 15, color: Color(0xFF607469)),
                        const SizedBox(width: 5),
                        Text(
                          _formatTime(predictionTime),
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFF40564A),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 18),
            _ResultField(
              icon: Icons.local_florist_outlined,
              label: 'Predicted Class',
              value: plantName,
              accentColor: const Color(0xFF1F7A4D),
            ),
            const SizedBox(height: 10),
            _ResultField(
              icon: Icons.science_outlined,
              label: 'Botanical Label',
              value: scientificName,
              accentColor: const Color(0xFF6F4E9B),
              italicValue: true,
            ),
            const SizedBox(height: 18),
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Confidence Score',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF5C7065),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${confidence.toStringAsFixed(1)}%',
                        style: TextStyle(
                          fontSize: 30,
                          fontWeight: FontWeight.w900,
                          color: confidenceColor,
                          height: 1,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
                  decoration: BoxDecoration(
                    color: confidenceColor.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    _confidenceLabel(),
                    style: TextStyle(
                      color: confidenceColor,
                      fontWeight: FontWeight.w800,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: (confidence.clamp(0, 100) as num).toDouble() / 100,
                minHeight: 12,
                backgroundColor: const Color(0xFFE4ECE7),
                valueColor: AlwaysStoppedAnimation<Color>(confidenceColor),
              ),
            ),
            if (predictionTime != null) ...[
              const SizedBox(height: 12),
              Text(
                'Evaluated at ${_formatTime(predictionTime)} on this device/session.',
                style: const TextStyle(
                  color: Color(0xFF6B7E73),
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultField extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color accentColor;
  final bool italicValue;

  const _ResultField({
    required this.icon,
    required this.label,
    required this.value,
    required this.accentColor,
    this.italicValue = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8FAF8),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE1EAE4)),
      ),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: accentColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(9),
            ),
            child: Icon(icon, size: 19, color: accentColor),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    color: Color(0xFF6A7C71),
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  value,
                  style: TextStyle(
                    color: const Color(0xFF1B3426),
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                    fontStyle: italicValue ? FontStyle.italic : FontStyle.normal,
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
