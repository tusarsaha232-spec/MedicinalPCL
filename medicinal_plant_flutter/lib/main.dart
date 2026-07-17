import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const MedicinalPlantApp());
}

class MedicinalPlantApp extends StatelessWidget {
  const MedicinalPlantApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medicinal Plant Classifier',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
        ),
      ),
      home: const PlantClassifierScreen(),
    );
  }
}
