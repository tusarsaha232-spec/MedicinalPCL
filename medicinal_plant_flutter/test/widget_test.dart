import 'package:flutter_test/flutter_test.dart';
import 'package:medicinal_plant_flutter/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const MedicinalPlantApp());
    expect(find.text('Medicinal Plant Classifier'), findsOneWidget);
  });
}
