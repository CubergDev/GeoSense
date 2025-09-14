import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:geosense_ai/main.dart';

void main() {
  testWidgets('App builds and shows navigation bar', (tester) async {
    await tester.pumpWidget(const App());
    // Smoke test: main scaffold with bottom NavigationBar appears
    expect(find.byType(NavigationBar), findsOneWidget);
  });
}
