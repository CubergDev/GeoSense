import 'package:flutter/material.dart';
import 'package:geosense_ai/core/tab_select_notification.dart';
import 'package:geosense_ai/features/frontend/frontend_screen.dart';
import 'package:geosense_ai/features/map/map_screen.dart';
import 'package:geosense_ai/features/analytics/analytics_screen.dart';
import 'package:geosense_ai/features/ml/ml_screen.dart';
import 'core/brand.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data' show ByteData;
import 'package:google_fonts/google_fonts.dart';

// app entrypoint
void main() => runApp(const App());

class App extends StatefulWidget {
  const App({super.key});
  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  // keep current theme mode in state (light by default)
  ThemeMode _mode = ThemeMode.light; // default to milky light theme

  // setter used by toggle in appbar
  void _setTheme(ThemeMode mode) => setState(() => _mode = mode);

  @override
  Widget build(BuildContext context) {
    final lightScheme = BrandColors.buildLightScheme();
    final darkScheme = BrandColors.buildDarkScheme();

    // Base themes
    final baseLight = ThemeData(
      colorScheme: lightScheme,
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: lightScheme.background,
      appBarTheme: AppBarTheme(backgroundColor: lightScheme.surface, foregroundColor: lightScheme.onSurface),
      floatingActionButtonTheme: FloatingActionButtonThemeData(backgroundColor: lightScheme.primary, foregroundColor: lightScheme.onPrimary),
      filledButtonTheme: FilledButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStatePropertyAll(lightScheme.primary),
          foregroundColor: WidgetStatePropertyAll(lightScheme.onPrimary),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStatePropertyAll(lightScheme.onSurface),
          side: WidgetStatePropertyAll(BorderSide(color: lightScheme.outline)),
        ),
      ),
      chipTheme: ChipThemeData(
        shape: StadiumBorder(side: BorderSide(color: lightScheme.outlineVariant)),
        labelStyle: TextStyle(color: lightScheme.onSurfaceVariant),
        backgroundColor: lightScheme.surfaceContainerHighest,
      ),
    );

    final baseDark = ThemeData(
      colorScheme: darkScheme,
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: darkScheme.background,
      appBarTheme: AppBarTheme(backgroundColor: darkScheme.surface, foregroundColor: darkScheme.onSurface),
      floatingActionButtonTheme: FloatingActionButtonThemeData(backgroundColor: darkScheme.primary, foregroundColor: darkScheme.onPrimary),
      filledButtonTheme: FilledButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStatePropertyAll(darkScheme.primary),
          foregroundColor: WidgetStatePropertyAll(darkScheme.onPrimary),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStatePropertyAll(darkScheme.onSurface),
          side: WidgetStatePropertyAll(BorderSide(color: darkScheme.outline)),
        ),
      ),
      chipTheme: ChipThemeData(
        shape: StadiumBorder(side: BorderSide(color: darkScheme.outlineVariant)),
        labelStyle: TextStyle(color: darkScheme.onSurfaceVariant),
        backgroundColor: darkScheme.surfaceContainerHighest,
      ),
    );

    // Apply Inter font across the app for both themes
    final lightText = GoogleFonts.interTextTheme(baseLight.textTheme).apply(
      bodyColor: lightScheme.onSurface,
      displayColor: lightScheme.onSurface,
    );
    final darkText = GoogleFonts.interTextTheme(baseDark.textTheme).apply(
      bodyColor: darkScheme.onSurface,
      displayColor: darkScheme.onSurface,
    );

    final lightTheme = baseLight.copyWith(textTheme: lightText);
    final darkTheme = baseDark.copyWith(textTheme: darkText);

    return MaterialApp(
      title: 'GeoSense by inDrive',
      theme: lightTheme,
      darkTheme: darkTheme,
      themeMode: _mode,
      home: HomeScreen(
        onChangeTheme: _setTheme,
        themeMode: _mode,
      ),
    );
  }
}

class HomeScreen extends StatefulWidget {
  final void Function(ThemeMode mode) onChangeTheme;
  final ThemeMode themeMode;
  const HomeScreen({super.key, required this.onChangeTheme, required this.themeMode});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int idx = 0;
  bool demoMode = true; // default to demo while backend is not connected

  @override
  Widget build(BuildContext context) {
    final pages = [
      MapScreen(demoMode: demoMode),
      AnalyticsScreen(demoMode: demoMode),
      MlScreen(demoMode: demoMode),
      FrontendScreen(demoMode: demoMode),
    ];
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        leading: Padding(
          padding: const EdgeInsets.only(left: 8.0),
          child: _LogoImage(size: 36),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.bolt, size: 18, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 6),
            const Text('Демо'),
            const SizedBox(width: 6),
            Switch.adaptive(
              value: demoMode,
              onChanged: (v) => setState(() => demoMode = v),
            ),
          ],
        ),
        actions: [
          // Theme switcher: Black vs Milky (icons only)
          Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: SegmentedButton<ThemeMode>(
              segments: const [
                ButtonSegment(value: ThemeMode.dark, icon: Icon(Icons.dark_mode)),
                ButtonSegment(value: ThemeMode.light, icon: Icon(Icons.light_mode)),
              ],
              selected: {widget.themeMode},
              onSelectionChanged: (s) => widget.onChangeTheme(s.first),
              showSelectedIcon: false,
              multiSelectionEnabled: false,
            ),
          ),
        ],
      ),
      body: NotificationListener<TabSelectNotification>(
        onNotification: (n) {
          setState(() => idx = n.index);
          return true;
        },
        child: pages[idx],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: idx,
        onDestinationSelected: (i) => setState(() => idx = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.map_outlined), label: 'Карта'),
          NavigationDestination(icon: Icon(Icons.analytics_outlined), label: 'Аналитика'),
          NavigationDestination(icon: Icon(Icons.memory), label: 'ML'),
          NavigationDestination(icon: Icon(Icons.dashboard_outlined), label: 'Дашборд'),
        ],
      ),
    );
  }
}

class _DemoBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(12),
      color: cs.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.lightbulb, color: cs.onPrimaryContainer),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                'Демо-режим: данные сгенерированы локально для презентации',
                style: TextStyle(color: cs.onPrimaryContainer),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class PlaceholderWidget extends StatelessWidget {
  final String title;
  const PlaceholderWidget(this.title, {super.key});
  @override
  Widget build(BuildContext context) {
    return Center(child: Text(title, style: const TextStyle(fontSize: 24)));
  }
}

class _LogoImage extends StatelessWidget {
  final double size;
  const _LogoImage({required this.size});
  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ByteData>(
      future: rootBundle.load('lib/images/unnamed.jpg'),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          final bytes = snapshot.data!.buffer.asUint8List();
          return ClipOval(
            child: Image.memory(bytes, width: size, height: size, fit: BoxFit.cover),
          );
        }
        // Fallback: reserved space without custom drawing
        return SizedBox(width: size, height: size);
      },
    );
  }
}
