import 'package:flutter/material.dart';

/// centralized brand colors and theme builders (indrive-like, slightly tweaked)
class BrandColors {
  // accent green (logo-ish)
  static const accentGreen = Color(0xFFC1F11D); // c1f11d
  // secondary accent gray
  static const accentGray = Color(0xFF939393); // 939393

  // bg colors for both themes
  static const darkBg = Color(0xFF141414); // true-ish black
  static const lightBg = Color(0xFFFFFEE9); // milky white

  // build a light ColorScheme w/ milky bg + our accents
  static ColorScheme buildLightScheme() {
    final base = ColorScheme.fromSeed(seedColor: accentGreen, brightness: Brightness.light);
    return base.copyWith(
      primary: accentGreen,
      secondary: accentGray,
      background: lightBg,
      surface: lightBg,
      onPrimary: Colors.black,
      onBackground: Colors.black,
      onSurface: Colors.black,
    );
  }

  // build a dark ColorScheme w/ true black + accents
  static ColorScheme buildDarkScheme() {
    final base = ColorScheme.fromSeed(seedColor: accentGreen, brightness: Brightness.dark);
    return base.copyWith(
      primary: accentGreen,
      secondary: accentGray,
      background: darkBg,
      surface: darkBg,
      onPrimary: Colors.black, // good contrast on bright green
      onBackground: Colors.white,
      onSurface: Colors.white,
    );
  }
}
