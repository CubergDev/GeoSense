import 'package:flutter/material.dart';
import 'package:geosense_ai/widgets/indrive_logo.dart';

/// brand logo widget: tries to load raster asset, falls back to vector mark
/// place your jpg at assets/brand/unnamed.jpg (or change the path below)
class BrandLogo extends StatelessWidget {
  final double size;
  const BrandLogo({super.key, this.size = 24});

  @override
  Widget build(BuildContext context) {
    final borderRadius = BorderRadius.circular(6);
    return ClipRRect(
      borderRadius: borderRadius,
      child: Image.asset(
        'assets/brand/unnamed.jpg', // expected provided logo path (jpg)
        width: size * 1.8,
        height: size * 1.8,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stack) {
          // fallback to the vector logo if asset not packaged yet
          return Container(
            width: size * 1.8,
            height: size * 1.8,
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: InDriveLogo(size: size),
          );
        },
      ),
    );
  }
}
