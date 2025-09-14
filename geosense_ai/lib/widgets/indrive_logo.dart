import 'package:flutter/material.dart';

/// simple indrive-like logo that adapts to theme
/// left: green rounded square with "in"; right: "Drive" wordmark
class InDriveLogo extends StatelessWidget {
  final double size; // size of the green square
  const InDriveLogo({super.key, this.size = 28});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            color: cs.primary,
            borderRadius: BorderRadius.circular(6),
          ),
          alignment: Alignment.center,
          child: Text(
            'in',
            style: TextStyle(
              color: cs.onPrimary,
              fontWeight: FontWeight.w800,
              fontSize: size * 0.46,
              letterSpacing: -0.5,
            ),
          ),
        ),
        const SizedBox(width: 6),
        Text(
          'Drive',
          style: TextStyle(
            color: cs.onSurface,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }
}
