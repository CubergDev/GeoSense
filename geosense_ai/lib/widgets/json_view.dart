import 'dart:convert';

import 'package:flutter/material.dart';

// tiny pretty-json box (selectable)
class JsonView extends StatelessWidget {
  final dynamic data; // any json-serializable thing
  const JsonView({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final pretty = const JsonEncoder.withIndent('  ').convert(data);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: SelectableText(
        pretty,
        style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
      ),
    );
  }
}
