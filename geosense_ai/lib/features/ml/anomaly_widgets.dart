import 'package:flutter/material.dart';

class AnomalyList extends StatelessWidget {
  final Map<String, dynamic> data;
  const AnomalyList({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final items = (data['items'] as List?)?.cast<Map<String, dynamic>>() ?? const <Map<String, dynamic>>[];
    final model = data['model'] as Map<String, dynamic>?;

    if (items.isEmpty) {
      return _EmptyState(
        title: 'Аномалий не найдено',
        subtitle: 'Попробуйте расширить область поиска или изменить фильтры',
        icon: Icons.verified,
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (model != null) _ModelHeader(model: model),
        ListView.separated(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: items.length,
          separatorBuilder: (_, __) => const SizedBox(height: 12),
          itemBuilder: (context, i) => _AnomalyCard(item: items[i]),
        ),
      ],
    );
  }
}

class _ModelHeader extends StatelessWidget {
  final Map<String, dynamic> model;
  const _ModelHeader({required this.model});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        border: Border.all(color: cs.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(Icons.insights, color: cs.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Модель: ${model['name'] ?? '—'} • v${model['version'] ?? '—'}',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnomalyCard extends StatelessWidget {
  final Map<String, dynamic> item;
  const _AnomalyCard({required this.item});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final severity = (item['severity'] as String?) ?? 'medium';
    final score = (item['score'] is num) ? (item['score'] as num).toDouble() : (item['score'] is String ? double.tryParse(item['score']) ?? 0.0 : 0.0);
    final flags = (item['flags'] as List?)?.cast<String>() ?? const <String>[];
    final tripId = item['trip_id']?.toString();
    final why = (item['why'] as Map?)?.cast<String, dynamic>();
    final actionSuggestion = item['action_suggestion']?.toString();
    final policyViolation = item['policy_violation'] == true;

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(policyViolation ? Icons.report : Icons.warning_amber_outlined, color: policyViolation ? cs.error : cs.tertiary),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          if (tripId != null) Expanded(child: Text('Поездка $tripId', style: const TextStyle(fontWeight: FontWeight.w600))),
                          SeverityChip(severity: severity),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          const Text('Аномальность:'),
                          const SizedBox(width: 8),
                          Expanded(child: LinearProgressIndicator(value: score.clamp(0, 1), minHeight: 6)),
                          const SizedBox(width: 8),
                          Text('${(score * 100).toStringAsFixed(0)}%'),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
            if (flags.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [for (final f in flags) Chip(label: Text(_prettyFlag(f)))],
              ),
            ],
            if (why != null) ...[
              const SizedBox(height: 8),
              _Reasons(why: why),
            ],
            if (actionSuggestion != null) ...[
              const SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: cs.surfaceContainerHigh,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: cs.outlineVariant),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.lightbulb, color: cs.primary),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('Рекомендация', style: TextStyle(fontWeight: FontWeight.w600)),
                          const SizedBox(height: 4),
                          Text(actionSuggestion),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _prettyFlag(String f) {
    switch (f) {
      case 'overspeed':
        return 'Превышение скорости';
      case 'high_detour':
        return 'Высокий крюк';
      default:
        return f;
    }
  }
}

class SeverityChip extends StatelessWidget {
  final String severity; // low/medium/high
  const SeverityChip({super.key, required this.severity});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    Color bg;
    Color fg;
    String label;
    switch (severity) {
      case 'low':
        bg = cs.secondaryContainer;
        fg = cs.onSecondaryContainer;
        label = 'Низкая';
        break;
      case 'high':
        bg = cs.errorContainer;
        fg = cs.onErrorContainer;
        label = 'Высокая';
        break;
      default:
        bg = cs.tertiaryContainer;
        fg = cs.onTertiaryContainer;
        label = 'Средняя';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
      child: Text(label, style: TextStyle(color: fg, fontWeight: FontWeight.w600)),
    );
  }
}

class _Reasons extends StatelessWidget {
  final Map<String, dynamic> why;
  const _Reasons({required this.why});
  @override
  Widget build(BuildContext context) {
    final top = (why['top_reasons'] as List?)?.cast<Map<String, dynamic>>() ?? const <Map<String, dynamic>>[];
    if (top.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Почему это аномалия', style: TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        ...top.map((e) => _ReasonBar(name: _prettyFeature(e['feature']), value: (e['contrib'] as num?)?.toDouble() ?? 0.0)),
      ],
    );
  }

  String _prettyFeature(dynamic key) {
    switch (key) {
      case 'detour_ratio':
        return 'Коэф. крюка';
      case 'p95_spd_kmh':
        return 'P95 скорость, км/ч';
      case 'circ_std_azm':
        return 'Разброс азимутов';
      default:
        return key?.toString() ?? '';
    }
  }
}

class _ReasonBar extends StatelessWidget {
  final String name;
  final double value; // 0..1
  const _ReasonBar({required this.name, required this.value});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 150, child: Text(name)),
          const SizedBox(width: 8),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: value.clamp(0, 1),
                minHeight: 8,
                color: cs.primary,
                backgroundColor: cs.surfaceContainerHighest,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(width: 36, child: Text('${(value * 100).toStringAsFixed(0)}%', textAlign: TextAlign.right)),
        ],
      ),
    );
  }
}

class ModelStatusList extends StatelessWidget {
  final Map<String, dynamic> data;
  const ModelStatusList({super.key, required this.data});
  @override
  Widget build(BuildContext context) {
    final models = (data['models'] as List?)?.cast<Map<String, dynamic>>() ?? const <Map<String, dynamic>>[];
    if (models.isEmpty) {
      return const _EmptyState(
        title: 'Нет данных о моделях',
        subtitle: 'Сервис не вернул список моделей',
        icon: Icons.info_outline,
      );
    }
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Статус моделей', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            ...models.map((m) => _ModelTile(model: m)),
          ],
        ),
      ),
    );
  }
}

class _ModelTile extends StatelessWidget {
  final Map<String, dynamic> model;
  const _ModelTile({required this.model});
  @override
  Widget build(BuildContext context) {
    final name = model['name']?.toString() ?? '—';
    final version = model['version']?.toString();
    final ready = model['ready'] == true || model['status'] == 'healthy';
    final trainedAt = model['trained_at']?.toString();
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(horizontal: 8),
      leading: CircleAvatar(
        child: Icon(ready ? Icons.check : Icons.sync_problem),
      ),
      title: Text(name, style: const TextStyle(fontWeight: FontWeight.w600)),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (version != null) Text('Версия: $version'),
          if (trainedAt != null) Text('Обучена: $trainedAt'),
        ],
      ),
      trailing: _StatusPill(ok: ready),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final bool ok;
  const _StatusPill({required this.ok});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bg = ok ? cs.secondaryContainer : cs.errorContainer;
    final fg = ok ? cs.onSecondaryContainer : cs.onErrorContainer;
    final text = ok ? 'Готово' : 'Проблемы';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
      child: Text(text, style: TextStyle(color: fg, fontWeight: FontWeight.w600)),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  const _EmptyState({required this.title, required this.subtitle, required this.icon});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(icon, color: cs.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(subtitle),
              ],
            ),
          ),
        ],
      ),
    );
  }
}