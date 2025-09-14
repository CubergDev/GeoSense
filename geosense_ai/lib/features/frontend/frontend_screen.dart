import 'package:flutter/material.dart';
import 'package:geosense_ai/core/tab_select_notification.dart';

// native dashboard (no webview), indrive vibe
class FrontendScreen extends StatelessWidget {
  final bool demoMode;
  const FrontendScreen({super.key, this.demoMode = false});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme; // short alias to color scheme
    return Scaffold(
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _Header(cs: cs, demoMode: demoMode), // hero-ish header
          const SizedBox(height: 16),
          // quick nav cards
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: const [
              _DashCard(
                icon: Icons.map_outlined,
                title: 'Карта спроса',
                subtitle: 'Тепловая карта и «горячие зоны»',
              ),
              _DashCard(
                icon: Icons.analytics_outlined,
                title: 'Аналитика',
                subtitle: 'Паттерны, метрики, сравнения',
              ),
              _DashCard(
                icon: Icons.memory,
                title: 'ML инструменты',
                subtitle: 'Прогноз спроса, аномалии, маршруты',
              ),
            ],
          ),
          const SizedBox(height: 24),
          // quick actions
          Text('Быстрые действия', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilledButton.icon(
                onPressed: () => _selectTab(context, 0),
                icon: const Icon(Icons.map),
                label: const Text('Открыть карту'),
              ),
              FilledButton.tonalIcon(
                onPressed: () => _selectTab(context, 1),
                icon: const Icon(Icons.query_stats),
                label: const Text('Запросить аналитику'),
              ),
              OutlinedButton.icon(
                onPressed: () => _selectTab(context, 2),
                icon: const Icon(Icons.psychology),
                label: const Text('ML панель'),
              ),
            ],
          ),
          const SizedBox(height: 24),
          _PrivacyNote(color: cs.primary),
        ],
      ),
    );
  }

  void _selectTab(BuildContext context, int index) {
    // ask parent to switch tab via notification (loose coupling ftw)
    TabSelectNotification(index).dispatch(context);
  }
}

class _Header extends StatelessWidget {
  final ColorScheme cs;
  final bool demoMode;
  const _Header({required this.cs, required this.demoMode});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [cs.primary.withOpacity(0.15), cs.primary.withOpacity(0.05)]),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: cs.primary.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          ClipOval(
            child: Image.asset(
              'lib/images/unnamed.jpg',
              width: 52,
              height: 52,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'GeoSense by inDrive',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 4),
                Text(
                  'Аналитика спроса, безопасность и ML на обезличенных геотреках',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DashCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  const _DashCard({required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return SizedBox(
      width: 360,
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              Icon(icon, size: 36, color: cs.primary),
              const SizedBox(width: 12),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text(subtitle, style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)), 
                ]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PrivacyNote extends StatelessWidget {
  final Color color;
  const _PrivacyNote({required this.color});
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.privacy_tip, color: color),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'Приватность: используются агрегированные и анонимизированные данные. Личные данные пассажиров/водителей недоступны и не собираются.',
            ),
          ),
        ],
      ),
    );
  }
}
