import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:geosense_ai/services/analytics_api.dart';
import 'package:geosense_ai/services/routes_api.dart';
import 'package:geosense_ai/widgets/json_view.dart';
import 'package:geosense_ai/demo/demo_data.dart';

// analytics cards: calls api and shows pretty json (or demo data)
class AnalyticsScreen extends StatefulWidget {
  final bool demoMode;
  const AnalyticsScreen({super.key, this.demoMode = false});

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  final _api = AnalyticsApi();
  final _routes = RoutesApi();

  Future<Response> _load(String key) {
    // Default bbox around Almaty: [minLng, minLat, maxLng, maxLat]
    const bbox = [76.70, 43.10, 77.10, 43.35];
    switch (key) {
      case 'demand-zones':
        return _api.getDemandZones(epsM: 250, minSamples: 10, bbox: bbox);
      case 'efficiency-metrics':
        return _api.getEfficiencyMetrics(bbox: bbox);
      case 'popular-routes':
        return _routes.analyzePopular({
          'bbox': bbox,
          'top_n': 10,
          'simplify_tolerance_m': 50,
        });
      default:
        // Fallback to efficiency metrics
        return _api.getEfficiencyMetrics(bbox: bbox);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cards = <_CardSpec>[
      _CardSpec(title: 'Зоны высокого спроса', keyName: 'demand-zones', icon: Icons.place),
      _CardSpec(title: 'Метрики эффективности', keyName: 'efficiency-metrics', icon: Icons.speed),
      _CardSpec(title: 'Популярные маршруты', keyName: 'popular-routes', icon: Icons.alt_route),
    ];

    return Scaffold(
      body: ListView.separated(
        padding: const EdgeInsets.all(12),
        itemCount: cards.length,
        separatorBuilder: (_, __) => const SizedBox(height: 12),
        itemBuilder: (context, i) {
          final c = cards[i];
          return _AnalyticsCard(spec: c, loader: () => _load(c.keyName), demoMode: widget.demoMode);
        },
      ),
    );
  }
}

class _CardSpec {
  final String title;
  final String keyName;
  final IconData icon;
  _CardSpec({required this.title, required this.keyName, required this.icon});
}

class _AnalyticsCard extends StatefulWidget {
  final _CardSpec spec;
  final Future<Response> Function() loader;
  final bool demoMode;
  const _AnalyticsCard({required this.spec, required this.loader, required this.demoMode});

  @override
  State<_AnalyticsCard> createState() => _AnalyticsCardState();
}

class _AnalyticsCardState extends State<_AnalyticsCard> {
  bool _expanded = false;
  bool _loading = false;
  dynamic _data;
  String? _error;

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    // Demo mode: provide synthetic data instantly
    if (widget.demoMode) {
      dynamic demo;
      switch (widget.spec.keyName) {
        case 'demand-zones':
          demo = DemoData.demandZonesSpec();
          break;
        case 'efficiency-metrics':
          demo = DemoData.efficiencySpec();
          break;
        case 'popular-routes':
          demo = DemoData.popularRoutesSpec();
          break;
        default:
          demo = {'ok': true};
      }
      setState(() {
        _data = demo;
        _loading = false;
      });
      return;
    }

    try {
      final res = await widget.loader();
      setState(() {
        _data = res.data;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(widget.spec.icon, color: Theme.of(context).colorScheme.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(widget.spec.title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                ),
                FilledButton.tonal(
                  onPressed: _loading ? null : _fetch,
                  child: _loading ? const SizedBox(height: 16, width: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Запросить'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (_error != null) Text('Ошибка: $_error', style: const TextStyle(color: Colors.red)),
            if (_data != null)
              AnimatedCrossFade(
                firstChild: const SizedBox.shrink(),
                secondChild: JsonView(data: _data),
                crossFadeState: _expanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
                duration: const Duration(milliseconds: 200),
              ),
            if (_data != null)
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => setState(() => _expanded = !_expanded),
                  child: Text(_expanded ? 'Свернуть' : 'Показать данные'),
                ),
              )
          ],
        ),
      ),
    );
  }
}
