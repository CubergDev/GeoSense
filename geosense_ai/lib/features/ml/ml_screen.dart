import 'package:flutter/material.dart';
import 'package:geosense_ai/services/ml_api.dart';
import 'package:geosense_ai/widgets/json_view.dart';
import 'package:geosense_ai/features/ml/anomaly_widgets.dart';
import 'package:geosense_ai/demo/demo_data.dart';

class MlScreen extends StatefulWidget {
  final bool demoMode;
  const MlScreen({super.key, this.demoMode = false});

  @override
  State<MlScreen> createState() => _MlScreenState();
}

class _MlScreenState extends State<MlScreen> {
  final _api = MlApi();

  // Default bbox around Almaty area [minLng, minLat, maxLng, maxLat]
  static const List<double> _defaultBbox = [76.70, 43.10, 77.10, 43.35];

  bool _loading = false;
  String? _error;
  dynamic _result;


  Future<void> _callAnomalies() async {
    await _wrap(() async {
      final payload = {
        'top_n': 100,
        'filters': {
          'bbox': _defaultBbox,
        },
        'return': {'explanations': true}
      };
      if (widget.demoMode) {
        _result = DemoData.anomaliesSpec();
      } else {
        final res = await _api.detectAnomalies(payload);
        _result = res.data;
      }
    });
  }

  Future<void> _callStatus() async {
    await _wrap(() async {
      if (widget.demoMode) {
        _result = DemoData.modelStatusSpec();
      } else {
        final res = await _api.modelStatus();
        _result = res.data;
      }
    });
  }

  Future<void> _wrap(Future<void> Function() job) async {
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });
    try {
      await job();
    } catch (e) {
      _error = e.toString();
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Wrap(spacing: 8, runSpacing: 8, children: [
            FilledButton.tonal(
              onPressed: _loading ? null : _callAnomalies,
              child: const Text('Выявить аномалии'),
            ),
            OutlinedButton(
              onPressed: _loading ? null : _callStatus,
              child: const Text('Статус моделей'),
            ),
          ]),
          const SizedBox(height: 16),
          if (_loading) const LinearProgressIndicator(),
          if (_error != null) Text('Ошибка: $_error', style: const TextStyle(color: Colors.red)),
          if (_result != null) _buildResult(context),
        ],
      ),
    );
  }

  Widget _buildResult(BuildContext context) {
    final data = _result;
    if (data is Map<String, dynamic>) {
      if (data.containsKey('items')) {
        return AnomalyList(data: data);
      }
      if (data.containsKey('models')) {
        return ModelStatusList(data: data);
      }
    }
    // Fоллбэк: показать JSON, если структура неизвестна
    return JsonView(data: _result);
  }

}
