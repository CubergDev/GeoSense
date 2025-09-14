import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_heatmap/flutter_map_heatmap.dart';
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import 'package:latlong2/latlong.dart';
import 'package:geosense_ai/services/analytics_api.dart';
import 'package:geosense_ai/demo/demo_data.dart';

// map + heatmap + demand clusters ui
class MapScreen extends StatefulWidget {
  final bool demoMode;
  const MapScreen({super.key, this.demoMode = false});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  final _mapController = MapController();
  final _api = AnalyticsApi();

  List<WeightedLatLng> _heatPoints = [];
  List<Marker> _demandMarkers = [];
  bool _loading = true;
  String? _error;

  // Default bbox around Almaty area [minLng, minLat, maxLng, maxLat]
  static const List<double> _defaultBbox = [76.70, 43.10, 77.10, 43.35];

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    // Pure demo mode: skip network and generate locally
    if (widget.demoMode) {
      final demoPoints = DemoData.heatPoints();
      final specZones = DemoData.demandZonesSpec();
      final markers = <Marker>[];
      if (specZones is Map && specZones['clusters'] is List) {
        for (final z in (specZones['clusters'] as List)) {
          final center = z['center'];
          final lat = (center?['lat'] as num?)?.toDouble();
          final lon = (center?['lng'] as num?)?.toDouble();
          final count = (z['count'] ?? z['demand_score'] ?? 0).toString();
          if (lat != null && lon != null) {
            markers.add(Marker(
              point: LatLng(lat, lon),
              width: 44,
              height: 44,
              child: _DemandBadge(count: count),
            ));
          }
        }
      }
      setState(() {
        _heatPoints = demoPoints;
        _demandMarkers = markers;
        _loading = false;
      });
      return;
    }

    try {
      // Heatmap (spec-compliant payload and parsing)
      final heatPayload = {
        'bbox': _defaultBbox,
        'time_window': null,
        'grid': {'type': 'h3', 'level': 8},
        'metrics': ['points', 'unique_ids', 'avg_speed_kmh']
      };
      final heatRes = await _api.getHeatmap(heatPayload);
      final List<WeightedLatLng> points = [];
      final heat = heatRes.data;
      if (heat is Map && heat['cells'] is List) {
        for (final c in (heat['cells'] as List)) {
          final centroid = c['centroid'];
          final lat = (centroid?['lat'] as num?)?.toDouble();
          final lon = (centroid?['lng'] as num?)?.toDouble();
          final w = (c['intensity'] as num?)?.toDouble() ?? 0.0;
          if (lat != null && lon != null) {
            points.add(WeightedLatLng(LatLng(lat, lon), w));
          }
        }
      }

      // Demand zones (clusters) per spec
      final zonesRes = await _api.getDemandZones(epsM: 250, minSamples: 10, bbox: _defaultBbox);
      final markers = <Marker>[];
      final zones = zonesRes.data;
      if (zones is Map && zones['clusters'] is List) {
        for (final z in (zones['clusters'] as List)) {
          final center = z['center'];
          final lat = (center?['lat'] as num?)?.toDouble();
          final lon = (center?['lng'] as num?)?.toDouble();
          final count = (z['count'] ?? z['demand_score'] ?? 0).toString();
          if (lat != null && lon != null) {
            markers.add(Marker(
              point: LatLng(lat, lon),
              width: 44,
              height: 44,
              child: _DemandBadge(count: count),
            ));
          }
        }
      }

      // Fallback demo if backend empty
      if (points.isEmpty && markers.isEmpty) {
        final rnd = Random(1);
        for (int i = 0; i < 200; i++) {
          final lat = 43.238949 + (rnd.nextDouble() - .5) * 0.2; // Almaty-ish
          final lon = 76.889709 + (rnd.nextDouble() - .5) * 0.2;
          points.add(WeightedLatLng(LatLng(lat, lon), rnd.nextDouble()));
        }
        markers.add(Marker(
          point: LatLng(43.25654, 76.92848),
          width: 44,
          height: 44,
          child: const _DemandBadge(count: '42'),
        ));
      }

      setState(() {
        _heatPoints = points;
        _demandMarkers = markers;
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
    return Scaffold(
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text('Ошибка: $_error'))
              : FlutterMap(
                  mapController: _mapController,
                  options: MapOptions(
                    initialCenter: const LatLng(43.238949, 76.889709),
                    initialZoom: 11,
                  ),
                  children: [
                    TileLayer(
                      urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      userAgentPackageName: 'com.example.geosense_ai',
                    ),
                    if (_heatPoints.isNotEmpty)
                      HeatMapLayer(
                        heatMapDataSource: InMemoryHeatMapDataSource(data: _heatPoints),
                        heatMapOptions: HeatMapOptions(radius: 25),
                      ),
                    if (_demandMarkers.isNotEmpty)
                      MarkerClusterLayerWidget(
                        options: MarkerClusterLayerOptions(
                          markers: _demandMarkers,
                          maxClusterRadius: 45,
                          size: const Size(44, 44),
                          builder: (context, markers) => _ClusterBadge(count: markers.length.toString()),
                        ),
                      ),
                  ],
                ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _loadData,
        icon: const Icon(Icons.insights),
        label: const Text('Обновить'),
      ),
    );
  }
}

class _DemandBadge extends StatelessWidget {
  final String count;
  const _DemandBadge({required this.count});
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      decoration: BoxDecoration(
        color: cs.primary,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.onPrimary.withOpacity(0.9), width: 1),
        boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 6, offset: Offset(0, 2))],
      ),
      alignment: Alignment.center,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Text(count, style: TextStyle(color: cs.onPrimary, fontWeight: FontWeight.bold)),
    );
  }
}

class _ClusterBadge extends StatelessWidget {
  final String count;
  const _ClusterBadge({required this.count});
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(16),
      ),
      alignment: Alignment.center,
      child: Text(count, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
    );
  }
}
