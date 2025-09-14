import 'dart:math';

import 'package:latlong2/latlong.dart';
import 'package:flutter_map_heatmap/flutter_map_heatmap.dart';

class DemoData {
  static final _rnd = Random(42);

  // Heatmap demo points around Almaty
  static List<WeightedLatLng> heatPoints({int count = 240}) {
    final List<WeightedLatLng> pts = [];
    for (int i = 0; i < count; i++) {
      final lat = 43.238949 + (_rnd.nextDouble() - .5) * 0.28;
      final lon = 76.889709 + (_rnd.nextDouble() - .5) * 0.28;
      pts.add(WeightedLatLng(LatLng(lat, lon), _rnd.nextDouble()));
    }
    return pts;
  }

  // Demand zone demo markers (centers only, counts as strings)
  static List<Map<String, dynamic>> demandZones() {
    return [
      {
        'lat': 43.25654,
        'lon': 76.92848,
        'count': 42,
        'label': 'Центр',
      },
      {
        'lat': 43.2310,
        'lon': 76.9450,
        'count': 27,
        'label': 'Вокзал',
      },
      {
        'lat': 43.2178,
        'lon': 76.8721,
        'count': 33,
        'label': 'БЦ',
      },
    ];
  }

  // Analytics demo payloads (simplified synthetic structures)
  static Map<String, dynamic> tripPatterns() => {
        'daily_peaks': [
          {'hour': 8, 'intensity': 0.72},
          {'hour': 18, 'intensity': 0.95},
        ],
        'week_trend': [0.6, 0.7, 0.68, 0.73, 0.9, 1.0, 0.75],
      };

  static List<Map<String, dynamic>> zones() => demandZones();

  static Map<String, dynamic> safety() => {
        'anomalies': [
          {
            'type': 'route_deviation',
            'severity': 'medium',
            'location': {'lat': 43.245, 'lon': 76.91}
          }
        ]
      };

  static Map<String, dynamic> efficiency() => {
        'route_efficiency': 0.86,
        'time_efficiency': 0.78,
        'speed_consistency': 0.81,
        'fuel_estimate': {'avg_l_per_100km': 8.2}
      };

  static Map<String, dynamic> comparative() => {
        'period_a': {'rides': 1240, 'avg_eta': 7.1},
        'period_b': {'rides': 1378, 'avg_eta': 6.6},
        'delta': {'rides': '+11.1%', 'avg_eta': '-7.0%'}
      };

  // ML demo responses
  static Map<String, dynamic> predictDemand({required double lat, required double lon, required int hour}) => {
        'input': {'lat': lat, 'lon': lon, 'hour': hour, 'weekday': DateTime.now().weekday},
        'demand_score': (0.4 + _rnd.nextDouble() * 0.6).toStringAsFixed(2),
        'label': hour >= 17 && hour <= 21 ? 'Пик спроса' : 'Нормальный'
      };

  static Map<String, dynamic> anomalies() => {
        'found': true,
        'items': [
          {
            'type': 'speed_pattern',
            'score': (0.7 + _rnd.nextDouble() * 0.3).toStringAsFixed(2),
            'note': 'Нестабильная скорость на участке Абай — Наурызбай батыр'
          }
        ]
      };

  static Map<String, dynamic> optimizedRoute() => {
        'objective': 'time',
        'distance_km': (5 + _rnd.nextDouble() * 5).toStringAsFixed(2),
        'eta_min': (12 + _rnd.nextDouble() * 8).toStringAsFixed(1),
        'order': [0, 2, 1]
      };

  static Map<String, dynamic> modelStatus() => {
        'models': [
          {'name': 'demand_rf', 'status': 'healthy'},
          {'name': 'anomaly_iforest', 'status': 'healthy'},
        ],
        'last_retrain': DateTime.now().toIso8601String()
      };

  static Map<String, dynamic> retrain() => {
        'ok': true,
        'message': 'Переобучение поставлено в очередь',
      };


  // ----- Spec-compliant demo outputs (for P0) -----
  static Map<String, dynamic> _meta() => {
        'generated_at': DateTime.now().toUtc().toIso8601String(),
        'data_version': 'trips_csv_v1',
        'query': {},
        'privacy': {'k_anon': 5, 'epsilon': null, 'suppressed': 0},
        'speed_unit': 'km/h',
        'unit_detection_method': 'auto-p95',
        'timings_ms': {'compute': 120, 'read_cache': 20}
      };

  static Map<String, dynamic> demandZonesSpec() => {
        'clusters': [
          {
            'cluster_id': 1,
            'center': {'lat': 43.25654, 'lng': 76.92848},
            'radius_p95_m': 180,
            'count': 63,
            'density_km2': 1120.5,
            'demand_score': 0.87
          },
          {
            'cluster_id': 2,
            'center': {'lat': 43.2310, 'lng': 76.9450},
            'radius_p95_m': 140,
            'count': 27,
            'density_km2': 820.3,
            'demand_score': 0.54
          },
        ],
        'suppressed_below_k': 5,
        'meta': _meta(),
      };

  static Map<String, dynamic> efficiencySpec() => {
        'scope': {
          'type': 'bbox',
          'value': [76.70, 43.10, 77.10, 43.35]
        },
        'items': [],
        'distributions': {
          'detour_ratio': {
            'bins': [1.0, 1.1, 1.2, 1.4, 1.6, 2.0],
            'counts': [12, 45, 60, 31, 9, 3]
          },
          'p95_spd_kmh': {
            'bins': [20, 40, 60, 80, 100, 120, 140],
            'counts': [5, 22, 48, 44, 19, 4, 1]
          }
        },
        'speed_unit': 'km/h',
        'unit_detection_method': 'auto-p95',
        'meta': _meta(),
      };

  static Map<String, dynamic> popularRoutesSpec() => {
        'corridors': [
          {
            'corridor_id': 'c1',
            'polyline': '_ibE_seK...',
            'trips': 87,
            'width_p90_m': 95,
            'median_speed_kmh': 41.2,
            'median_detour_ratio': 1.18,
            'confidence': 0.81
          },
          {
            'corridor_id': 'c2',
            'polyline': 'a~l~Fjk~uOwHJy@P',
            'trips': 55,
            'width_p90_m': 80,
            'median_speed_kmh': 38.4,
            'median_detour_ratio': 1.12,
            'confidence': 0.74
          }
        ],
        'suppressed_below_k': 5,
        'meta': _meta(),
      };

  static Map<String, dynamic> anomaliesSpec() => {
        'items': [
          {
            'trip_id': '3f5c19d8',
            'score': 0.97,
            'severity': 'high',
            'flags': ['overspeed', 'high_detour'],
            'why': {
              'detour_ratio': 1.85,
              'p95_spd_kmh': 132.4,
              'circ_std_azm': 18.7,
              'top_reasons': [
                {'feature': 'detour_ratio', 'contrib': 0.46},
                {'feature': 'p95_spd_kmh', 'contrib': 0.39}
              ]
            },
            'policy_violation': true,
            'action_suggestion': 'Flag for review; show driver guidance on safe routing.'
          }
        ],
        'model': {'name': 'iforest_anomaly', 'version': '2025-09-14'},
        'meta': _meta(),
      };

  static Map<String, dynamic> modelStatusSpec() => {
        'models': [
          {
            'name': 'iforest_anomaly',
            'ready': true,
            'version': '2025-09-14T10:05:33Z',
            'trained_at': '2025-09-14T10:05:33Z',
            'artifact_path': 'models/anom_iforest.joblib',
            'artifact_size_mb': 3.1,
            'artifact_sha256': 'c1f0deadbeefab9',
            'features': [
              'path_len_km',
              'straight_len_km',
              'detour_ratio',
              'mean_spd_kmh',
              'p95_spd_kmh',
              'std_spd_kmh',
              'circ_std_azm'
            ],
            'training_data': {
              'source': 'trip_features.parquet',
              'rows': 18342,
              'time_window': null
            }
          }
        ]
      };


}
