import 'package:dio/dio.dart';
import '../core/http.dart';

// thin client over analytics endpoints (p0 scope only)
class AnalyticsApi {
  final Dio _dio = buildDio();

  // post b/c payload can be kinda heavy (bbox, grid, etc.)
  Future<Response> getHeatmap(Map<String, dynamic> payload) => _dio.post('/api/v1/analytics/heatmap', data: payload);

  // zones use query params for simpler caching (cdn/proxy friendly)
  Future<Response> getDemandZones({required int epsM, required int minSamples, required List<double> bbox}) =>
      _dio.get('/api/v1/analytics/demand-zones', queryParameters: {
        'eps_m': epsM,
        'min_samples': minSamples,
        'bbox': bbox.join(',')
      });

  // optional filters; we only send non-empty
  Future<Response> getEfficiencyMetrics({List<String>? ids, List<double>? bbox}) {
    final qp = <String, dynamic>{};
    if (ids != null && ids.isNotEmpty) qp['ids'] = ids.join(',');
    if (bbox != null && bbox.length == 4) qp['bbox'] = bbox.join(',');
    return _dio.get('/api/v1/analytics/efficiency-metrics', queryParameters: qp.isEmpty ? null : qp);
  }
}
