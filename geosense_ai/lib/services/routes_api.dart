import 'package:dio/dio.dart';
import '../core/http.dart';

// p0 routes api per spec (only popular routes atm)
class RoutesApi {
  final Dio _dio = buildDio();

  // payload carries bbox + knobs (top_n, simplify_tolerance_m, etc.)
  Future<Response> analyzePopular(Map<String, dynamic> payload) =>
      _dio.post('/api/v1/routes/analyze/popular', data: payload);
}
