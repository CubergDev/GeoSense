import 'package:dio/dio.dart';
import '../core/http.dart';

// p0 ml api (per spec). only detect-anomalies + model-status for now.
class MlApi {
  final Dio _dio = buildDio();

  // anomalies supports explanations in payload (see demo/spec)
  Future<Response> detectAnomalies(Map<String, dynamic> payload) =>
      _dio.post('/api/v1/ml/detect-anomalies', data: payload);

  // health/info about models serving
  Future<Response> modelStatus() => _dio.get('/api/v1/ml/model-status');
}
