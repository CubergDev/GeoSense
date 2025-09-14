import 'package:dio/dio.dart';
import 'env.dart';

// tiny http client factory
// btw: we keep timeouts modest to avoid hanging ui; tune per backend perf.
Dio buildDio() {
  final dio = Dio(BaseOptions(
    baseUrl: Env.apiBase,
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(seconds: 20),
  ));
  // log req bodies for easier dev/debug; resp bodies off to cut noise
  dio.interceptors.add(LogInterceptor(requestBody: true, responseBody: false));
  return dio;
}
