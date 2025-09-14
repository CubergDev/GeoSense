import 'package:dio/dio.dart';
import '../core/http.dart';

// basic crud for trips (not fully used in p0 ui but handy for tools/tests)
class TripsApi {
  final Dio _dio = buildDio();

  Future<Response> listTrips({Map<String, dynamic>? query}) => _dio.get('/api/v1/trips', queryParameters: query);
  Future<Response> createTrip(Map<String, dynamic> payload) => _dio.post('/api/v1/trips', data: payload);
  Future<Response> getTrip(String id) => _dio.get('/api/v1/trips/$id');
  Future<Response> updateTrip(String id, Map<String, dynamic> payload) => _dio.put('/api/v1/trips/$id', data: payload);
  Future<Response> deleteTrip(String id) => _dio.delete('/api/v1/trips/$id');
  Future<Response> searchInArea(Map<String, dynamic> payload) => _dio.post('/api/v1/trips/search-area', data: payload);
}