// env vars helper
// note: api base is read from --dart-define at build/run time. handy for switching between local/emulator/prod.
class Env {
  // fallback points to android emulator host (10.0.2.2), tweak as needed
  static const apiBase = String.fromEnvironment(
    'API_BASE',
    defaultValue: 'http://10.0.2.2:8000',
  );
}
