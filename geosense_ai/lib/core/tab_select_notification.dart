import 'package:flutter/widgets.dart';

// simple notification to ask parent to switch tabs
// fyi: avoids tight coupling w/ bottom nav state
class TabSelectNotification extends Notification {
  final int index; // target tab index
  const TabSelectNotification(this.index);
}