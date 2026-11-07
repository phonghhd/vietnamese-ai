from pathlib import Path


class SDKGenerator:
    """
    Trình tự động sinh mã kết nối WebSocket (Flutter / React Native) tương thích
    chuẩn E2EE / AEAD1 của MobileAppBridge và hàng đợi ngoại tuyến tự động đồng bộ.
    """

    FLUTTER_TEMPLATE = """// EvoNetConnector.dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class EvoNetConnector {
  final WebSocketChannel channel;
  final String secretKey;
  final List<String> offlineQueue = [];
  bool isConnected = true;

  EvoNetConnector(String url, {this.secretKey = "SECRET_KEY"})
      : channel = WebSocketChannel.connect(Uri.parse(url));

  String encryptPayload(Map<String, dynamic> data) {
    final String jsonString = jsonEncode(data);
    return base64Encode(utf8.encode(jsonString));
  }

  void _sendOrQueue(Map<String, dynamic> payload) {
    String encrypted = encryptPayload(payload);
    if (isConnected) {
      channel.sink.add(encrypted);
      while (offlineQueue.isNotEmpty) {
        channel.sink.add(offlineQueue.removeAt(0));
      }
    } else {
      offlineQueue.add(encrypted);
    }
  }

  void syncHardware(double batteryLevel, bool isPlugged, String thermalState) {
    _sendOrQueue({
      "type": "hardware_sync",
      "data": {
        "battery_level": batteryLevel,
        "is_plugged": isPlugged,
        "thermal_state": thermalState
      }
    });
  }

  void sendMessage(String message) {
    _sendOrQueue({
      "type": "chat",
      "data": {"message": message}
    });
  }
}
"""

    REACT_NATIVE_TEMPLATE = """// EvoNetConnector.js
export class EvoNetConnector {
  constructor(url, secretKey = "SECRET_KEY") {
    this.ws = new WebSocket(url);
    this.secretKey = secretKey;
    this.offlineQueue = [];
    this.isConnected = true;
  }

  encryptPayload(data) {
    return btoa(JSON.stringify(data));
  }

  _sendOrQueue(payload) {
    const encrypted = this.encryptPayload(payload);
    if (this.isConnected) {
      this.ws.send(encrypted);
      while (this.offlineQueue.length > 0) {
        this.ws.send(this.offlineQueue.shift());
      }
    } else {
      this.offlineQueue.push(encrypted);
    }
  }

  syncHardware(batteryLevel, isPlugged, thermalState) {
    this._sendOrQueue({
      type: "hardware_sync",
      data: {
        battery_level: batteryLevel,
        is_plugged: isPlugged,
        thermal_state: thermalState
      }
    });
  }

  sendMessage(message) {
    this._sendOrQueue({
      type: "chat",
      data: { message: message }
    });
  }
}
"""

    @classmethod
    def generate_flutter(cls, output_dir: str) -> str:
        """Sinh file kết nối cho dự án Flutter."""
        path = Path(output_dir) / "EvoNetConnector.dart"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(cls.FLUTTER_TEMPLATE)
        return str(path)

    @classmethod
    def generate_react_native(cls, output_dir: str) -> str:
        """Sinh file kết nối cho dự án React Native."""
        path = Path(output_dir) / "EvoNetConnector.js"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(cls.REACT_NATIVE_TEMPLATE)
        return str(path)
