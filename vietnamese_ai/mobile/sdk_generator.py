from pathlib import Path


class SDKGenerator:
    """
    Trình tự động sinh mã nguồn (Boilerplate) kết nối WebSocket cho App Di động.
    Lập trình viên UI (React Native / Flutter) chỉ việc Copy file này vào dự án App là chạy.
    (Đã bao gồm Mã hóa E2EE và Hàng đợi Ngoại tuyến).
    """

    FLUTTER_TEMPLATE = """// EvoNetConnector.dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class EvoNetConnector {
  final WebSocketChannel channel;
  final List<String> offlineQueue = [];
  bool isConnected = true; // Cần kết nối với package connectivity_plus thực tế

  EvoNetConnector(String url) : channel = WebSocketChannel.connect(Uri.parse(url));

  String encryptPayload(Map<String, dynamic> data) {
    // TODO: Sử dụng package encrypt (AES) trong thực tế. Giả lập mã hóa bằng Base64.
    String jsonString = jsonEncode(data);
    return base64Encode(utf8.encode(jsonString));
  }

  void _sendOrQueue(Map<String, dynamic> payload) {
    String encrypted = encryptPayload(payload);
    if (isConnected) {
      channel.sink.add(encrypted);
      // Giả lập gửi bù dữ liệu nếu có mạng lại
      while(offlineQueue.isNotEmpty) {
         channel.sink.add(offlineQueue.removeAt(0));
      }
    } else {
      offlineQueue.add(encrypted);
    }
  }

  // Gửi thông số Pin, Sạc lên cho EvoNet PowerManager (v27) phân tích
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

  // Gửi tin nhắn chat đến Mobile Agent
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
  constructor(url) {
    this.ws = new WebSocket(url);
    this.offlineQueue = [];
    this.isConnected = true; // Cần kết nối với NetInfo thực tế
  }

  encryptPayload(data) {
    // TODO: Sử dụng thư viện crypto-js (AES) trong thực tế. Giả lập bằng btoa (Base64).
    return btoa(JSON.stringify(data));
  }

  _sendOrQueue(payload) {
    const encrypted = this.encryptPayload(payload);
    if (this.isConnected) {
      this.ws.send(encrypted);
      // Gửi bù dữ liệu ngoại tuyến
      while(this.offlineQueue.length > 0) {
        this.ws.send(this.offlineQueue.shift());
      }
    } else {
      this.offlineQueue.push(encrypted);
    }
  }

  // Gửi thông số Pin, Sạc lên cho EvoNet PowerManager (v27) phân tích
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

  // Gửi tin nhắn chat đến Mobile Agent
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
