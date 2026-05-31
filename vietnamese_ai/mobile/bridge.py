import base64
import json
from typing import Any, Dict

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent
from vietnamese_ai.mobile.power_manager import PowerManager


class MobileAppBridge:
    """
    Cầu nối (Bridge) giữa giao diện người dùng (React Native / Flutter) và Lõi EvoNet AI.
    Hỗ trợ WebSockets, Mã hóa E2EE và Push Notification cho các tác vụ ngầm.
    """

    def __init__(self, mobile_agent: MobileHybridAgent, e2e_key: str = "SECRET_KEY"):
        self.mobile_agent = mobile_agent
        self.clients = set()
        self.e2e_key = e2e_key

    def _giai_ma_e2e(self, payload_ma_hoa: str) -> Dict[str, Any]:
        """Giả lập giải mã AES-256 GCM (E2EE)."""
        try:
            ban_ro = base64.b64decode(payload_ma_hoa).decode("utf-8")
            return json.loads(ban_ro)
        except Exception:
            return {}

    def _ma_hoa_e2e(self, data: Dict[str, Any]) -> str:
        """Giả lập mã hóa dữ liệu trả về (E2EE)."""
        chuoi_json = json.dumps(data)
        return base64.b64encode(chuoi_json.encode("utf-8")).decode("utf-8")

    def _goi_push_notification(self, thong_diep: str) -> None:
        """Giả lập Ping APNs (Apple) / FCM (Firebase) vào màn hình khóa."""
        print(f"[MobileAppBridge] PUSH NOTIFICATION: {thong_diep}")

    async def _handle_hardware_sync(self, payload: Dict[str, Any]) -> str:
        """Xử lý tín hiệu phần cứng do React Native / Flutter gửi lên."""
        pin = float(payload.get("battery_level", 100.0))
        cam_sac = bool(payload.get("is_plugged", True))
        nhiet_do = str(payload.get("thermal_state", "normal"))

        PowerManager.dong_bo_phan_cung(pin, cam_sac, nhiet_do)
        return json.dumps({"status": "ok", "message": "Đã đồng bộ phần cứng Mobile"})

    async def _handle_chat(self, payload: Dict[str, Any], is_background: bool = False) -> str:
        """Nhận tin nhắn từ giao diện App, chuyển cho Tác tử xử lý."""
        prompt = str(payload.get("message", ""))

        ket_qua = self.mobile_agent.chay(prompt)

        if is_background:
            self._goi_push_notification("EvoNet: Tác tử đã xử lý xong tác vụ của bạn!")

        return json.dumps({"text": ket_qua, "is_done": True})

    async def handle_request(self, event_type: str, payload_raw: str, is_background: bool = False) -> str:
        """Hàm phân phối Request từ WebSocket (Đã hỗ trợ E2EE)."""
        payload = self._giai_ma_e2e(payload_raw)
        if not payload:
            return self._ma_hoa_e2e({"error": "Lỗi giải mã E2EE", "status": "failed"})

        if event_type == "hardware_sync":
            kq_json_str = await self._handle_hardware_sync(payload)
            return self._ma_hoa_e2e(json.loads(kq_json_str))
        elif event_type == "chat":
            kq_json_str = await self._handle_chat(payload, is_background)
            return self._ma_hoa_e2e(json.loads(kq_json_str))

        return self._ma_hoa_e2e({"error": "Unknown event_type", "status": "failed"})

    def get_websocket_handler(self):
        """Giả lập hàm lấy Websocket Handler để cắm vào FastAPI/Sanic."""
        pass
