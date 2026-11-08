from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Any, Callable, Dict, List, Tuple

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent
from vietnamese_ai.mobile.power_manager import PowerManager


class MobileAppBridge:
    """
    Cầu nối (Bridge) giữa giao diện người dùng (React Native / Flutter) và Lõi EvoNet AI.
    Hỗ trợ WebSockets, Mã hóa xác thực AEAD (PBKDF2-HMAC-SHA256 + SHA256-CTR + HMAC-SHA256)
    và Push Notification cho các tác vụ ngầm.
    """

    def __init__(
        self,
        mobile_agent: MobileHybridAgent,
        e2e_key: str = "SECRET_KEY",
        strict_aead: bool = False,
    ):
        self.mobile_agent = mobile_agent
        self.clients = set()
        self.e2e_key = e2e_key
        self.strict_aead = strict_aead
        self.notification_history: List[str] = []

    def _dan_xuat_khoa(self, nonce: bytes) -> Tuple[bytes, bytes]:
        """Dẫn xuất khóa mã hóa (enc_key) và khóa xác thực (mac_key) bằng PBKDF2-HMAC-SHA256."""
        dk = hashlib.pbkdf2_hmac(
            "sha256", self.e2e_key.encode("utf-8"), nonce, iterations=1000, dklen=64
        )
        return dk[:32], dk[32:]

    @staticmethod
    def _xor_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
        """Mã hóa/giải mã dòng bằng bộ sinh khóa SHA256-CTR."""
        out = bytearray(len(data))
        num_blocks = (len(data) + 31) // 32
        for counter in range(num_blocks):
            block_key = hashlib.sha256(
                enc_key + nonce + counter.to_bytes(4, "big")
            ).digest()
            start = counter * 32
            end = min(start + 32, len(data))
            for i in range(start, end):
                out[i] = data[i] ^ block_key[i - start]
        return bytes(out)

    def ma_hoa_aead(self, data: Dict[str, Any]) -> str:
        """Mã hóa xác thực AEAD thực tế với Nonce 16-byte ngẫu nhiên và HMAC-SHA256 tag."""
        plaintext = json.dumps(data, ensure_ascii=False).encode("utf-8")
        nonce = os.urandom(16)
        enc_key, mac_key = self._dan_xuat_khoa(nonce)
        ciphertext = self._xor_keystream(plaintext, enc_key, nonce)
        tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        envelope = {
            "v": "AEAD1",
            "n": base64.b64encode(nonce).decode("ascii"),
            "c": base64.b64encode(ciphertext).decode("ascii"),
            "t": base64.b64encode(tag).decode("ascii"),
        }
        raw_env = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        return "AEAD1." + base64.b64encode(raw_env).decode("ascii")

    def giai_ma_aead(self, token: str) -> Dict[str, Any]:
        """Giải mã và xác thực tính toàn vẹn của gói tin AEAD1."""
        raw_b64 = token[6:] if token.startswith("AEAD1.") else token
        env = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        if env.get("v") != "AEAD1":
            raise ValueError("Phiên bản AEAD không hợp lệ.")
        nonce = base64.b64decode(env["n"])
        ciphertext = base64.b64decode(env["c"])
        tag = base64.b64decode(env["t"])
        enc_key, mac_key = self._dan_xuat_khoa(nonce)
        expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Xác thực HMAC thất bại: Dữ liệu đã bị can thiệp hoặc sai khóa.")
        plaintext = self._xor_keystream(ciphertext, enc_key, nonce)
        return json.loads(plaintext.decode("utf-8"))

    def _giai_ma_e2e(self, payload_ma_hoa: str) -> Dict[str, Any]:
        """Giải mã E2EE (hỗ trợ cả chuẩn AEAD1 và Base64 truyền thống)."""
        try:
            if payload_ma_hoa.startswith("AEAD1."):
                return self.giai_ma_aead(payload_ma_hoa)
            if self.strict_aead:
                return {}
            ban_ro = base64.b64decode(payload_ma_hoa).decode("utf-8")
            obj = json.loads(ban_ro)
            if isinstance(obj, dict) and obj.get("v") == "AEAD1":
                return self.giai_ma_aead(payload_ma_hoa)
            return obj
        except Exception:
            return {}

    def _ma_hoa_e2e(self, data: Dict[str, Any], use_aead: bool = False) -> str:
        """Mã hóa dữ liệu trả về (E2EE)."""
        if use_aead or self.strict_aead:
            return self.ma_hoa_aead(data)
        chuoi_json = json.dumps(data)
        return base64.b64encode(chuoi_json.encode("utf-8")).decode("utf-8")

    def _goi_push_notification(self, thong_diep: str) -> None:
        """Gửi thông báo đẩy và lưu vào hàng đợi thông báo của thiết bị."""
        self.notification_history.append(thong_diep)
        try:
            from vietnamese_ai.mobile.mobile_tools import MobileDeviceState

            MobileDeviceState.them_thong_bao(f"(Push) {thong_diep}")
        except Exception:
            pass
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

    async def handle_request(
        self, event_type: str, payload_raw: str, is_background: bool = False
    ) -> str:
        """Hàm phân phối Request từ WebSocket (Hỗ trợ cả AEAD1 và Base64)."""
        is_aead_request = payload_raw.startswith("AEAD1.")
        payload = self._giai_ma_e2e(payload_raw)
        if not payload:
            return self._ma_hoa_e2e(
                {"error": "Lỗi giải mã E2EE", "status": "failed"}, use_aead=is_aead_request
            )

        if event_type == "hardware_sync":
            kq_json_str = await self._handle_hardware_sync(payload)
            return self._ma_hoa_e2e(json.loads(kq_json_str), use_aead=is_aead_request)
        elif event_type == "chat":
            kq_json_str = await self._handle_chat(payload, is_background)
            return self._ma_hoa_e2e(json.loads(kq_json_str), use_aead=is_aead_request)

        return self._ma_hoa_e2e(
            {"error": "Unknown event_type", "status": "failed"}, use_aead=is_aead_request
        )

    def get_websocket_handler(self) -> Callable:
        """Trả về hàm xử lý WebSocket ASGI/FastAPI thực tế."""

        async def _ws_handler(websocket: Any) -> None:
            if hasattr(websocket, "accept"):
                await websocket.accept()
            self.clients.add(websocket)
            try:
                while True:
                    raw_msg = await websocket.receive_text()
                    try:
                        frame = json.loads(raw_msg)
                        event_type = str(frame.get("type", "chat"))
                        payload_raw = str(frame.get("payload", ""))
                        is_bg = bool(frame.get("is_background", False))
                    except Exception:
                        event_type = "chat"
                        payload_raw = raw_msg
                        is_bg = False

                    resp = await self.handle_request(
                        event_type, payload_raw, is_background=is_bg
                    )
                    await websocket.send_text(resp)
            except Exception:
                pass
            finally:
                self.clients.discard(websocket)

        return _ws_handler
