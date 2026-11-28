"""KetNoiZaloOA (ZaloOAConnector) - Bộ Kết Nối Zalo Official Account Webhook v3 & OpenAPI v3.0."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Union

from vietnamese_ai.connectors.base import KetNoiKenhCoSo, PhanHoiKenh, TinNhanKenh
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ kết nối Zalo Official Account (OA) v3 hỗ trợ xác thực chữ ký MAC SHA-256 (`X-ZEvent-Signature`), "
        "bóc tách sự kiện `user_send_text`, bảo vệ Tường lửa AI và gửi phản hồi qua Zalo OpenAPI v3.0."
    ),
    nhom_extra="core",
)
class KetNoiZaloOA(KetNoiKenhCoSo):
    """
    Bộ kết nối Webhook & OpenAPI v3.0 cho Zalo Official Account (OA).

    Hỗ trợ:
    - Xác thực chữ ký MAC chuẩn Zalo OA: `mac=sha256(app_id + raw_body + timestamp + oa_secret_key)` bằng `hmac.compare_digest`.
    - Bóc tách sự kiện `user_send_text` sang `TinNhanKenh` chuẩn.
    - Đóng gói và gửi tin nhắn chăm sóc khách hàng qua `https://openapi.zalo.me/v3.0/oa/message/cs`.
    """

    TEN_KENH = "zalo_oa"
    API_GUI_TIN_CS = "https://openapi.zalo.me/v3.0/oa/message/cs"

    def __init__(
        self,
        bo_xu_ly: Optional[Union[Any, Callable[..., str]]] = None,
        app_id: str = "",
        oa_secret_key: str = "",
        access_token: str = "",
        bat_tuong_lua: bool = True,
        che_do_gui_that: bool = False,
        timeout_giay: int = 15,
        quan_ly_workspace: Optional[Any] = None,
        ma_workspace: str = "mac_dinh",
        cho_phep_khong_xac_thuc: bool = False,
        han_lech_timestamp_giay: Optional[int] = 300,
        kich_thuoc_bo_nho_replay: int = 4096,
    ) -> None:
        super().__init__(
            bo_xu_ly=bo_xu_ly,
            bat_tuong_lua=bat_tuong_lua,
            che_do_gui_that=che_do_gui_that,
            quan_ly_workspace=quan_ly_workspace,
            ma_workspace=ma_workspace,
        )
        self.app_id = app_id
        self.oa_secret_key = oa_secret_key
        self.access_token = access_token
        self.timeout_giay = max(1, int(timeout_giay))
        self.cho_phep_khong_xac_thuc = bool(cho_phep_khong_xac_thuc)
        self.han_lech_timestamp_giay = han_lech_timestamp_giay
        self._kich_thuoc_bo_nho_replay = max(16, int(kich_thuoc_bo_nho_replay))
        self._msg_id_da_xu_ly: "OrderedDict[str, float]" = OrderedDict()
        self._replay_lock = threading.Lock()

    def tao_chu_ky_mac(self, raw_body: Union[str, bytes], timestamp: str) -> str:
        """Tính chữ ký MAC SHA-256 theo đặc tả Zalo OA Webhook (`mac=<hex>`)."""
        body_str = raw_body.decode("utf-8") if isinstance(raw_body, bytes) else str(raw_body)
        chuoi = f"{self.app_id}{body_str}{timestamp}{self.oa_secret_key}"
        return "mac=" + hashlib.sha256(chuoi.encode("utf-8")).hexdigest()

    def _kiem_tra_timestamp_hop_le(self, timestamp: str) -> bool:
        if self.han_lech_timestamp_giay is None or self.cho_phep_khong_xac_thuc:
            return True
        if not timestamp or not str(timestamp).strip():
            return False
        try:
            ts_val = float(str(timestamp).strip())
        except ValueError:
            return False
        ts_sec = ts_val / 1000.0 if ts_val > 1e11 else ts_val
        return abs(time.time() - ts_sec) <= float(self.han_lech_timestamp_giay)

    def _ghi_nhan_va_kiem_tra_replay(self, msg_id: str) -> bool:
        if not msg_id:
            return True
        now = time.time()
        with self._replay_lock:
            if msg_id in self._msg_id_da_xu_ly:
                return False
            self._msg_id_da_xu_ly[msg_id] = now
            while len(self._msg_id_da_xu_ly) > self._kich_thuoc_bo_nho_replay:
                self._msg_id_da_xu_ly.popitem(last=False)
        return True

    def xac_thuc_chu_ky(
        self,
        raw_body: Union[str, bytes],
        timestamp: str,
        signature_header: Optional[str],
    ) -> bool:
        """Xác thực header `X-ZEvent-Signature` bằng so sánh hằng thời gian `hmac.compare_digest`."""
        if not self.oa_secret_key:
            return bool(self.cho_phep_khong_xac_thuc)
        if not signature_header:
            return False
        if not self._kiem_tra_timestamp_hop_le(timestamp):
            return False
        ky_vong = self.tao_chu_ky_mac(raw_body, str(timestamp))
        return hmac.compare_digest(ky_vong.strip().lower(), signature_header.strip().lower())

    def phan_tich_webhook(self, payload: Dict[str, Any]) -> Optional[TinNhanKenh]:
        """Chuyển đổi payload Zalo OA Webhook (`event_name == 'user_send_text'`) sang `TinNhanKenh`."""
        if not isinstance(payload, dict):
            return None
        event_name = str(payload.get("event_name", ""))
        if event_name and event_name != "user_send_text":
            return None
        sender = payload.get("sender") or {}
        message = payload.get("message") or {}
        user_id = str(sender.get("id", "")).strip()
        text = str(message.get("text", "")).strip()
        if not user_id or not text:
            return None
        return TinNhanKenh(
            kenh=self.TEN_KENH,
            nguoi_gui_id=user_id,
            noi_dung=text,
            ma_tin_nhan=str(message.get("msg_id", "")),
            cuoc_hoi_thoai_id=user_id,
            du_lieu_goc=payload,
        )

    def tao_payload_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Tạo JSON payload chuẩn Zalo OpenAPI v3.0 (`/v3.0/oa/message/cs`, giới hạn 2000 ký tự)."""
        van_ban_cat = noi_dung[:1990] + ("..." if len(noi_dung) > 1990 else "")
        return {
            "recipient": {"user_id": str(nguoi_nhan_id)},
            "message": {"text": van_ban_cat},
        }

    def gui_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Gửi tin nhắn phản hồi tới Zalo OA OpenAPI v3.0 (yêu cầu `access_token`)."""
        payload = self.tao_payload_tra_loi(nguoi_nhan_id, noi_dung)
        if not self.access_token:
            return {
                "thanh_cong": False,
                "loi": "Thiếu Zalo OA `access_token` để gửi tin nhắn thực tế.",
                "payload": payload,
            }
        req = urllib.request.Request(
            self.API_GUI_TIN_CS,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "access_token": self.access_token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return {"thanh_cong": int(data.get("error", 0)) == 0, "phan_hoi_api": data}
        except Exception as exc:
            return {"thanh_cong": False, "loi": str(exc)}

    def xu_ly_webhook(
        self,
        raw_body: Union[str, bytes],
        signature_header: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Xác thực MAC, phân tích sự kiện Zalo OA và trả về kết quả xử lý đầy đủ."""
        try:
            body_str = raw_body.decode("utf-8") if isinstance(raw_body, bytes) else str(raw_body)
            payload = json.loads(body_str) if body_str.strip() else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"thanh_cong": False, "ma_http": 400, "loi": "Payload JSON Zalo OA không hợp lệ."}
        if not isinstance(payload, dict):
            return {"thanh_cong": False, "ma_http": 400, "loi": "Payload JSON Zalo OA phải là đối tượng JSON."}
        timestamp = str(payload.get("timestamp", ""))
        if not self.xac_thuc_chu_ky(body_str, timestamp, signature_header):
            return {"thanh_cong": False, "ma_http": 401, "loi": "Chữ ký MAC Zalo OA không hợp lệ."}

        tin = self.phan_tich_webhook(payload)
        if tin is None:
            return {"thanh_cong": True, "bo_qua": True, "ly_do": "Sự kiện không phải tin nhắn văn bản."}
        if tin.ma_tin_nhan and not self._ghi_nhan_va_kiem_tra_replay(tin.ma_tin_nhan):
            return {"thanh_cong": False, "ma_http": 409, "loi": "Tin nhắn Zalo OA bị trùng lặp (replay)."}

        phan_hoi: PhanHoiKenh = self.xu_ly_tin_nhan(tin)
        return {"thanh_cong": True, "bo_qua": False, "phan_hoi": phan_hoi.to_dict()}


ZaloOAConnector = KetNoiZaloOA

__all__ = [
    "KetNoiZaloOA",
    "ZaloOAConnector",
]
