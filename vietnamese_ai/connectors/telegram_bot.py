"""KetNoiTelegram (TelegramBotConnector) - Bộ Kết Nối Telegram Bot Webhook & Long-Polling."""

from __future__ import annotations

import hmac
import json
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Union

from vietnamese_ai.connectors.base import KetNoiKenhCoSo, PhanHoiKenh, TinNhanKenh
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ kết nối Telegram Bot API hỗ trợ xác thực `X-Telegram-Bot-Api-Secret-Token`, "
        "tự chia nhỏ câu trả lời dài vượt 4096 ký tự, Tường lửa AI và chế độ Webhook / Polling."
    ),
    nhom_extra="core",
)
class KetNoiTelegram(KetNoiKenhCoSo):
    """
    Bộ kết nối Telegram Bot API cho hệ sinh thái Vietnamese AI.

    Hỗ trợ:
    - Xác thực header bí mật `X-Telegram-Bot-Api-Secret-Token` bằng `hmac.compare_digest`.
    - Bóc tách bản tin Telegram Update (`message.text`, `message.chat.id`, `message.from`).
    - Tự động cắt đoạn an toàn dưới ngưỡng 4,096 ký tự của Telegram khi gửi câu trả lời pháp lý dài.
    """

    TEN_KENH = "telegram"
    GIOI_HAN_KY_TU = 4000

    def __init__(
        self,
        bo_xu_ly: Optional[Union[Any, Callable[..., str]]] = None,
        bot_token: str = "",
        webhook_secret: str = "",
        bat_tuong_lua: bool = True,
        che_do_gui_that: bool = False,
        timeout_giay: int = 15,
        quan_ly_workspace: Optional[Any] = None,
        ma_workspace: str = "mac_dinh",
    ) -> None:
        super().__init__(
            bo_xu_ly=bo_xu_ly,
            bat_tuong_lua=bat_tuong_lua,
            che_do_gui_that=che_do_gui_that,
            quan_ly_workspace=quan_ly_workspace,
            ma_workspace=ma_workspace,
        )
        self.bot_token = bot_token
        self.webhook_secret = webhook_secret
        self.timeout_giay = max(1, int(timeout_giay))

    def xac_thuc_secret_token(self, secret_header: Optional[str]) -> bool:
        """Xác thực header `X-Telegram-Bot-Api-Secret-Token` bằng so sánh hằng thời gian."""
        if not self.webhook_secret:
            return True
        if not secret_header:
            return False
        return hmac.compare_digest(self.webhook_secret, secret_header.strip())

    def phan_tich_webhook(self, payload: Dict[str, Any]) -> Optional[TinNhanKenh]:
        """Chuyển đổi Telegram Update JSON sang `TinNhanKenh`."""
        if not isinstance(payload, dict):
            return None
        msg = payload.get("message") or payload.get("edited_message")
        if not isinstance(msg, dict):
            return None
        text = str(msg.get("text", "")).strip()
        chat = msg.get("chat") or {}
        sender = msg.get("from") or {}
        chat_id = str(chat.get("id", "")).strip()
        user_id = str(sender.get("id", chat_id)).strip()
        if not chat_id or not text:
            return None
        ten = str(sender.get("username") or sender.get("first_name") or "").strip()
        return TinNhanKenh(
            kenh=self.TEN_KENH,
            nguoi_gui_id=user_id,
            ten_nguoi_gui=ten,
            noi_dung=text,
            ma_tin_nhan=str(msg.get("message_id", "")),
            cuoc_hoi_thoai_id=chat_id,
            du_lieu_goc=payload,
        )

    @classmethod
    def chia_doan_tin_nhan(cls, noi_dung: str) -> List[str]:
        """Chia văn bản dài thành các đoạn an toàn dưới giới hạn 4,096 ký tự của Telegram."""
        if len(noi_dung) <= cls.GIOI_HAN_KY_TU:
            return [noi_dung]
        cac_doan: List[str] = []
        con_lai = noi_dung
        while con_lai:
            if len(con_lai) <= cls.GIOI_HAN_KY_TU:
                cac_doan.append(con_lai)
                break
            vi_tri = con_lai.rfind("\n", 0, cls.GIOI_HAN_KY_TU)
            if vi_tri < cls.GIOI_HAN_KY_TU // 2:
                vi_tri = cls.GIOI_HAN_KY_TU
            cac_doan.append(con_lai[:vi_tri].rstrip())
            con_lai = con_lai[vi_tri:].lstrip()
        return cac_doan

    def tao_payload_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Tạo payload chuẩn Telegram `sendMessage`."""
        doan_dau = self.chia_doan_tin_nhan(noi_dung)[0]
        return {
            "chat_id": str(nguoi_nhan_id),
            "text": doan_dau,
        }

    def gui_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Gửi toàn bộ các đoạn trả lời tới Telegram Bot API (`sendMessage`)."""
        if not self.bot_token:
            return {
                "thanh_cong": False,
                "loi": "Thiếu Telegram `bot_token` để gửi tin nhắn thực tế.",
                "payload": self.tao_payload_tra_loi(nguoi_nhan_id, noi_dung),
            }
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        ket_qua_cac_doan: List[Dict[str, Any]] = []
        for doan in self.chia_doan_tin_nhan(noi_dung):
            payload = {"chat_id": str(nguoi_nhan_id), "text": doan}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                ket_qua_cac_doan.append(res)
            except Exception as exc:
                return {"thanh_cong": False, "loi": str(exc), "cac_doan_da_gui": ket_qua_cac_doan}
        return {
            "thanh_cong": all(bool(x.get("ok")) for x in ket_qua_cac_doan),
            "so_doan_da_gui": len(ket_qua_cac_doan),
            "cac_doan_da_gui": ket_qua_cac_doan,
        }

    def xu_ly_webhook(
        self,
        raw_body: Union[str, bytes],
        secret_header: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Xác thực secret token, phân tích bản tin Telegram Update và trả về kết quả xử lý."""
        if not self.xac_thuc_secret_token(secret_header):
            return {"thanh_cong": False, "ma_http": 401, "loi": "Telegram secret token không hợp lệ."}
        body_str = raw_body.decode("utf-8") if isinstance(raw_body, bytes) else str(raw_body)
        payload = json.loads(body_str) if body_str.strip() else {}
        tin = self.phan_tich_webhook(payload)
        if tin is None:
            return {"thanh_cong": True, "bo_qua": True, "ly_do": "Update không chứa tin nhắn văn bản."}
        phan_hoi: PhanHoiKenh = self.xu_ly_tin_nhan(tin)
        return {"thanh_cong": True, "bo_qua": False, "phan_hoi": phan_hoi.to_dict()}


TelegramBotConnector = KetNoiTelegram

__all__ = [
    "KetNoiTelegram",
    "TelegramBotConnector",
]
