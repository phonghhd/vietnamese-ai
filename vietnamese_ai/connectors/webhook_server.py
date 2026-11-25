"""MayChuWebhookKenh (ChannelWebhookServer) - Máy Chủ HTTP Webhook Hợp Nhất Cho Zalo OA & Telegram Bot."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from vietnamese_ai.connectors.telegram_bot import KetNoiTelegram
from vietnamese_ai.connectors.zalo_oa import KetNoiZaloOA
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

MAX_WEBHOOK_BODY_BYTES = 4 * 1024 * 1024


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Máy chủ HTTP Webhook zero-dependency tích hợp sẵn `/webhook/zalo`, `/webhook/telegram` và `/health` "
        "nối thẳng tới `TroLyPhapLy`, `TongDaiCSKH` hoặc `RAGPipeline`."
    ),
    nhom_extra="core",
)
class MayChuWebhookKenh:
    """Máy chủ Webhook thống nhất phục vụ đồng thời Zalo OA và Telegram Bot."""

    def __init__(
        self,
        ket_noi_zalo: Optional[KetNoiZaloOA] = None,
        ket_noi_telegram: Optional[KetNoiTelegram] = None,
        host: str = "0.0.0.0",
        port: int = 8095,
    ) -> None:
        self.ket_noi_zalo = ket_noi_zalo
        self.ket_noi_telegram = ket_noi_telegram
        self.host = host
        self.port = int(port)

    def tao_server(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> ThreadingHTTPServer:
        """Tạo đối tượng `ThreadingHTTPServer` sẵn sàng phục vụ Webhook."""
        h = host or self.host
        p = self.port if port is None else int(port)
        owner = self

        class _WebhookHandler(BaseHTTPRequestHandler):
            def _gui_json(self, obj: Dict[str, Any], status: int = 200) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == "/health":
                    self._gui_json(
                        {
                            "trang_thai": "ok",
                            "zalo_oa_bat": owner.ket_noi_zalo is not None,
                            "telegram_bat": owner.ket_noi_telegram is not None,
                        }
                    )
                    return
                self.send_error(404, "Not Found")

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                raw_len = self.headers.get("Content-Length", "0")
                try:
                    length = int(raw_len)
                except (TypeError, ValueError):
                    self._gui_json({"thanh_cong": False, "loi": "Content-Length không hợp lệ."}, status=400)
                    return
                if length < 0 or length > MAX_WEBHOOK_BODY_BYTES:
                    self._gui_json(
                        {"thanh_cong": False, "loi": "Kích thước gói tin Webhook vượt giới hạn cho phép."},
                        status=413 if length > MAX_WEBHOOK_BODY_BYTES else 400,
                    )
                    return
                raw = self.rfile.read(length) if length > 0 else b"{}"

                if path == "/webhook/zalo":
                    if owner.ket_noi_zalo is None:
                        self._gui_json({"thanh_cong": False, "loi": "Chưa kích hoạt kênh Zalo OA."}, status=503)
                        return
                    sig = self.headers.get("X-ZEvent-Signature")
                    kq = owner.ket_noi_zalo.xu_ly_webhook(raw, signature_header=sig)
                    self._gui_json(kq, status=int(kq.get("ma_http", 200)))
                    return

                if path == "/webhook/telegram":
                    if owner.ket_noi_telegram is None:
                        self._gui_json({"thanh_cong": False, "loi": "Chưa kích hoạt kênh Telegram."}, status=503)
                        return
                    sec = self.headers.get("X-Telegram-Bot-Api-Secret-Token")
                    kq = owner.ket_noi_telegram.xu_ly_webhook(raw, secret_header=sec)
                    self._gui_json(kq, status=int(kq.get("ma_http", 200)))
                    return

                self.send_error(404, "Not Found")

            def log_message(self, format: str, *args: Any) -> None:
                return

        return ThreadingHTTPServer((h, p), _WebhookHandler)

    def chay(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Khởi động máy chủ Webhook lắng nghe sự kiện Zalo OA / Telegram."""
        httpd = self.tao_server(host=host, port=port)
        actual_port = httpd.server_port
        print(f"Vietnamese AI Channel Webhook Server đang chạy tại http://{host or self.host}:{actual_port}")
        if self.ket_noi_zalo is not None:
            print(f"  Zalo OA Webhook:  POST http://{host or self.host}:{actual_port}/webhook/zalo")
        if self.ket_noi_telegram is not None:
            print(f"  Telegram Webhook: POST http://{host or self.host}:{actual_port}/webhook/telegram")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng Channel Webhook Server.")
        finally:
            httpd.server_close()


ChannelWebhookServer = MayChuWebhookKenh

__all__ = [
    "MayChuWebhookKenh",
    "ChannelWebhookServer",
]
