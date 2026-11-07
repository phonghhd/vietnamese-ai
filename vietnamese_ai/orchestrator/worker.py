"""
NutPhu (Worker Node) - Chạy tiến trình xử lý kèm giới hạn Payload và xác thực khóa cụm HMAC.
"""

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional

logger = logging.getLogger("V-Orchestrator")

MAX_WORKER_BODY_BYTES = 1_048_576  # 1 MB chống DoS tràn bộ nhớ


class WorkerHandler(BaseHTTPRequestHandler):
    """Trình xử lý HTTP cho Worker."""

    def __init__(
        self,
        bo_xu_ly: Callable,
        ma_bi_mat_cum: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.bo_xu_ly = bo_xu_ly
        self.ma_bi_mat_cum = ma_bi_mat_cum
        super().__init__(*args, **kwargs)

    def _tra_json(self, ma_http: int, du_lieu: dict) -> None:
        self.send_response(ma_http)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(du_lieu, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        """Xử lý yêu cầu Health Check."""
        if self.path == "/health":
            self._tra_json(200, {"trang_thai": "hoat_dong"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Xử lý yêu cầu Inference/AI có bảo vệ kích thước và xác thực cụm."""
        if self.path == "/predict":
            if self.ma_bi_mat_cum is not None:
                token_gui = self.headers.get("X-Cluster-Token", "")
                if not token_gui or not hmac.compare_digest(token_gui, self.ma_bi_mat_cum):
                    self._tra_json(401, {"thanh_cong": False, "loi": "Từ chối truy cập: Khóa cụm không hợp lệ"})
                    return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_WORKER_BODY_BYTES:
                self._tra_json(
                    413,
                    {"thanh_cong": False, "loi": f"Payload vượt quá giới hạn {MAX_WORKER_BODY_BYTES} bytes"},
                )
                return

            post_data = self.rfile.read(max(0, content_length))
            try:
                data = json.loads(post_data.decode("utf-8")) if post_data else {}
                ket_qua = self.bo_xu_ly(data)
                self._tra_json(200, {"thanh_cong": True, "ket_qua": ket_qua})
            except Exception as e:
                self._tra_json(500, {"thanh_cong": False, "loi": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Vô hiệu hóa log mặc định của HTTPServer để tránh in rác ra màn hình."""
        pass


def tao_handler(bo_xu_ly: Callable, ma_bi_mat_cum: Optional[str] = None):
    """Tạo class Handler với hàm xử lý và khóa cụm được nhúng sẵn."""
    return lambda *args, **kwargs: WorkerHandler(bo_xu_ly, ma_bi_mat_cum, *args, **kwargs)


class NutPhu:
    """
    Worker Node (Nút Phụ).
    Nhận dữ liệu từ Master, xác thực khóa cụm (nếu bật) và thực thi suy luận an toàn.
    """

    def __init__(
        self,
        bo_xu_ly: Callable,
        host: str = "127.0.0.1",
        port: int = 8081,
        ma_bi_mat_cum: Optional[str] = None,
    ):
        self.bo_xu_ly = bo_xu_ly
        self.host = host
        self.port = port
        self.ma_bi_mat_cum = ma_bi_mat_cum
        self.server: Optional[HTTPServer] = None

    def kiem_tra_token_cum(self, token: str) -> bool:
        """Xác thực khóa cụm bằng `hmac.compare_digest` chống tấn công định thời."""
        if self.ma_bi_mat_cum is None:
            return True
        if not token:
            return False
        return hmac.compare_digest(token, self.ma_bi_mat_cum)

    def chay(self):
        """Khởi động Worker Server."""
        handler_class = tao_handler(self.bo_xu_ly, self.ma_bi_mat_cum)
        self.server = HTTPServer((self.host, self.port), handler_class)
        logger.info(f"[NutPhu] Khởi động thành công tại http://{self.host}:{self.port}")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.dung()

    def dung(self):
        """Dừng Worker Server."""
        if self.server:
            logger.info(f"[NutPhu] Đang dừng server tại cổng {self.port}...")
            self.server.server_close()
