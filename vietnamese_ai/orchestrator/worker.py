"""
NutPhu (Worker Node) - Chạy tiến trình xử lý và cung cấp API đơn giản 0-dependency.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional

logger = logging.getLogger("V-Orchestrator")


class WorkerHandler(BaseHTTPRequestHandler):
    """Trình xử lý HTTP cho Worker."""

    def __init__(self, bo_xu_ly: Callable, *args, **kwargs):
        self.bo_xu_ly = bo_xu_ly
        # Gọi __init__ của lớp cha ở cuối vì nó sẽ khởi động vòng lặp server ngay lập tức
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Xử lý yêu cầu Health Check."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"trang_thai": "hoat_dong"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Xử lý yêu cầu Inference/AI."""
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                # Gọi hàm xử lý cốt lõi của AI
                ket_qua = self.bo_xu_ly(data)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"thanh_cong": True, "ket_qua": ket_qua}).encode("utf-8")
                )

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"thanh_cong": False, "loi": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Vô hiệu hóa log mặc định của HTTPServer để tránh in rác ra màn hình."""
        pass


def tao_handler(bo_xu_ly: Callable):
    """Tạo class Handler với hàm xử lý được nhúng sẵn."""
    return lambda *args, **kwargs: WorkerHandler(bo_xu_ly, *args, **kwargs)


class NutPhu:
    """
    Worker Node (Nút Phụ).
    Nhận dữ liệu từ Master, chạy logic suy luận/huấn luyện và trả kết quả về.
    """

    def __init__(self, bo_xu_ly: Callable, host: str = "127.0.0.1", port: int = 8081):
        """
        Khởi tạo Nút Phụ.

        Args:
            bo_xu_ly: Một hàm Python nhận vào một Dict (dữ liệu đầu vào) và trả về một Dict (kết quả).
            host: IP lắng nghe (mặc định localhost).
            port: Cổng lắng nghe (mặc định 8081).
        """
        self.bo_xu_ly = bo_xu_ly
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None

    def chay(self):
        """Khởi động Worker Server."""
        handler_class = tao_handler(self.bo_xu_ly)
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
