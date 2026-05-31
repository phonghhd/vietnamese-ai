"""ServerDonGian - Server HTTP đơn giản để phục vụ mô hình."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np

from vietnamese_ai.utils.logger import Logger


class _RequestHandler(BaseHTTPRequestHandler):
    """Xử lý HTTP request."""

    mo_hinh = None
    logger = Logger("API")

    def do_POST(self):
        if self.path == "/du_doan":
            self._xu_ly_du_doan()
        else:
            self._tra_loi(404, {"loi": "Khong tim thay endpoint"})

    def do_GET(self):
        if self.path == "/":
            self._tra_loi(
                200,
                {
                    "ten": "Vietnamese AI API",
                    "trang_thai": "hoat_dong",
                    "endpoints": ["/du_doan (POST)", "/ (GET)"],
                },
            )
        elif self.path == "/suc_khoe":
            self._tra_loi(200, {"trang_thai": "tot"})
        else:
            self._tra_loi(404, {"loi": "Khong tim thay endpoint"})

    def _xu_ly_du_doan(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            if self.mo_hinh is None:
                self._tra_loi(500, {"loi": "Mo hinh chua duoc tai"})
                return

            du_lieu = np.array(data.get("du_lieu", []))
            if du_lieu.ndim == 1:
                du_lieu = du_lieu.reshape(1, -1)

            ket_qua = self.mo_hinh.du_doan(du_lieu)
            self._tra_loi(
                200,
                {
                    "ket_qua": ket_qua.tolist(),
                    "so_mau": len(ket_qua),
                },
            )

        except Exception as e:
            self._tra_loi(500, {"loi": str(e)})

    def _tra_loi(self, ma: int, data: dict):
        self.send_response(ma)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        self.logger.info(f"{self.address_string()} - {format % args}")


class ServerDonGian:
    """
    Server HTTP đơn giản để phục vụ mô hình AI qua API.

    Sử dụng:
        >>> server = ServerDonGian(mo_hinh=mo_hinh_da_huan_luyen)
        >>> server.chay(port=8080)

    Gửi request:
        POST /du_doan
        {"du_lieu": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    """

    def __init__(self, mo_hinh: Any, ten: str = "VietnameseAI"):
        self.mo_hinh = mo_hinh
        self.ten = ten
        self.logger = Logger(ten)

    def chay(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Khởi động server."""
        _RequestHandler.mo_hinh = self.mo_hinh
        _RequestHandler.logger = self.logger

        server = HTTPServer((host, port), _RequestHandler)
        self.logger.info(f"Server '{self.ten}' đang chạy tại http://{host}:{port}")
        self.logger.info("Endpoints: GET / | GET /suc_khoe | POST /du_doan")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("Server đang tắt...")
            server.server_close()
