"""
NutChinh (Master Node) - Quản lý Workers, cân bằng tải, Auto-healing.
"""

import json
import logging
import multiprocessing
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List, Optional

from .balancer import CanBangTai
from .worker import NutPhu

logger = logging.getLogger("V-Orchestrator")


class MasterHandler(BaseHTTPRequestHandler):
    """Trình xử lý HTTP cho Master."""

    def __init__(self, master_node: "NutChinh", *args, **kwargs):
        self.master = master_node
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Xử lý Health Check."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            # Trả về trạng thái của Master và danh sách worker
            status = {
                "trang_thai": "hoat_dong",
                "so_worker_dang_chay": len(self.master.danh_sach_active),
                "workers": self.master.danh_sach_active,
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Chuyển tiếp yêu cầu (Proxy/Route) đến một Worker."""
        if self.path == "/predict":
            # Chọn worker
            worker_id = self.master.can_bang_tai.chon_worker(self.master.danh_sach_active)
            if not worker_id:
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"loi": "Không có Worker nào đang hoạt động"}).encode("utf-8")
                )
                return

            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            # Chuyển tiếp request
            try:
                # worker_id là "host:port"
                req = urllib.request.Request(
                    f"http://{worker_id}/predict",
                    data=post_data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    worker_response = response.read()

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(worker_response)

            except urllib.error.URLError as e:
                # Đánh dấu worker này có thể bị lỗi (để health check xử lý sau)
                logger.warning(f"Lỗi khi gọi worker {worker_id}: {e}")
                self.send_response(502)  # Bad Gateway
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"loi": f"Worker {worker_id} không phản hồi."}).encode("utf-8")
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"loi": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def tao_master_handler(master: "NutChinh"):
    return lambda *args, **kwargs: MasterHandler(master, *args, **kwargs)


def _chay_worker(bo_xu_ly: Callable, host: str, port: int):
    """Hàm chạy riêng biệt cho process."""
    worker = NutPhu(bo_xu_ly=bo_xu_ly, host=host, port=port)
    worker.chay()


class NutChinh:
    """
    Master Node (Nút Chính).
    Điều phối cụm, auto-scaling tĩnh, health check và routing.
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 8080, chien_luoc_can_bang_tai: str = "round_robin"
    ):
        self.host = host
        self.port = port
        self.can_bang_tai = CanBangTai(chien_luoc=chien_luoc_can_bang_tai)

        self.server: Optional[HTTPServer] = None

        # Quản lý Workers
        # Dict lưu cấu hình: "host:port" -> {"bo_xu_ly": callable, "process": Process}
        self.workers_config = {}
        self.danh_sach_active: List[str] = []

        self._chay_health_check = False
        self._health_thread: Optional[threading.Thread] = None

    def them_worker(self, port: int, bo_xu_ly: Callable, host: str = "127.0.0.1"):
        """Đăng ký một worker để chạy trên port cụ thể."""
        worker_id = f"{host}:{port}"
        self.workers_config[worker_id] = {
            "bo_xu_ly": bo_xu_ly,
            "host": host,
            "port": port,
            "process": None,
        }

    def _khoi_dong_worker(self, worker_id: str):
        """Khởi động process cho worker."""
        config = self.workers_config[worker_id]
        if config["process"] is not None and config["process"].is_alive():
            return  # Đang chạy rồi

        p = multiprocessing.Process(
            target=_chay_worker, args=(config["bo_xu_ly"], config["host"], config["port"])
        )
        p.daemon = True  # Chết theo master
        p.start()
        config["process"] = p
        logger.info(f"[NutChinh] Đã khởi động Worker {worker_id} (PID: {p.pid})")

    def _kiem_tra_suc_khoe(self):
        """Background thread liên tục ping các worker."""
        while self._chay_health_check:
            active_hien_tai = []

            for worker_id, config in self.workers_config.items():
                # 1. Kiểm tra process có còn sống không
                p = config["process"]
                if p is None or not p.is_alive():
                    logger.warning(
                        f"[NutChinh] Worker {worker_id} đã chết process. Đang khởi động lại (Auto-healing)..."
                    )
                    self._khoi_dong_worker(worker_id)
                    continue

                # 2. Ping HTTP
                try:
                    req = urllib.request.Request(f"http://{worker_id}/health")
                    with urllib.request.urlopen(req, timeout=2) as response:
                        if response.status == 200:
                            active_hien_tai.append(worker_id)
                except Exception:
                    # Worker chưa khởi động xong hoặc bị treo
                    pass

            self.danh_sach_active = active_hien_tai
            time.sleep(3)  # Ping mỗi 3 giây

    def chay(self):
        """Khởi động Master và toàn bộ hệ sinh thái."""
        # Bật tất cả worker đã đăng ký
        for worker_id in self.workers_config:
            self._khoi_dong_worker(worker_id)

        # Bật Health check
        self._chay_health_check = True
        self._health_thread = threading.Thread(target=self._kiem_tra_suc_khoe, daemon=True)
        self._health_thread.start()

        # Đợi một chút để worker boot up
        time.sleep(1)

        # Bật Master HTTP Server
        handler_class = tao_master_handler(self)
        self.server = HTTPServer((self.host, self.port), handler_class)
        logger.info(f"[NutChinh] 🚀 Master Node đang hoạt động tại http://{self.host}:{self.port}")
        logger.info(f"[NutChinh] Chiến lược cân bằng tải: {self.can_bang_tai.chien_luoc}")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.dung()

    def dung(self):
        """Dừng sạch sẽ."""
        self._chay_health_check = False
        if self._health_thread:
            self._health_thread.join(timeout=2)

        if self.server:
            logger.info("[NutChinh] Tắt Master Server...")
            self.server.server_close()

        for worker_id, config in self.workers_config.items():
            p = config["process"]
            if p and p.is_alive():
                p.terminate()
                p.join(timeout=1)
                logger.info(f"[NutChinh] Đã tắt Worker {worker_id}")
