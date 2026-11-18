"""
NutCauNoiSwarm (SwarmBridgeWorker / Cầu nối Swarm - Orchestrator).
Cầu nối giao thức chuyển tiếp yêu cầu HTTP REST từ NutChinh (V-Orchestrator)
sang mạng suy luận phân tán Pipeline Parallelism CPU Swarm (NutSwarmCPU / DieuPhoiSwarmCPU).
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

logger = logging.getLogger("V-Orchestrator.SwarmBridge")

MAX_SWARM_BODY_BYTES = 2_097_152  # 2 MB cho các batch prompts hoặc tensor metadata


class SwarmBridgeHandler(BaseHTTPRequestHandler):
    """Trình xử lý HTTP cho Cầu nối Swarm Worker."""

    def __init__(self, bridge: NutCauNoiSwarm, *args, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)

    def _tra_json(self, ma_http: int, du_lieu: Dict[str, Any]) -> None:
        self.send_response(ma_http)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(du_lieu, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        """Health check và thông tin trạng thái cụm Swarm."""
        if self.path == "/health":
            thong_tin_cum = (
                self.bridge.swarm.lay_trang_thai_cum()
                if hasattr(self.bridge.swarm, "lay_trang_thai_cum")
                else {}
            )
            so_node = len(getattr(self.bridge.swarm, "danh_sach_node", []))
            tong_so_layer = getattr(self.bridge.swarm, "tong_so_layer", 0)
            self._tra_json(
                200,
                {
                    "trang_thai": "hoat_dong",
                    "loai": "swarm",
                    "so_node": so_node,
                    "tong_so_layer": tong_so_layer,
                    "cum_swarm": thong_tin_cum,
                },
            )
        elif self.path in ("/status", "/swarm/status"):
            thong_tin_cum = (
                self.bridge.swarm.lay_trang_thai_cum()
                if hasattr(self.bridge.swarm, "lay_trang_thai_cum")
                else {}
            )
            self._tra_json(
                200,
                {
                    "trang_thai": "hoat_dong",
                    "thong_tin": thong_tin_cum,
                },
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Xử lý yêu cầu suy luận phân tán qua cụm Swarm CPU."""
        if self.path in ("/predict", "/swarm/predict"):
            if self.bridge.ma_bi_mat_cum is not None:
                token_gui = self.headers.get("X-Cluster-Token", "")
                if not token_gui or not hmac.compare_digest(token_gui, self.bridge.ma_bi_mat_cum):
                    self._tra_json(401, {"thanh_cong": False, "loi": "Từ chối truy cập: Khóa cụm không hợp lệ"})
                    return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_SWARM_BODY_BYTES:
                self._tra_json(
                    413,
                    {"thanh_cong": False, "loi": f"Payload vượt quá giới hạn an toàn ({MAX_SWARM_BODY_BYTES} bytes)"},
                )
                return

            post_data = self.rfile.read(max(0, content_length))
            try:
                data = json.loads(post_data.decode("utf-8")) if post_data else {}
                ket_qua = self.bridge.xu_ly_payload(data)
                self._tra_json(200, {"thanh_cong": True, "loai": "swarm", "ket_qua": ket_qua})
            except Exception as e:
                logger.error(f"Lỗi suy luận Swarm Bridge: {e}")
                self._tra_json(500, {"thanh_cong": False, "loi": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class NutCauNoiSwarm:
    """
    Cầu nối Worker Swarm (Swarm Bridge Worker Node):
    Đóng vai trò là một HTTP Worker hợp lệ trong cụm NutChinh,
    nhưng chuyển hướng tính toán suy luận sang cụm P2P Swarm CPU (Pipeline Parallelism).
    """

    def __init__(
        self,
        swarm: Optional[Any] = None,
        host: str = "127.0.0.1",
        port: int = 8085,
        ma_bi_mat_cum: Optional[str] = None,
        so_node_mac_dinh: int = 3,
        tong_so_layer: int = 6,
    ):
        if swarm is None:
            from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU

            swarm = DieuPhoiSwarmCPU(tong_so_layer=tong_so_layer)
            swarm.khoi_tao_cum_gia_lap(so_node=so_node_mac_dinh)

        self.swarm = swarm
        self.host = host
        self.port = port
        self.ma_bi_mat_cum = ma_bi_mat_cum

        self.server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._dang_chay = False
        self._ready_event = threading.Event()

    @property
    def dang_chay(self) -> bool:
        """Kiểm tra máy chủ cầu nối có đang hoạt động hay không."""
        return self._dang_chay

    def xu_ly_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trích xuất và điều phối suy luận Swarm từ JSON payload gửi tới HTTP Worker.
        """
        prompt = (
            data.get("prompt")
            or data.get("van_ban")
            or data.get("text")
            or data.get("cau_hoi")
            or ""
        )
        so_token = int(data.get("so_token_sinh") or data.get("so_token") or data.get("tokens") or 16)
        xac_minh = bool(data.get("xac_minh_proof", data.get("verify", True)))

        return self.swarm.suy_luan_swarm(
            prompt=str(prompt),
            so_token_sinh=so_token,
            xac_minh_proof=xac_minh,
        )

    def khoi_dong_ngam(self) -> None:
        """Khởi động máy chủ HTTP Bridge trong luồng ngầm (Daemon Thread)."""
        if self._dang_chay:
            return

        self._ready_event.clear()

        def _worker():
            def handler(*args, **kwargs):
                return SwarmBridgeHandler(self, *args, **kwargs)

            try:
                self.server = HTTPServer((self.host, self.port), handler)
                self._dang_chay = True
                self._ready_event.set()
                self.server.serve_forever()
            except Exception:
                pass
            finally:
                self._dang_chay = False
                self._ready_event.set()

        self._server_thread = threading.Thread(target=_worker, daemon=True)
        self._server_thread.start()
        self._ready_event.wait(timeout=3.0)

    def chay(self) -> None:
        """Khởi động máy chủ HTTP Bridge (chặn - blocking, cho tiến trình độc lập)."""
        def handler(*args, **kwargs):
            return SwarmBridgeHandler(self, *args, **kwargs)

        self.server = HTTPServer((self.host, self.port), handler)
        self._dang_chay = True
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.dung()
        finally:
            self._dang_chay = False

    def dung(self) -> None:
        """Dừng sạch sẽ máy chủ HTTP Bridge."""
        self._dang_chay = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            self.server = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
            self._server_thread = None


# Aliases
SwarmBridgeWorker = NutCauNoiSwarm
