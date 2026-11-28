"""
NutChinh (Master Node / DieuPhoiCum) - Quản lý Workers, cân bằng tải, Auto-healing,
kết nối Tường lửa AI Đa tầng và xác thực khóa cụm HMAC.
"""

from __future__ import annotations

import hmac
import json
import logging
import multiprocessing
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from .balancer import CanBangTai
from .worker import NutPhu

logger = logging.getLogger("V-Orchestrator")

MAX_CLUSTER_BODY_BYTES = 1_048_576  # 1 MB chống tràn RAM DoS


class MasterHandler(BaseHTTPRequestHandler):
    """Trình xử lý HTTP cho Master có tích hợp Tường lửa AI và xác thực cụm."""

    def __init__(self, master_node: "NutChinh", *args, **kwargs):
        self.master = master_node
        super().__init__(*args, **kwargs)

    def _tra_json(self, ma_http: int, du_lieu: Dict[str, Any]) -> None:
        self.send_response(ma_http)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(du_lieu, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        """Xử lý Health Check và Swarm Status."""
        if self.path == "/health":
            status = {
                "trang_thai": "hoat_dong",
                "so_worker_dang_chay": len(self.master.danh_sach_active),
                "workers": self.master.danh_sach_active,
                "so_worker_swarm": len(self.master.danh_sach_swarm_active),
                "workers_swarm": self.master.danh_sach_swarm_active,
                "bao_mat_tuong_lua": self.master.bat_tuong_lua,
            }
            self._tra_json(200, status)
        elif self.path in ("/swarm/status", "/swarm"):
            self._tra_json(
                200,
                {
                    "trang_thai": "hoat_dong",
                    "so_worker_swarm": len(self.master.danh_sach_swarm_active),
                    "workers_swarm": self.master.danh_sach_swarm_active,
                    "thong_tin_swarm": self.master.lay_thong_tin_tat_ca_swarm(),
                },
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Kiểm duyệt bảo mật và chuyển tiếp yêu cầu (Proxy/Route) đến Worker hoặc Swarm."""
        if self.path in ("/predict", "/swarm/predict"):
            hop_le_auth, ly_do_auth = self.master.kiem_tra_xac_thuc_header(
                self.headers.get("Authorization", ""),
                self.headers.get("X-Cluster-Token", ""),
            )
            if not hop_le_auth:
                self._tra_json(401, {"loi": ly_do_auth})
                return

            raw_content_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_content_length)
            except (TypeError, ValueError):
                self._tra_json(400, {"loi": "Header Content-Length không hợp lệ"})
                return
            if content_length < 0:
                self._tra_json(400, {"loi": "Header Content-Length không được âm"})
                return
            if content_length > MAX_CLUSTER_BODY_BYTES:
                self._tra_json(
                    413, {"loi": f"Payload vượt quá giới hạn an toàn ({MAX_CLUSTER_BODY_BYTES} bytes)"}
                )
                return

            post_data = self.rfile.read(max(0, content_length))
            hop_le_fw, ly_do_fw = self.master.kiem_tra_payload_dau_vao(
                post_data, nguon=self.client_address[0] if self.client_address else "cluster_client"
            )
            if not hop_le_fw:
                self._tra_json(403, {"loi": f"Tường lửa AI chặn yêu cầu: {ly_do_fw}"})
                return

            # Kiểm tra yêu cầu định tuyến Swarm
            la_yeu_cau_swarm = (
                self.path == "/swarm/predict"
                or self.headers.get("X-Route", "").strip().lower() == "swarm"
            )
            if not la_yeu_cau_swarm and post_data:
                try:
                    payload_check = json.loads(post_data.decode("utf-8"))
                    if payload_check.get("che_do") == "swarm":
                        la_yeu_cau_swarm = True
                except Exception:
                    pass

            if la_yeu_cau_swarm:
                danh_sach_swarm = self.master.danh_sach_swarm_active
                if not danh_sach_swarm:
                    self._tra_json(503, {"loi": "Không có Nút Swarm nào đang hoạt động trong cụm"})
                    return
                worker_id = self.master.can_bang_tai.chon_worker(danh_sach_swarm)
            else:
                worker_id = self.master.can_bang_tai.chon_worker(self.master.danh_sach_active)

            if not worker_id:
                self._tra_json(503, {"loi": "Không có Worker nào đang hoạt động"})
                return

            try:
                headers = {"Content-Type": "application/json"}
                if self.master.ma_bi_mat_cum:
                    headers["X-Cluster-Token"] = self.master.ma_bi_mat_cum
                req = urllib.request.Request(
                    f"http://{worker_id}/predict",
                    data=post_data,
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    worker_response = response.read()

                worker_clean = self.master.loc_phan_hoi_dau_ra(worker_response)
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(worker_clean)

            except urllib.error.URLError as e:
                logger.warning(f"Lỗi khi gọi worker {worker_id}: {e}")
                self._tra_json(502, {"loi": f"Worker {worker_id} không phản hồi."})
            except Exception as e:
                self._tra_json(500, {"loi": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def tao_master_handler(master: "NutChinh"):
    return lambda *args, **kwargs: MasterHandler(master, *args, **kwargs)


def _chay_worker(
    bo_xu_ly: Callable,
    host: str,
    port: int,
    ma_bi_mat_cum: Optional[str] = None,
):
    """Hàm chạy riêng biệt cho process Worker."""
    worker = NutPhu(bo_xu_ly=bo_xu_ly, host=host, port=port, ma_bi_mat_cum=ma_bi_mat_cum)
    worker.chay()


class NutChinh:
    """
    Master Node (Nút Chính / `DieuPhoiCum`).
    Điều phối cụm, auto-healing, kiểm duyệt Tường lửa AI và xác thực khóa cụm HMAC.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        chien_luoc_can_bang_tai: str = "round_robin",
        bat_tuong_lua: bool = True,
        ma_bi_mat_cum: Optional[str] = None,
    ):
        from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

        self.host = host
        self.port = port
        self.can_bang_tai = CanBangTai(chien_luoc=chien_luoc_can_bang_tai)
        self.bat_tuong_lua = bat_tuong_lua
        self.ma_bi_mat_cum = ma_bi_mat_cum
        self.tuong_lua = TuongLuaBaoVeFramework() if bat_tuong_lua else None

        self.server: Optional[ThreadingHTTPServer] = None
        self.workers_config: Dict[str, Dict[str, Any]] = {}
        self.danh_sach_active: List[str] = []

        self._chay_health_check = False
        self._health_thread: Optional[threading.Thread] = None

    def kiem_tra_xac_thuc_header(
        self, auth_header: str = "", cluster_token: str = ""
    ) -> Tuple[bool, str]:
        """Kiểm tra khóa bí mật cụm (`Bearer <token>` hoặc `X-Cluster-Token`) bằng `hmac.compare_digest`."""
        if not self.ma_bi_mat_cum:
            return True, "OK"
        ung_vien = cluster_token.strip()
        if not ung_vien and auth_header.lower().startswith("bearer "):
            ung_vien = auth_header[7:].strip()
        if ung_vien and hmac.compare_digest(ung_vien, self.ma_bi_mat_cum):
            return True, "OK"
        return False, "Thiếu hoặc sai mã xác thực cụm (X-Cluster-Token / Bearer)"

    def kiem_tra_payload_dau_vao(self, raw_bytes: bytes, nguon: str = "client") -> Tuple[bool, str]:
        """Quét JSON payload qua `TuongLuaBaoVeFramework` trước khi chuyển tiếp xuống Worker."""
        if len(raw_bytes) > MAX_CLUSTER_BODY_BYTES:
            return False, "Payload vượt quá 1 MB"
        if self.tuong_lua is None or not raw_bytes:
            return True, "OK"
        try:
            text_body = raw_bytes.decode("utf-8", errors="ignore")
            kq = self.tuong_lua.kiem_tra_dau_vao(text_body, client_id=nguon)
            return bool(kq["an_toan"]), str(kq["ly_do"])
        except Exception as e:
            return False, str(e)

    def loc_phan_hoi_dau_ra(self, raw_bytes: bytes) -> bytes:
        """Lọc DLP / ẩn khóa bí mật và PII rò rỉ trong kết quả trả về từ Worker."""
        if self.tuong_lua is None or not raw_bytes:
            return raw_bytes
        try:
            decoded = raw_bytes.decode("utf-8")
            cleaned = self.tuong_lua.loc_dau_ra(decoded)
            return cleaned.encode("utf-8")
        except Exception:
            return raw_bytes

    def them_worker(self, port: int, bo_xu_ly: Callable, host: str = "127.0.0.1"):
        """Đăng ký một worker để chạy trên port cụ thể."""
        worker_id = f"{host}:{port}"
        self.workers_config[worker_id] = {
            "bo_xu_ly": bo_xu_ly,
            "host": host,
            "port": port,
            "process": None,
            "loai": "standard",
        }

    def them_nut_swarm(
        self,
        port: int,
        swarm: Optional[Any] = None,
        host: str = "127.0.0.1",
        trong_so_canary: Optional[float] = None,
        chay_ngam: bool = True,
    ) -> str:
        """
        Đăng ký một Cụm Swarm P2P (Distributed CPU Swarm) làm Worker trong cụm NutChinh.
        Tự động tạo Cầu nối NutCauNoiSwarm (HTTP -> TCP Swarm Pipeline).
        """
        from .swarm_bridge import NutCauNoiSwarm

        bridge = NutCauNoiSwarm(
            swarm=swarm,
            host=host,
            port=port,
            ma_bi_mat_cum=self.ma_bi_mat_cum,
        )
        worker_id = f"{host}:{port}"
        self.workers_config[worker_id] = {
            "bo_xu_ly": bridge.xu_ly_payload,
            "host": host,
            "port": port,
            "process": None,
            "loai": "swarm",
            "bridge": bridge,
        }
        if trong_so_canary is not None and self.can_bang_tai.chien_luoc == "canary":
            moi = dict(self.can_bang_tai.trong_so_canary)
            moi[worker_id] = float(trong_so_canary)
            self.can_bang_tai.cap_nhat_trong_so(moi)

        if chay_ngam:
            bridge.khoi_dong_ngam()
            if worker_id not in self.danh_sach_active:
                self.danh_sach_active.append(worker_id)

        logger.info(f"[NutChinh] Đã liên kết Swarm Worker {worker_id} (loại: P2P Pipeline)")
        return worker_id

    @property
    def danh_sach_swarm_active(self) -> List[str]:
        """Danh sách các Worker Swarm đang hoạt động."""
        return [
            w_id
            for w_id in self.danh_sach_active
            if self.workers_config.get(w_id, {}).get("loai") == "swarm"
        ]

    def lay_thong_tin_tat_ca_swarm(self) -> Dict[str, Any]:
        """Thu thập trạng thái toàn bộ cụm Swarm kết nối vào NutChinh."""
        res: Dict[str, Any] = {}
        for w_id in self.danh_sach_swarm_active:
            cfg = self.workers_config.get(w_id, {})
            bridge = cfg.get("bridge")
            if bridge and hasattr(bridge, "swarm"):
                res[w_id] = (
                    bridge.swarm.lay_trang_thai_cum()
                    if hasattr(bridge.swarm, "lay_trang_thai_cum")
                    else {}
                )
        return res

    def dieu_huong_suy_luan_swarm(
        self,
        prompt: str,
        so_token_sinh: int = 16,
        xac_minh_proof: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực thi suy luận phân tán Swarm trực tiếp thông qua NutChinh (In-memory Direct Call).
        """
        for w_id in self.danh_sach_swarm_active:
            cfg = self.workers_config.get(w_id, {})
            bridge = cfg.get("bridge")
            if bridge and hasattr(bridge, "swarm"):
                return bridge.swarm.suy_luan_swarm(
                    prompt=prompt,
                    so_token_sinh=so_token_sinh,
                    xac_minh_proof=xac_minh_proof,
                )
        raise RuntimeError("Không có Worker Swarm nào đang kết nối trong NutChinh")

    def dieu_phoi_nhiem_vu(self, nhiem_vu: Dict[str, Any]) -> Dict[str, Any]:
        """Điều phối và thực thi một nhiệm vụ trực tiếp qua Worker hoạt động hoặc Cụm Swarm."""
        task_id = str(nhiem_vu.get("id", "task"))
        payload = nhiem_vu.get("data", nhiem_vu)

        if self.danh_sach_swarm_active:
            kq_swarm = self.dieu_huong_suy_luan_swarm(prompt=str(payload))
            return {
                "task_id": task_id,
                "worker": self.danh_sach_swarm_active[0],
                "loai": "swarm",
                "ket_qua": kq_swarm,
            }

        ung_vien = list(self.danh_sach_active) or list(self.workers_config.keys())
        worker_id = self.can_bang_tai.chon_worker(ung_vien)
        if not worker_id or worker_id not in self.workers_config:
            raise RuntimeError(
                "Không có Worker nào được đăng ký trong NutChinh để điều phối nhiệm vụ."
            )

        if self.tuong_lua is not None:
            raw_bytes = json.dumps({"data": payload}, ensure_ascii=False).encode("utf-8")
            hop_le, ly_do = self.kiem_tra_payload_dau_vao(raw_bytes, nguon="state_graph")
            if not hop_le:
                raise PermissionError(f"Tường lửa AI chặn nhiệm vụ cụm: {ly_do}")

        bo_xu_ly = self.workers_config[worker_id]["bo_xu_ly"]
        ket_qua = bo_xu_ly(payload)
        return {
            "task_id": task_id,
            "worker": worker_id,
            "loai": self.workers_config[worker_id].get("loai", "standard"),
            "ket_qua": ket_qua,
        }

    def _khoi_dong_worker(self, worker_id: str):
        """Khởi động process hoặc thread cho worker."""
        config = self.workers_config[worker_id]
        if config.get("loai") == "swarm":
            bridge = config.get("bridge")
            if bridge is not None:
                if not bridge.dang_chay:
                    bridge.khoi_dong_ngam()
                return

        if config["process"] is not None and config["process"].is_alive():
            return

        worker_args = (
            (config["bo_xu_ly"], config["host"], config["port"], self.ma_bi_mat_cum)
            if self.ma_bi_mat_cum is not None
            else (config["bo_xu_ly"], config["host"], config["port"])
        )
        p = multiprocessing.Process(target=_chay_worker, args=worker_args)
        p.daemon = True
        p.start()
        config["process"] = p
        logger.info(f"[NutChinh] Đã khởi động Worker {worker_id} (PID: {p.pid})")

    def _kiem_tra_suc_khoe(self):
        """Background thread liên tục ping các worker."""
        while self._chay_health_check:
            active_hien_tai = []

            for worker_id, config in self.workers_config.items():
                if config.get("loai") == "swarm":
                    bridge = config.get("bridge")
                    if bridge is not None and not bridge.dang_chay:
                        logger.warning(
                            f"[NutChinh] Swarm Worker {worker_id} đã tắt. Đang khởi động lại (Auto-healing)..."
                        )
                        self._khoi_dong_worker(worker_id)
                        continue
                else:
                    p = config.get("process")
                    if p is None or not p.is_alive():
                        logger.warning(
                            f"[NutChinh] Worker {worker_id} đã chết process. Đang khởi động lại (Auto-healing)..."
                        )
                        self._khoi_dong_worker(worker_id)
                        continue

                try:
                    req = urllib.request.Request(f"http://{worker_id}/health")
                    with urllib.request.urlopen(req, timeout=2) as response:
                        if response.status == 200:
                            active_hien_tai.append(worker_id)
                except Exception:
                    pass

            self.danh_sach_active = active_hien_tai
            time.sleep(3)

    def chay(self):
        """Khởi động Master và toàn bộ hệ sinh thái."""
        for worker_id in self.workers_config:
            self._khoi_dong_worker(worker_id)

        self._chay_health_check = True
        self._health_thread = threading.Thread(target=self._kiem_tra_suc_khoe, daemon=True)
        self._health_thread.start()

        time.sleep(1)

        handler_class = tao_master_handler(self)
        self.server = ThreadingHTTPServer((self.host, self.port), handler_class)
        logger.info(f"[NutChinh] Master Node đang hoạt động tại http://{self.host}:{self.port}")
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
            if config.get("loai") == "swarm" and config.get("bridge"):
                try:
                    config["bridge"].dung()
                except Exception:
                    pass
            p = config.get("process")
            if p and p.is_alive():
                p.terminate()
                p.join(timeout=1)
                logger.info(f"[NutChinh] Đã tắt Worker {worker_id}")
