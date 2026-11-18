"""
Kiểm thử tích hợp Cầu nối Cluster Orchestrator (NutChinh HTTP) và Distributed CPU Swarm (NutSwarmCPU / DieuPhoiSwarmCPU).
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU
from vietnamese_ai.orchestrator.master import NutChinh, tao_master_handler
from vietnamese_ai.orchestrator.swarm_bridge import NutCauNoiSwarm, SwarmBridgeWorker


def _tim_port_trong() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestNutCauNoiSwarm:
    def test_khoi_tao_va_xu_ly_payload(self):
        """Kiểm tra khởi tạo NutCauNoiSwarm và hàm xử lý payload."""
        swarm = DieuPhoiSwarmCPU(tong_so_layer=4, bat_xac_thuc_mat_ma=True)
        swarm.khoi_tao_cum_gia_lap(so_node=2, so_core_moi_node=2)

        bridge = NutCauNoiSwarm(swarm=swarm, port=18090)
        assert bridge.swarm is swarm
        assert bridge.port == 18090
        assert SwarmBridgeWorker is NutCauNoiSwarm

        # Kiểm tra xử lý payload
        payload = {"prompt": "Trí tuệ nhân tạo Việt Nam", "so_token_sinh": 8, "xac_minh_proof": True}
        kq = bridge.xu_ly_payload(payload)
        assert "cau_tra_loi" in kq
        assert kq["so_proof_da_xac_minh"] >= 1
        assert len(kq["danh_sach_node"]) == 2

    def test_http_health_va_predict_cua_bridge(self):
        """Kiểm tra máy chủ HTTP của NutCauNoiSwarm qua các endpoint GET /health và POST /predict."""
        port = _tim_port_trong()
        swarm = DieuPhoiSwarmCPU(tong_so_layer=4, bat_xac_thuc_mat_ma=True)
        swarm.khoi_tao_cum_gia_lap(so_node=2)

        bridge = NutCauNoiSwarm(swarm=swarm, host="127.0.0.1", port=port, ma_bi_mat_cum="bi_mat_123")
        bridge.khoi_dong_ngam()
        assert bridge.dang_chay is True

        try:
            # 1. GET /health
            req_health = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req_health, timeout=3) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["trang_thai"] == "hoat_dong"
                assert data["loai"] == "swarm"
                assert data["so_node"] == 2
                assert data["tong_so_layer"] == 4

            # 2. POST /predict không có hoặc sai mã bảo mật -> 401
            req_unauth = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict",
                data=json.dumps({"prompt": "Hello"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req_unauth, timeout=3)
            assert exc_info.value.code == 401

            # 3. POST /predict có mã bảo mật hợp lệ
            req_auth = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict",
                data=json.dumps({"prompt": "Học máy tiếng Việt", "so_token_sinh": 4}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Cluster-Token": "bi_mat_123",
                },
            )
            with urllib.request.urlopen(req_auth, timeout=5) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["thanh_cong"] is True
                assert data["loai"] == "swarm"
                assert "ket_qua" in data
                assert "cau_tra_loi" in data["ket_qua"]

        finally:
            bridge.dung()
            assert bridge.dang_chay is False


class TestNutChinhSwarmIntegration:
    def test_them_nut_swarm_va_lay_thong_tin(self):
        """Kiểm tra NutChinh đăng ký Swarm Worker và quản lý trạng thái cụm."""
        port_swarm = _tim_port_trong()
        master = NutChinh(port=18080, bat_tuong_lua=False)

        swarm = DieuPhoiSwarmCPU(tong_so_layer=6, bat_xac_thuc_mat_ma=True)
        swarm.khoi_tao_cum_gia_lap(so_node=3)

        worker_id = master.them_nut_swarm(port=port_swarm, swarm=swarm, chay_ngam=True)
        assert worker_id == f"127.0.0.1:{port_swarm}"
        assert worker_id in master.workers_config
        assert master.workers_config[worker_id]["loai"] == "swarm"
        assert worker_id in master.danh_sach_swarm_active

        # Lấy thông tin trạng thái toàn bộ Swarm
        tt = master.lay_thong_tin_tat_ca_swarm()
        assert worker_id in tt
        assert tt[worker_id]["so_node"] == 3
        assert tt[worker_id]["tong_so_layer"] == 6

        # Điều hướng suy luận trực tiếp qua NutChinh
        res = master.dieu_huong_suy_luan_swarm("Phát triển AI cho Việt Nam", so_token_sinh=4)
        assert "cau_tra_loi" in res
        assert res["so_proof_da_xac_minh"] >= 1

        master.dung()

    def test_master_http_routing_to_swarm(self):
        """Kiểm tra Master HTTP Handler định tuyến /swarm/status và /swarm/predict."""
        port_master = _tim_port_trong()
        port_swarm = _tim_port_trong()

        master = NutChinh(port=port_master, bat_tuong_lua=False)
        swarm = DieuPhoiSwarmCPU(tong_so_layer=4, bat_xac_thuc_mat_ma=True)
        swarm.khoi_tao_cum_gia_lap(so_node=2)

        master.them_nut_swarm(port=port_swarm, swarm=swarm, chay_ngam=True)

        handler_cls = tao_master_handler(master)
        server = HTTPServer(("127.0.0.1", port_master), handler_cls)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            # 1. GET /health
            with urllib.request.urlopen(f"http://127.0.0.1:{port_master}/health", timeout=3) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["so_worker_swarm"] == 1
                assert f"127.0.0.1:{port_swarm}" in data["workers_swarm"]

            # 2. GET /swarm/status
            with urllib.request.urlopen(f"http://127.0.0.1:{port_master}/swarm/status", timeout=3) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["so_worker_swarm"] == 1
                assert "thong_tin_swarm" in data

            # 3. POST /swarm/predict (Định tuyến chuyên biệt Swarm)
            payload = json.dumps({"prompt": "Mạng Nơ-ron Biên", "so_token_sinh": 4}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port_master}/swarm/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["thanh_cong"] is True
                assert data["loai"] == "swarm"
                assert "cau_tra_loi" in data["ket_qua"]

            # 4. POST /predict với header X-Route: swarm
            req_header = urllib.request.Request(
                f"http://127.0.0.1:{port_master}/predict",
                data=payload,
                headers={"Content-Type": "application/json", "X-Route": "swarm"},
            )
            with urllib.request.urlopen(req_header, timeout=5) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["thanh_cong"] is True
                assert data["loai"] == "swarm"

        finally:
            server.shutdown()
            server.server_close()
            master.dung()

    def test_dieu_phoi_swarm_dang_ky_vao_nut_chinh(self):
        """Kiểm tra API tiện ích 1-click: swarm.dang_ky_vao_nut_chinh(master)."""
        port_swarm = _tim_port_trong()
        master = NutChinh(port=18080, bat_tuong_lua=False)

        swarm = DieuPhoiSwarmCPU(tong_so_layer=6)
        swarm.khoi_tao_cum_gia_lap(so_node=2)

        # 1-click đăng ký
        w_id = swarm.dang_ky_vao_nut_chinh(master, port_http=port_swarm)
        assert w_id == f"127.0.0.1:{port_swarm}"
        assert w_id in master.danh_sach_swarm_active

        # Tạo cầu nối độc lập qua tao_cau_noi_orchestrator
        bridge = swarm.tao_cau_noi_orchestrator(port_http=18099)
        assert isinstance(bridge, NutCauNoiSwarm)
        assert bridge.swarm is swarm
        assert bridge.port == 18099

        master.dung()
