"""Tests kiểm tra Vòng lặp Tự động Đóng gói Edge & Phân phối Mô hình (Edge Swarm Packaging Loop)."""

from pathlib import Path

import numpy as np

from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi
from vietnamese_ai.models.base import BaseModel


class MoHinhKiemThu(BaseModel):
    def __init__(self, ten: str = "TestModel"):
        super().__init__(ten=ten)
        self.ten = ten
        self.tok_emb = np.random.randn(32, 128).astype(np.float32)
        self.pos_emb = np.random.randn(32, 128).astype(np.float32)
        self.lm_head = np.random.randn(32, 128).astype(np.float32)

    def huan_luyen(self, X, y=None, **kwargs):
        self.da_fit = True
        return self

    def danh_gia(self, X, y):
        return {"accuracy": 1.0}

    def du_doan(self, X):
        return np.array([1])

    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        return f"[TestModel]: Phản hồi cho '{prompt}'"


def test_dong_goi_va_phan_phoi_edge_swarm(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="TroLyEdgeSwarm",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )
    e2e.mo_hinh_hien_tai = MoHinhKiemThu()
    e2e.tao_tac_tu()

    # Tự động đóng gói GGUF và phân mảnh sang 3 node CPU
    kq = e2e.dong_goi_va_phan_phoi_edge(so_node=3, quantization="q8_0")

    assert kq["trang_thai"] == "thanh_cong"
    assert kq["so_node"] == 3
    assert Path(kq["duong_dan_gguf"]).exists()
    assert e2e.dieu_phoi_swarm is not None
    assert len(e2e.dieu_phoi_swarm.danh_sach_node) == 3

    # Kiểm tra tool đã được tự động thêm vào TacTu
    assert "suy_luan_edge_swarm" in e2e.tac_tu.cong_cu


def test_suy_luan_edge_va_hoi_routing(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="EdgeInference",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )
    e2e.mo_hinh_hien_tai = MoHinhKiemThu()

    # Suy luận trực tiếp qua edge swarm
    kq_edge = e2e.suy_luan_edge("Học máy tiếng Việt", so_token=8)
    assert "cau_tra_loi" in kq_edge
    assert kq_edge["tokens_moi_giay"] >= 0.0
    assert len(kq_edge["danh_sach_node"]) > 0

    # Gọi qua hoi() với cờ dung_edge=True
    kq_hoi = e2e.hoi("Kiểm tra edge", dung_edge=True)
    assert kq_hoi["edge"] is True
    assert "Kiểm tra edge" in kq_hoi["tra_loi"]


def test_chay_dau_cuoi_voi_phan_phoi_edge(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="E2EEdgeLoop",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )
    du_lieu = [
        {"instruction": "Hà Nội là gì?", "output": "Hà Nội là thủ đô của Việt Nam."},
        {"instruction": "TP.HCM là gì?", "output": "TP.HCM là trung tâm kinh tế phía Nam."},
    ]

    bao_cao = e2e.chay_dau_cuoi(
        du_lieu=du_lieu,
        mo_hinh="vgpt2",
        so_vong=2,
        phan_phoi_edge=True,
        so_node_edge=2,
    )

    assert "edge_swarm" in bao_cao
    assert bao_cao["edge_swarm"]["trang_thai"] == "thanh_cong"
    assert bao_cao["edge_swarm"]["so_node"] == 2
    assert e2e.dieu_phoi_swarm is not None


def test_fastapi_server_swarm_endpoints(tmp_path):
    from vietnamese_ai.api.fastapi_server import ChatCompletionRequest, ChatMessage, FastAPIServer

    e2e = QuyTrinhDauCuoi(
        ten_du_an="APISwarmTest",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )
    e2e.mo_hinh_hien_tai = MoHinhKiemThu()
    e2e.dong_goi_va_phan_phoi_edge(so_node=2)

    server = FastAPIServer(bo_xu_ly=e2e)

    # Kiểm tra xử lý logic với model="swarm"
    req_swarm = ChatCompletionRequest(
        model="swarm",
        messages=[ChatMessage(role="user", content="Chào cụm Swarm CPU")],
    )
    res_text = server._xu_ly_logic(req_swarm)
    assert "Chào cụm Swarm CPU" in res_text
