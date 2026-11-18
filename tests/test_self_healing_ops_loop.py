"""Tests kiểm tra Vòng lặp Tự phục hồi & Tự động Rollback (Self-Healing Ops Loop)."""

import numpy as np
import pytest

from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi
from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.production.circuit_breaker import MachCat
from vietnamese_ai.production.health import KiemTraSucKhoe
from vietnamese_ai.registry.model_registry import QuanLyMoHinh


class MoHinhGiaLap(BaseModel):
    def __init__(self, ten: str = "v1", gay_loi: bool = False):
        super().__init__(ten=ten)
        self.ten = ten
        self.gay_loi = gay_loi

    def huan_luyen(self, X, y=None, **kwargs):
        self.da_fit = True
        return self

    def danh_gia(self, X, y):
        return {"accuracy": 1.0}

    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        if self.gay_loi:
            raise RuntimeError("Lỗi sụp đổ mô hình runtime!")
        return f"[{self.ten}]: Phản hồi cho '{prompt}'"

    def du_doan(self, X):
        if self.gay_loi:
            raise RuntimeError("Lỗi suy luận!")
        return np.array([1])


def test_model_registry_rollback(tmp_path):
    reg = QuanLyMoHinh(str(tmp_path / "registry"))
    m1 = MoHinhGiaLap(ten="v1")
    m2 = MoHinhGiaLap(ten="v2")
    m3 = MoHinhGiaLap(ten="v3")

    reg.dang_ky(m1, ten="mo_hinh_test", version="1.0.0")
    reg.dang_ky(m2, ten="mo_hinh_test", version="1.1.0")
    reg.dang_ky(m3, ten="mo_hinh_test", version="1.2.0")

    reg.promote("mo_hinh_test", "1.2.0", "production")
    assert reg.tai("mo_hinh_test", "production").ten == "v3"

    # Rollback tự động về phiên bản trước (1.1.0)
    ver_rb, m_rb = reg.rollback("mo_hinh_test", alias_muc_tieu="production")
    assert ver_rb == "1.1.0"
    assert m_rb.ten == "v2"
    assert reg.tai("mo_hinh_test", "production").ten == "v2"

    # Rollback chỉ định đích danh về 1.0.0
    ver_rb2, m_rb2 = reg.rollback("mo_hinh_test", target_version="1.0.0", alias_muc_tieu="production")
    assert ver_rb2 == "1.0.0"
    assert m_rb2.ten == "v1"
    assert reg.tai("mo_hinh_test", "production").ten == "v1"

    # Khi đã ở phiên bản đầu tiên, rollback tự động tiếp sẽ báo lỗi
    with pytest.raises(ValueError):
        reg.rollback("mo_hinh_test", alias_muc_tieu="production")


def test_circuit_breaker_on_open_callback():
    da_goi_callback = False

    def on_open(circuit):
        nonlocal da_goi_callback
        da_goi_callback = True
        assert circuit.trang_thai == "mo"

    mc = MachCat(so_loi_toi_da=2, on_circuit_open=on_open)
    mc.ghi_nhan_loi()
    assert da_goi_callback is False
    mc.ghi_nhan_loi()
    assert da_goi_callback is True
    assert mc.trang_thai == "mo"


def test_health_check_circuit_and_unhealthy_callback():
    da_goi_unhealthy = False

    def on_unhealthy(health_checker, ket_qua):
        nonlocal da_goi_unhealthy
        da_goi_unhealthy = True

    mc = MachCat(so_loi_toi_da=1)
    ktsk = KiemTraSucKhoe(on_unhealthy=on_unhealthy)
    ktsk.dang_ky_checks_mac_dinh(mach_cat=mc)

    # Khi mạch đóng -> Healthy
    kq1 = ktsk.kiem_tra()
    assert kq1["trang_thai"] == "healthy"
    assert da_goi_unhealthy is False

    # Khi mạch ngắt -> Unhealthy & Callback tự động kích hoạt
    mc.ghi_nhan_loi()
    assert mc.trang_thai == "mo"
    kq2 = ktsk.kiem_tra()
    assert kq2["trang_thai"] == "unhealthy"
    assert da_goi_unhealthy is True


def test_self_healing_ops_loop_in_ecosystem(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="HeThongTuPhucHoi",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )

    # Đăng ký 2 phiên bản vào registry
    m_on_dinh = MoHinhGiaLap(ten="v1_on_dinh", gay_loi=False)
    m_loi = MoHinhGiaLap(ten="v2_loi", gay_loi=True)

    e2e.quan_ly_mo_hinh.dang_ky(m_on_dinh, ten="HeThongTuPhucHoi", version="1.0.0")
    e2e.quan_ly_mo_hinh.promote("HeThongTuPhucHoi", "1.0.0", "production")

    e2e.quan_ly_mo_hinh.dang_ky(m_loi, ten="HeThongTuPhucHoi", version="2.0.0")
    e2e.quan_ly_mo_hinh.promote("HeThongTuPhucHoi", "2.0.0", "production")

    # Đưa mô hình lỗi v2 vào hoạt động
    e2e.mo_hinh_hien_tai = m_loi
    e2e._ten_mo_hinh_dang_ky = "HeThongTuPhucHoi"

    # Gọi suy luận liên tiếp: CircuitBreaker phát hiện lỗi dồn dập và tự kích hoạt rollback về v1.0.0
    for _ in range(3):
        e2e.hoi("Kiểm tra khả năng phục hồi")

    # Mô hình phải được tự động chuyển về v1_on_dinh
    assert e2e.mo_hinh_hien_tai is not None
    assert getattr(e2e.mo_hinh_hien_tai, "ten", "") == "v1_on_dinh"
    assert e2e.mach_cat.trang_thai == "dong"

    # Lần gọi tiếp theo diễn ra hoàn toàn bình thường
    kq = e2e.hoi("Xin chào")
    assert "[v1_on_dinh]" in kq["tra_loi"]


def test_proactive_self_healing_api(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="HeThongProactive",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
    )

    m1 = MoHinhGiaLap(ten="v1", gay_loi=False)
    m2 = MoHinhGiaLap(ten="v2", gay_loi=True)

    e2e.quan_ly_mo_hinh.dang_ky(m1, ten="HeThongProactive", version="1.0.0")
    e2e.quan_ly_mo_hinh.promote("HeThongProactive", "1.0.0", "production")

    e2e.quan_ly_mo_hinh.dang_ky(m2, ten="HeThongProactive", version="2.0.0")
    e2e.quan_ly_mo_hinh.promote("HeThongProactive", "2.0.0", "production")

    e2e.mo_hinh_hien_tai = m2
    e2e._ten_mo_hinh_dang_ky = "HeThongProactive"

    # Giả lập một health check thất bại khiến hệ sinh thái chuyển sang unhealthy
    e2e.suc_khoe.dang_ky_check("check_tai_nguyen", lambda: False, quan_trong=True)

    # Gọi kiểm tra & tự động phục hồi chủ động
    kq_phuc_hoi = e2e.tu_dong_phuc_hoi()
    assert kq_phuc_hoi["da_phuc_hoi"] is True
    assert kq_phuc_hoi["chi_tiet_phuc_hoi"]["version"] == "1.0.0"
    assert getattr(e2e.mo_hinh_hien_tai, "ten", "") == "v1"
