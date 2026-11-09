"""Kiểm thử Hệ thống Xác thực Zero-Trust & Tích hợp Tailscale Cụm Swarm CPU (Swarm Auth & Tailscale)."""

import socket

import numpy as np
import pytest

from vietnamese_ai.edge.swarm_auth import (
    XacThucCumSwarm,
    _la_ip_tailscale_hop_le,
    phat_hien_ip_tailscale,
    tao_ma_pin_ngau_nhien,
)
from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU, NutSwarmCPU


def test_kiem_tra_ip_tailscale_hop_le():
    """Kiểm tra nhận diện chính xác dải Carrier-Grade NAT của Tailscale (100.64.0.0/10)."""
    assert _la_ip_tailscale_hop_le("100.64.0.1") is True
    assert _la_ip_tailscale_hop_le("100.85.12.5") is True
    assert _la_ip_tailscale_hop_le("100.127.255.254") is True
    # Các IP ngoài dải Tailscale
    assert _la_ip_tailscale_hop_le("192.168.1.1") is False
    assert _la_ip_tailscale_hop_le("10.0.0.1") is False
    assert _la_ip_tailscale_hop_le("100.63.255.255") is False
    assert _la_ip_tailscale_hop_le("100.128.0.1") is False


def test_phat_hien_ip_tailscale_qua_bien_moi_truong(monkeypatch):
    """Kiểm tra phát hiện IP Tailscale từ môi trường."""
    monkeypatch.setenv("TAILSCALE_IP", "100.115.92.34")
    ip = phat_hien_ip_tailscale()
    assert ip == "100.115.92.34"


def test_tao_ma_pin_ngau_nhien():
    """Kiểm tra tính ngẫu nhiên và định dạng của mã PIN 6 số an toàn."""
    pin1 = tao_ma_pin_ngau_nhien(6)
    pin2 = tao_ma_pin_ngau_nhien(6)
    assert len(pin1) == 6
    assert len(pin2) == 6
    assert pin1.isdigit()
    assert pin2.isdigit()
    assert pin1 != pin2  # Xác suất trùng cực thấp (1/1,000,000)


def test_xac_thuc_thach_thuc_hmac_thanh_cong():
    """Kiểm tra bắt tay Thách thức - Phản hồi thành công khi đúng mã PIN."""
    auth = XacThucCumSwarm(ma_pin="829410")
    nonce = auth.tao_thach_thuc()
    chu_ky = auth.ky_thach_thuc(nonce, pin="829410")
    hop_le = auth.xac_minh_thach_thuc(nonce, chu_ky)
    assert hop_le is True

    # Cấp và xác minh Session Token
    token = auth.tao_session_token("node-test", "100.85.12.1")
    assert len(token) == 48
    assert auth.xac_minh_session_token(token) is True

    # Thu hồi token
    auth.thu_hoi_session_token(token)
    assert auth.xac_minh_session_token(token) is False


def test_xac_thuc_that_bai_khi_sai_pin():
    """Kiểm tra từ chối xác thực khi mã PIN sai hoặc giả mạo chữ ký."""
    auth = XacThucCumSwarm(ma_pin="829410")
    nonce = auth.tao_thach_thuc()
    chu_ky_sai = auth.ky_thach_thuc(nonce, pin="111222")
    assert auth.xac_minh_thach_thuc(nonce, chu_ky_sai) is False


def test_chong_tan_cong_vet_can_brute_force():
    """Kiểm tra khóa IP tự động khi thử sai mã PIN quá 5 lần."""
    auth = XacThucCumSwarm(ma_pin="999888", so_lan_sai_toi_da=5, thoi_gian_khoa_giay=60.0)
    fake_ip = "100.99.1.5"

    for _ in range(4):
        khoa = auth.ghi_nhan_that_bai(fake_ip)
        assert khoa is False
        assert auth.kiem_tra_ip_bi_khoa(fake_ip) is False

    # Lần thử thứ 5 -> Khóa IP
    khoa = auth.ghi_nhan_that_bai(fake_ip)
    assert khoa is True
    assert auth.kiem_tra_ip_bi_khoa(fake_ip) is True


@pytest.mark.asyncio
async def test_hang_doi_phe_duyet_mot_cham():
    """Kiểm tra cơ chế phê duyệt thủ công 1 chạm (One-Click Approval Queue)."""
    auth = XacThucCumSwarm(ma_pin="123456", yeu_cau_phe_duyet=True)
    fut = auth.dang_ky_cho_duyet("laptop-phong", "100.80.1.2", {"ram": 16})

    danh_sach = auth.lay_danh_sach_cho_duyet()
    assert len(danh_sach) == 1
    assert danh_sach[0]["nut_id"] == "laptop-phong"

    # Chủ máy bấm phê duyệt (Cho phép)
    duyet_xong = auth.phe_duyet_nut("laptop-phong", cho_phep=True)
    assert duyet_xong is True

    ket_qua = await fut
    assert ket_qua is True
    assert len(auth.lay_danh_sach_cho_duyet()) == 0


def _tim_cong_lan_trong() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_nut_swarm_lan_server_xac_thuc_thanh_cong():
    """Kiểm tra kết nối TCP Socket LAN thực tế giữa 2 nút có xác thực mã PIN thành công."""
    cong = _tim_cong_lan_trong()
    pin = "654321"

    # Server Node có đặt mã PIN
    server_node = NutSwarmCPU(
        node_id="server_node",
        dai_layer=(0, 4),
        ma_pin=pin,
    )
    server_node.khoi_dong_server_lan(host="127.0.0.1", port=cong, ma_pin=pin)

    # Client Node có mã PIN khớp
    client_node = NutSwarmCPU(
        node_id="client_node",
        dai_layer=(0, 4),
        endpoint_url=f"127.0.0.1:{cong}",
        che_do="mang_vat_ly_lan",
        ma_pin=pin,
    )

    try:
        x_in = np.ones((1, 128), dtype=np.float32)
        x_out, proof = client_node.thuc_thi_layer(x_in, tao_proof=True)

        assert x_out is not None
        assert x_out.shape == (1, 128)
        assert proof is not None
        assert "proof_zk" in proof
        assert "merkle_root" in proof
        assert server_node.verifier.xac_nhan_cam_ket_chuoi(proof["proof_zk"]) is True
        assert client_node.thoi_gian_tinh_toan_gan_nhat_ms >= 0
    finally:
        server_node.dung_server_lan()


def test_nut_swarm_lan_server_tu_choi_sai_pin():
    """Kiểm tra máy chủ ngắt kết nối và từ chối khi máy con cung cấp sai mã PIN."""
    cong = _tim_cong_lan_trong()

    # Server Node yêu cầu mã PIN đúng là "888999"
    server_node = NutSwarmCPU(
        node_id="server_secure",
        dai_layer=(0, 4),
        ma_pin="888999",
    )
    server_node.khoi_dong_server_lan(host="127.0.0.1", port=cong, ma_pin="888999")

    # Client Node cung cấp mã PIN sai "000000"
    client_node = NutSwarmCPU(
        node_id="client_intruder",
        dai_layer=(0, 4),
        endpoint_url=f"127.0.0.1:{cong}",
        che_do="mang_vat_ly_lan",
        ma_pin="000000",
    )

    try:
        x_in = np.ones((1, 128), dtype=np.float32)
        with pytest.raises(PermissionError) as exc_info:
            client_node.thuc_thi_layer(x_in, tao_proof=True)
        assert "Xác thực cụm Swarm thất bại" in str(exc_info.value)
    finally:
        server_node.dung_server_lan()


def test_dieu_phoi_khoi_tao_tailscale(monkeypatch):
    """Kiểm tra DieuPhoiSwarmCPU.khoi_tao_qua_tailscale tự động liên kết mạng Tailscale và cấu hình PIN."""
    monkeypatch.setenv("TAILSCALE_IP", "100.85.12.10")

    dieu_phoi = DieuPhoiSwarmCPU.khoi_tao_qua_tailscale(port=9999)
    assert dieu_phoi.ip_tailscale == "100.85.12.10"
    assert dieu_phoi.port_tailscale == 9999
    assert dieu_phoi.ma_pin is not None
    assert len(dieu_phoi.ma_pin) == 6
    assert dieu_phoi.xac_thuc is not None
