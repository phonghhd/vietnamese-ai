"""
Bộ kiểm thử toàn diện cho Plugin Tài Chính - Chứng Khoán (VN Stocks) & Crypto Real-time.
Kiểm tra:
1. Bộ thu thập & phát giá giả lập (Market Feed Engine & Deterministic Mock).
2. Bộ tính toán chỉ báo kỹ thuật Pure NumPy (RSI, MACD, Bollinger Bands, EMA, ATR, Pivot Points).
3. Định cỡ vị thế quản trị vốn (2% Rule & Position Sizing).
4. Phân tích tâm lý thị trường bản địa (F319/Telegram/CafeF sentiment).
5. Tích hợp Tool Calling với Tác tử AI (TacTu).
6. Tích hợp RAG Pipeline với tri thức quy chế TTCK (HOSE/HNX/UPCoM).
7. Nạp Hot-Swap động qua QuanLyPlugin và kiểm duyệt an ninh AST.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.plugin.market_feed import (
    BoDieuPhoiThiTruong,
    BoPhatGiaGiaLap,
    chuan_hoa_ma,
    la_ma_crypto,
)
from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin
from vietnamese_ai.plugin.quant_indicators import (
    phan_tich_tin_hieu_tong_hop,
    tinh_atr,
    tinh_bollinger_bands,
    tinh_ema,
    tinh_macd,
    tinh_pivot_points,
    tinh_quan_tri_rui_ro,
    tinh_rsi,
    tinh_sma,
)
from vietnamese_ai.plugins.plugin_chungkhoan_crypto import PluginChungKhoanCrypto
from vietnamese_ai.rag.rag_pipeline import RAGPipeline


def test_chuan_hoa_va_nhan_dien_ma():
    """Kiểm tra nhận diện mã cổ phiếu VN vs Crypto."""
    assert la_ma_crypto("BTCUSDT") is True
    assert la_ma_crypto("eth") is True
    assert la_ma_crypto("solusdt") is True
    assert la_ma_crypto("HPG") is False
    assert la_ma_crypto("VNM") is False

    ma_chuan, loai = chuan_hoa_ma("btc")
    assert ma_chuan == "BTCUSDT"
    assert loai == "crypto"

    ma_chuan_vn, loai_vn = chuan_hoa_ma("ssi")
    assert ma_chuan_vn == "SSI"
    assert loai_vn == "chung_khoan_vn"


def test_market_feed_mock():
    """Kiểm tra bộ tạo dữ liệu giả lập (Deterministic Mock Feed)."""
    feed = BoPhatGiaGiaLap(seed=123)

    # Test giá snapshot
    hpg = feed.sinh_gia_hien_tai("HPG")
    assert hpg["ma"] == "HPG"
    assert hpg["loai"] == "chung_khoan_vn"
    assert hpg["gia_hien_tai"] > 0
    assert "thay_doi_pct" in hpg
    assert hpg["nguon"] == "synthetic_mock"

    btc = feed.sinh_gia_hien_tai("BTCUSDT")
    assert btc["ma"] == "BTCUSDT"
    assert btc["loai"] == "crypto"
    assert btc["gia_hien_tai"] > 10_000

    # Test nến OHLCV
    nen_hpg = feed.sinh_du_lieu_nen("HPG", khung_thoi_gian="1D", so_nen=40)
    assert len(nen_hpg) == 40
    for n in nen_hpg:
        assert n["cao_nhat"] >= n["thap_nhat"]
        assert n["khoi_luong"] > 0


def test_market_feed_coordinator():
    """Kiểm tra bộ điều phối thị trường và đệm Ring Buffer."""
    coordinator = BoDieuPhoiThiTruong(dung_gia_lap_khi_loi=True)

    # Lấy giá VN Stock
    gia_vnm = coordinator.lay_gia_hien_tai("VNM")
    assert gia_vnm["ma"] == "VNM"
    assert gia_vnm["gia_hien_tai"] > 0

    # Lấy giá Crypto
    gia_eth = coordinator.lay_gia_hien_tai("ETHUSDT")
    assert gia_eth["ma"] == "ETHUSDT"
    assert gia_eth["gia_hien_tai"] > 0

    # Lấy nến
    nen = coordinator.lay_du_lieu_nen("HPG", so_nen=35)
    assert len(nen) == 35


def test_pure_numpy_quant_indicators():
    """Kiểm tra bộ chỉ báo định lượng thuần NumPy."""
    # Tạo chuỗi giá mẫu 60 phiên tăng dần có dao động
    rng = np.random.RandomState(42)
    gia_dong_cua = 100.0 + np.cumsum(rng.normal(0.5, 1.5, 60))
    gia_cao_nhat = gia_dong_cua + rng.uniform(0.5, 2.0, 60)
    gia_thap_nhat = gia_dong_cua - rng.uniform(0.5, 2.0, 60)

    # 1. SMA
    sma20 = tinh_sma(gia_dong_cua, chu_ky=20)
    assert len(sma20) == 60
    assert np.isnan(sma20[18])
    assert not np.isnan(sma20[19])

    # 2. EMA
    ema20 = tinh_ema(gia_dong_cua, chu_ky=20)
    assert len(ema20) == 60
    assert not np.isnan(ema20[-1])

    # 3. RSI
    rsi14 = tinh_rsi(gia_dong_cua, chu_ky=14)
    assert len(rsi14) == 60
    assert not np.isnan(rsi14[-1])
    assert 0.0 <= rsi14[-1] <= 100.0

    # 4. MACD
    macd, signal, hist = tinh_macd(gia_dong_cua, 12, 26, 9)
    assert len(macd) == 60
    assert len(signal) == 60
    assert len(hist) == 60
    assert not np.isnan(hist[-1])
    np.testing.assert_almost_equal(hist[-1], macd[-1] - signal[-1])

    # 5. Bollinger Bands
    upper, mid, lower, bw = tinh_bollinger_bands(gia_dong_cua, 20, 2.0)
    assert upper[-1] >= mid[-1] >= lower[-1]
    assert bw[-1] > 0

    # 6. ATR
    atr = tinh_atr(gia_cao_nhat, gia_thap_nhat, gia_dong_cua, 14)
    assert len(atr) == 60
    assert atr[-1] > 0

    # 7. Pivot Points
    pivot_classic = tinh_pivot_points(110.0, 95.0, 105.0, phuong_phap="classic")
    assert pivot_classic["p"] == pytest.approx((110.0 + 95.0 + 105.0) / 3.0, 0.01)
    assert pivot_classic["r1"] > pivot_classic["p"] > pivot_classic["s1"]

    pivot_fibo = tinh_pivot_points(110.0, 95.0, 105.0, phuong_phap="fibonacci")
    assert pivot_fibo["r1"] > pivot_fibo["p"] > pivot_fibo["s1"]


def test_tinh_quan_tri_rui_ro_position_sizing():
    """Kiểm tra tính kích thước vị thế theo quy tắc rủi ro 2%."""
    kq = tinh_quan_tri_rui_ro(
        tong_von=100_000_000.0,  # 100 triệu VNĐ
        gia_vao=30.0,            # Mua giá 30
        gia_cat_lo=28.5,         # Cắt lỗ ở 28.5 (lỗ 1.5 nghìn/cổ)
        ty_le_rui_ro=0.02,       # 2% rủi ro = 2 triệu VNĐ
    )

    assert kq["tien_rui_ro_cho_phep"] == 2_000_000.0
    assert kq["phan_tram_cat_lo"] == 5.0  # 1.5 / 30 = 5%
    # Số lượng mua = 2.000.000 / 1.5 = 1333.33 -> Lô VN là 1300 cổ
    assert kq["khoi_luong_lo_vn"] == 1300
    assert kq["muc_chot_loi_muc_tieu"]["rr_1_2_0"] == 33.0  # 30 + 2*1.5 = 33


def test_phan_tich_tin_hieu_tong_hop():
    """Kiểm tra hệ thống phát tín hiệu kỹ thuật tổng hợp."""
    feed = BoPhatGiaGiaLap(seed=99)
    nen = feed.sinh_du_lieu_nen("SSI", so_nen=60)

    pt = phan_tich_tin_hieu_tong_hop(nen)
    assert pt["tin_hieu"] in {"MUA_MANH", "MUA", "THEO_DOI", "BAN", "BAN_MANH"}
    assert -100 <= pt["diem_so"] <= 100
    assert "chi_bao" in pt
    assert "ho_tro_khang_cu" in pt
    assert len(pt["nhan_dinh"]) > 0


def test_plugin_chungkhoan_crypto_tools():
    """Kiểm tra toàn bộ công cụ của PluginChungKhoanCrypto."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)

    # 1. lay_gia_realtime
    res_gia = plugin.lay_gia_realtime("HPG")
    assert res_gia["trang_thai"] == "thanh_cong"
    assert "HPG" in res_gia["tom_tat"]
    assert "⚠️" in res_gia["canh_bao"]

    res_btc = plugin.lay_gia_realtime("BTCUSDT")
    assert res_btc["trang_thai"] == "thanh_cong"
    assert res_btc["du_lieu"]["loai"] == "crypto"

    # 2. phan_tich_ky_thuat
    res_pt = plugin.phan_tich_ky_thuat("SSI", so_nen=50)
    assert res_pt["trang_thai"] == "thanh_cong"
    assert "ket_qua" in res_pt
    assert res_pt["ket_qua"]["diem_so"] is not None

    # 3. tinh_quan_tri_rui_ro_lenh
    res_qr = plugin.tinh_quan_tri_rui_ro_lenh(50_000_000, 25.0, 23.5)
    assert res_qr["trang_thai"] == "thanh_cong"
    assert res_qr["ket_qua"]["khoi_luong_lo_vn"] > 0

    # 4. tra_cuu_luat_giao_dich
    res_t2 = plugin.tra_cuu_luat_giao_dich("T+2.5")
    assert res_t2["trang_thai"] == "thanh_cong"
    assert any("T+2.5" in item["muc"] for item in res_t2["ket_qua"])

    res_halving = plugin.tra_cuu_luat_giao_dich("Halving")
    assert res_halving["trang_thai"] == "thanh_cong"
    assert any("HALVING" in item["muc"] for item in res_halving["ket_qua"])

    # 5. phan_tich_tam_ly_thi_truong
    res_pos = plugin.phan_tich_tam_ly_thi_truong(
        "Thị trường bùng nổ, SSI và HPG tím lịm vượt đỉnh, dòng tiền vào cực mạnh mẽ!"
    )
    assert res_pos["tam_ly"] == "LAC_QUAN"
    assert res_pos["chi_so_so_hai_tham_lam"] > 50

    res_neg = plugin.phan_tich_tam_ly_thi_truong(
        "Thị trường sập nặng, áp lực bán tháo lan rộng, cổ phiếu lau sàn bị call margin giải chấp."
    )
    assert res_neg["tam_ly"] == "BI_QUAN"
    assert res_neg["chi_so_so_hai_tham_lam"] < 50


def test_plugin_tri_thuc_rag():
    """Kiểm tra việc tích hợp tri thức chứng khoán vào RAG Pipeline."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)
    tri_thuc = plugin.cung_cap_tri_thuc()
    assert len(tri_thuc) >= 5

    rag = RAGPipeline(bat_tuong_lua=False)
    so_tai_lieu = rag.nap_plugin(plugin)
    assert so_tai_lieu >= 5

    kq = rag.tim_kiem("Quy chế thanh toán T+2.5 cổ phiếu", top_k=2)
    assert len(kq) >= 1
    noi_dung_top = (kq[0].get("noi_dung") or kq[0].get("metadata", {}).get("noi_dung", "")).lower()
    assert "t+2.5" in noi_dung_top or "thanh toán" in noi_dung_top


def test_agent_integration_with_chungkhoan_plugin():
    """Kiểm tra Tác tử (TacTu) tích hợp PluginChungKhoanCrypto và gọi tool thành công."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)

    def llm_phan_hoi(prompt: str) -> str:
        return (
            "Suy nghĩ: Cần kiểm tra giá realtime của cổ phiếu HPG.\n"
            'Hành động: lay_gia_realtime\n'
            'Tham số: {"ma": "HPG"}'
        )

    tac_tu = TacTu(llm=llm_phan_hoi, max_iterations=2)
    so_cc = tac_tu.nap_plugin(plugin)
    assert so_cc == 6
    assert "lay_gia_realtime" in tac_tu.cong_cu
    assert "phan_tich_ky_thuat" in tac_tu.cong_cu
    assert "du_doan_xu_huong_5p" in tac_tu.cong_cu
    assert "tinh_quan_tri_rui_ro" in tac_tu.cong_cu

    # Thực thi công cụ qua TacTu
    cc_gia = tac_tu.cong_cu["lay_gia_realtime"]
    kq_gia = cc_gia.chay(ma="HPG")
    assert kq_gia["trang_thai"] == "thanh_cong"
    assert kq_gia["du_lieu"]["ma"] == "HPG"

    cc_pt = tac_tu.cong_cu["phan_tich_ky_thuat"]
    kq_pt = cc_pt.chay(ma="BTCUSDT", so_nen=40)
    assert kq_pt["trang_thai"] == "thanh_cong"
    assert kq_pt["ma"] == "BTCUSDT"


def test_quan_ly_plugin_hot_swap_chungkhoan():
    """Kiểm tra nạp động file plugins/plugin_chungkhoan_crypto.py qua QuanLyPlugin."""
    duong_dan = os.path.abspath("plugins/plugin_chungkhoan_crypto.py")
    assert os.path.exists(duong_dan)

    ql = QuanLyPlugin()
    ok, loi = ql.kiem_tra_an_ninh(duong_dan)
    assert ok is True, f"Kiểm tra an ninh AST thất bại: {loi}"

    plugin_nap = ql.nap_plugin(duong_dan)
    assert plugin_nap is not None
    assert isinstance(plugin_nap, PluginChungKhoanCrypto)
    assert "lay_gia_realtime" in [c["ten"] for c in plugin_nap.dang_ky_cong_cu()]
