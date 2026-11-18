"""
Bộ kiểm thử toàn diện cho Bộ Dự Đoán Xu Hướng Giá Ngắn Hạn (5 - 10 phút) Pure NumPy.
Kiểm tra:
1. Trích xuất 24 đặc trưng vi cấu trúc từ nến 1 phút (micro_features.py).
2. Xử lý thiếu nến và tự động đệm dữ liệu (auto-padding).
3. Lan truyền tiến của mạng nơ-ron đa tầng kép thuần NumPy (Dual-Head MLP).
4. Phân phối xác suất Softmax và biên độ dao động kỳ vọng.
5. Tích hợp công cụ du_doan_xu_huong_5p vào PluginChungKhoanCrypto.
6. Tích hợp Tool Calling với Tác tử AI (TacTu) và tri thức RAG.
"""

from __future__ import annotations

import numpy as np
import pytest

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.plugin.market_feed import BoPhatGiaGiaLap
from vietnamese_ai.plugin.market_predictor import (
    BoDuDoanXuHuongNganHan,
    MangNoRonViMoPureNumPy,
)
from vietnamese_ai.plugin.micro_features import (
    TEN_24_DAC_TRUNG,
    chuyen_doi_danh_sach_nen_sang_mang,
    trich_xuat_24_dac_trung_vi_mo,
)
from vietnamese_ai.plugins.plugin_chungkhoan_crypto import PluginChungKhoanCrypto
from vietnamese_ai.rag.rag_pipeline import RAGPipeline


def test_ten_va_so_luong_dac_trung():
    """Kiểm tra danh mục 24 đặc trưng vi mô."""
    assert len(TEN_24_DAC_TRUNG) == 24
    assert "log_ret_1m" in TEN_24_DAC_TRUNG
    assert "rsi_7_norm" in TEN_24_DAC_TRUNG
    assert "bb_percent_b" in TEN_24_DAC_TRUNG
    assert "vol_ratio_to_ma20" in TEN_24_DAC_TRUNG


def test_chuyen_doi_mang_va_trich_xuat_dac_trung():
    """Kiểm tra trích xuất 24 đặc trưng từ 35 nến 1 phút chuẩn."""
    feed = BoPhatGiaGiaLap(seed=777)
    nen = feed.sinh_du_lieu_nen("BTCUSDT", khung_thoi_gian="1m", so_nen=35)
    assert len(nen) == 35

    arr_o, arr_h, arr_l, arr_c, arr_v = chuyen_doi_danh_sach_nen_sang_mang(nen)
    assert len(arr_c) == 35
    assert np.all(arr_h >= arr_l)

    features = trich_xuat_24_dac_trung_vi_mo(nen)
    assert isinstance(features, np.ndarray)
    assert features.shape == (24,)
    # Không được chứa NaN hoặc Inf
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
    # Giá trị nằm trong ngưỡng an toàn [-5.0, 5.0]
    assert np.all(features >= -5.0)
    assert np.all(features <= 5.0)


def test_trich_xuat_khi_thieu_nen_auto_padding():
    """Kiểm tra cơ chế tự động đệm dữ liệu khi người dùng truyền ít hơn 25 nến."""
    feed = BoPhatGiaGiaLap(seed=888)
    nen_ngan = feed.sinh_du_lieu_nen("HPG", khung_thoi_gian="1m", so_nen=12)
    assert len(nen_ngan) == 12

    # Vẫn phải trích xuất thành công 24 đặc trưng hợp lệ, không sập
    features = trich_xuat_24_dac_trung_vi_mo(nen_ngan)
    assert features.shape == (24,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_mang_no_ron_vi_mo_forward_pass():
    """Kiểm tra lan truyền tiến mạng nơ-ron thuần NumPy (Dual-Head MLP)."""
    mlp = MangNoRonViMoPureNumPy(seed=42)

    # Vector đặc trưng mẫu
    dummy_feat = np.random.RandomState(123).normal(0.0, 1.0, 24)
    dummy_feat = np.clip(dummy_feat, -3.0, 3.0)

    probs, vol_pct = mlp.forward(dummy_feat)

    # 1. Head Phân loại
    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    # 2. Head Biến động
    assert isinstance(vol_pct, float)
    assert 0.10 <= vol_pct <= 3.50


def test_bo_du_doan_xu_huong_ngan_han_5p_va_10p():
    """Kiểm tra bộ dự đoán hoàn chỉnh cho khung 5 phút và 10 phút."""
    predictor = BoDuDoanXuHuongNganHan(seed=2026)
    feed = BoPhatGiaGiaLap(seed=101)
    nen = feed.sinh_du_lieu_nen("ETHUSDT", khung_thoi_gian="1m", so_nen=40)

    # 1. Dự đoán 5 phút
    res_5p = predictor.du_doan_tu_nen(nen, khung_du_doan="5p", gia_tham_chieu=2650.0)
    assert res_5p["trang_thai"] == "thanh_cong"
    assert res_5p["khung_du_doan"] == "5_phut_toi"
    assert "tang" in res_5p["xac_suat"]
    assert "di_ngang" in res_5p["xac_suat"]
    assert "giam" in res_5p["xac_suat"]

    tong_p = sum(res_5p["xac_suat"].values())
    assert pytest.approx(tong_p, 0.2) == 100.0

    assert res_5p["xu_huong_chu_dao"] in {
        "TĂNG_NGẮN_HẠN",
        "ĐI_NGANG_TÍCH_LŨY",
        "GIẢM_NGẮN_HẠN",
    }
    assert res_5p["do_tin_cay"] in {"CAO", "TRUNG_BÌNH", "THẤP"}

    vung_gia = res_5p["vung_gia_ky_vong"]
    assert len(vung_gia) == 2
    assert vung_gia[0] <= vung_gia[1]

    # 2. Dự đoán 10 phút
    res_10p = predictor.du_doan_tu_nen(nen, khung_du_doan="10p", gia_tham_chieu=2650.0)
    assert res_10p["trang_thai"] == "thanh_cong"
    assert res_10p["khung_du_doan"] == "10_phut_toi"


def test_plugin_chungkhoan_crypto_tool_du_doan_5p():
    """Kiểm tra công cụ du_doan_xu_huong_5p trong PluginChungKhoanCrypto."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)

    # 1. Dự đoán cho Crypto (BTCUSDT)
    res_btc = plugin.du_doan_xu_huong_5p("BTCUSDT", khung_du_doan="5p")
    assert res_btc["trang_thai"] == "thanh_cong"
    assert res_btc["ma"] == "BTCUSDT"
    assert "du_doan" in res_btc
    assert "xac_suat" in res_btc["du_doan"]
    assert "⚠️" in res_btc["canh_bao"]

    # 2. Dự đoán cho Phái sinh Việt Nam (VN30F1M)
    res_vn30f = plugin.du_doan_xu_huong_5p("VN30F1M", khung_du_doan="10p")
    assert res_vn30f["trang_thai"] == "thanh_cong"
    assert res_vn30f["ma"] == "VN30F1M"
    assert res_vn30f["du_doan"]["khung_du_doan"] == "10_phut_toi"


def test_agent_integration_du_doan_xu_huong():
    """Kiểm tra Tác tử AI (TacTu) gọi công cụ du_doan_xu_huong_5p thành công."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)

    def llm_phan_hoi(prompt: str) -> str:
        return (
            "Suy nghĩ: Cần dự báo xu hướng vi mô 5 phút tới của BTCUSDT.\n"
            'Hành động: du_doan_xu_huong_5p\n'
            'Tham số: {"ma": "BTCUSDT", "khung_du_doan": "5p"}'
        )

    tac_tu = TacTu(llm=llm_phan_hoi, max_iterations=2)
    tac_tu.nap_plugin(plugin)

    assert "du_doan_xu_huong_5p" in tac_tu.cong_cu
    cc = tac_tu.cong_cu["du_doan_xu_huong_5p"]

    kq = cc.chay(ma="BTCUSDT", khung_du_doan="5p")
    assert kq["trang_thai"] == "thanh_cong"
    assert kq["ma"] == "BTCUSDT"
    assert "xac_suat" in kq["du_doan"]


def test_rag_pipeline_tri_thuc_du_doan_vi_mo():
    """Kiểm tra tri thức dự đoán vi mô được RAGPipeline tìm kiếm chính xác."""
    plugin = PluginChungKhoanCrypto(dung_gia_lap_khi_loi=True)
    rag = RAGPipeline(bat_tuong_lua=False)
    rag.nap_plugin(plugin)

    kq = rag.tim_kiem("Phương pháp dự đoán vi cấu trúc 5 phút", top_k=2)
    assert len(kq) >= 1
    noi_dung = (kq[0].get("noi_dung") or kq[0].get("metadata", {}).get("noi_dung", "")).lower()
    assert "vi cấu trúc" in noi_dung or "xác suất" in noi_dung
