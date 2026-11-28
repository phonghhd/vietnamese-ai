"""
Micro Features Extractor - Bộ trích xuất 24 đặc trưng vi cấu trúc từ nến 1 phút.
Phục vụ mô hình dự báo xu hướng giá ngắn hạn (5 - 10 phút) cho Crypto & Phái sinh VN30.
Tối ưu hóa vector hóa 100% bằng NumPy, zero external dependency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from vietnamese_ai.plugins.finance.quant_indicators import (
    tinh_atr,
    tinh_bollinger_bands,
    tinh_ema,
    tinh_macd,
    tinh_rsi,
    tinh_sma,
)

TEN_24_DAC_TRUNG: List[str] = [
    "log_ret_1m",
    "log_ret_2m",
    "log_ret_3m",
    "log_ret_5m",
    "log_ret_10m",
    "dist_ema9",
    "dist_ema21",
    "ema9_ema21_diff",
    "rsi_7_norm",
    "rsi_14_norm",
    "macd_hist_norm",
    "macd_delta_hist",
    "body_to_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_loc_in_range",
    "norm_true_range_1m",
    "atr_ratio_5m_to_20m",
    "vol_ratio_to_ma20",
    "vol_surge_3m",
    "vol_price_trend",
    "bb_percent_b",
    "bb_bandwidth",
    "high_low_skew_5m",
]


def chuyen_doi_danh_sach_nen_sang_mang(
    danh_sach_nen: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển đổi danh sách nến dạng dict thành các mảng NumPy float:
    (arr_mo, arr_cao, arr_thap, arr_dong, arr_vol).
    """
    n = len(danh_sach_nen)
    arr_mo = np.zeros(n, dtype=float)
    arr_cao = np.zeros(n, dtype=float)
    arr_thap = np.zeros(n, dtype=float)
    arr_dong = np.zeros(n, dtype=float)
    arr_vol = np.zeros(n, dtype=float)

    for i, nen in enumerate(danh_sach_nen):
        arr_mo[i] = float(nen.get("mo_cua", 0.0))
        arr_cao[i] = float(nen.get("cao_nhat", 0.0))
        arr_thap[i] = float(nen.get("thap_nhat", 0.0))
        arr_dong[i] = float(nen.get("dong_cua", 0.0))
        arr_vol[i] = float(nen.get("khoi_luong", 0.0))

    return arr_mo, arr_cao, arr_thap, arr_dong, arr_vol


def trich_xuat_24_dac_trung_vi_mo(
    danh_sach_nen: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Trích xuất vector 24 chiều đặc trưng vi cấu trúc từ tối thiểu 25-30 nến 1 phút.
    Nếu danh sách nến không đủ, hàm sẽ tự động đệm dữ liệu hợp lý.
    Trả về mảng NumPy shape (24,), giá trị được giới hạn an toàn [-5.0, 5.0].
    """
    eps = 1e-8
    arr_mo, arr_cao, arr_thap, arr_dong, arr_vol = chuyen_doi_danh_sach_nen_sang_mang(danh_sach_nen)
    n = len(arr_dong)

    # Cần tối thiểu 25 nến để các chỉ báo EMA21, ATR, Bollinger Bands ổn định
    if n < 25:
        thieu = 25 - n
        arr_dong = np.pad(arr_dong, (thieu, 0), mode="edge")
        arr_mo = np.pad(arr_mo, (thieu, 0), mode="edge")
        arr_cao = np.pad(arr_cao, (thieu, 0), mode="edge")
        arr_thap = np.pad(arr_thap, (thieu, 0), mode="edge")
        arr_vol = np.pad(arr_vol, (thieu, 0), mode="edge")
        n = len(arr_dong)

    gia_ht = arr_dong[-1]
    features = np.zeros(24, dtype=float)

    # 1. Nhóm Lợi suất Động lượng (Momentum Log Returns)
    features[0] = math.log((arr_dong[-1] + eps) / (arr_dong[-2] + eps)) if n >= 2 else 0.0
    features[1] = math.log((arr_dong[-1] + eps) / (arr_dong[-3] + eps)) if n >= 3 else 0.0
    features[2] = math.log((arr_dong[-1] + eps) / (arr_dong[-4] + eps)) if n >= 4 else 0.0
    features[3] = math.log((arr_dong[-1] + eps) / (arr_dong[-6] + eps)) if n >= 6 else 0.0
    features[4] = math.log((arr_dong[-1] + eps) / (arr_dong[-11] + eps)) if n >= 11 else 0.0

    # Nhân 100 để đưa về đơn vị % quen thuộc cho mạng nơ-ron
    features[0:5] *= 100.0

    # 2. Nhóm Độ lệch Trung bình động (Mean Reversion)
    ema9 = tinh_ema(arr_dong, chu_ky=9)
    ema21 = tinh_ema(arr_dong, chu_ky=21)
    ema9_ht = ema9[-1] if not np.isnan(ema9[-1]) else gia_ht
    ema21_ht = ema21[-1] if not np.isnan(ema21[-1]) else gia_ht

    features[5] = ((gia_ht - ema9_ht) / (ema9_ht + eps)) * 100.0
    features[6] = ((gia_ht - ema21_ht) / (ema21_ht + eps)) * 100.0
    features[7] = ((ema9_ht - ema21_ht) / (ema21_ht + eps)) * 100.0

    # 3. Nhóm Dao động Vi mô (Micro Oscillators)
    rsi7 = tinh_rsi(arr_dong, chu_ky=7)
    rsi14 = tinh_rsi(arr_dong, chu_ky=14)
    r7_ht = rsi7[-1] if not np.isnan(rsi7[-1]) else 50.0
    r14_ht = rsi14[-1] if not np.isnan(rsi14[-1]) else 50.0
    features[8] = (r7_ht - 50.0) / 25.0
    features[9] = (r14_ht - 50.0) / 25.0

    # 4. Nhóm MACD Khung 1 phút
    macd, signal, hist = tinh_macd(arr_dong, nhanh=6, cham=13, tin_hieu=4)
    hist_ht = hist[-1] if not np.isnan(hist[-1]) else 0.0
    hist_tr = hist[-2] if len(hist) > 1 and not np.isnan(hist[-2]) else hist_ht
    features[10] = (hist_ht / (gia_ht + eps)) * 1000.0
    features[11] = ((hist_ht - hist_tr) / (gia_ht + eps)) * 1000.0

    # 5. Nhóm Cấu trúc Nến (Candlestick Geometry)
    range_nen = max(arr_cao[-1] - arr_thap[-1], eps)
    body_nen = abs(arr_dong[-1] - arr_mo[-1])
    features[12] = body_nen / range_nen
    features[13] = (arr_cao[-1] - max(arr_mo[-1], arr_dong[-1])) / range_nen
    features[14] = (min(arr_mo[-1], arr_dong[-1]) - arr_thap[-1]) / range_nen
    features[15] = (arr_dong[-1] - arr_thap[-1]) / range_nen

    # 6. Nhóm Biến động Vi mô (Micro Volatility)
    features[16] = (range_nen / (gia_ht + eps)) * 100.0
    atr5 = tinh_atr(arr_cao, arr_thap, arr_dong, chu_ky=5)
    atr20 = tinh_atr(arr_cao, arr_thap, arr_dong, chu_ky=20)
    atr5_ht = atr5[-1] if not np.isnan(atr5[-1]) else range_nen
    atr20_ht = atr20[-1] if not np.isnan(atr20[-1]) else range_nen
    features[17] = atr5_ht / (atr20_ht + eps)

    # 7. Nhóm Khối lượng Giao dịch (Volume Dynamics)
    vol_ma20 = tinh_sma(arr_vol, chu_ky=20)
    vol_ma20_ht = vol_ma20[-1] if not np.isnan(vol_ma20[-1]) else (arr_vol[-1] + eps)
    vol_ma3 = tinh_sma(arr_vol, chu_ky=3)
    vol_ma15 = tinh_sma(arr_vol, chu_ky=15)
    vol_ma3_ht = vol_ma3[-1] if not np.isnan(vol_ma3[-1]) else arr_vol[-1]
    vol_ma15_ht = vol_ma15[-1] if not np.isnan(vol_ma15[-1]) else (arr_vol[-1] + eps)

    features[18] = arr_vol[-1] / (vol_ma20_ht + eps)
    features[19] = vol_ma3_ht / (vol_ma15_ht + eps)

    huong_gia = 1.0 if arr_dong[-1] >= arr_dong[-2] else -1.0
    features[20] = huong_gia * math.log(1.0 + max(0.0, arr_vol[-1] / (vol_ma20_ht + eps)))

    # 8. Nhóm Bollinger Bands Khung 1 phút
    bb_u, bb_m, bb_l, bb_w = tinh_bollinger_bands(arr_dong, chu_ky=15, do_lech=2.0)
    b_u_ht = bb_u[-1] if not np.isnan(bb_u[-1]) else gia_ht * 1.01
    b_l_ht = bb_l[-1] if not np.isnan(bb_l[-1]) else gia_ht * 0.99
    b_w_ht = bb_w[-1] if not np.isnan(bb_w[-1]) else 0.02

    bb_span = max(b_u_ht - b_l_ht, eps)
    features[21] = (gia_ht - b_l_ht) / bb_span
    features[22] = b_w_ht * 100.0

    # 9. Độ lệch Đỉnh/Đáy 5 nến gần nhất (5m Skew)
    cao_5m = np.max(arr_cao[-5:]) if n >= 5 else arr_cao[-1]
    thap_5m = np.min(arr_thap[-5:]) if n >= 5 else arr_thap[-1]
    khoang_tren = max(0.0, cao_5m - gia_ht)
    khoang_duoi = max(0.0, gia_ht - thap_5m)
    features[23] = (khoang_duoi - khoang_tren) / (max(cao_5m - thap_5m, eps))

    # Làm sạch NaN / Inf và cắt ngưỡng chuẩn hóa an toàn [-5.0, 5.0]
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
    features = np.clip(features, -5.0, 5.0)

    return features
