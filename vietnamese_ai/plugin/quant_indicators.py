"""Shim tương thích ngược cho `vietnamese_ai.plugins.finance.quant_indicators` (deprecated từ v31.2.0)."""

from __future__ import annotations

import warnings

from vietnamese_ai.plugins.finance.quant_indicators import (
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

warnings.warn(
    "Module 'vietnamese_ai.plugin.quant_indicators' đã được di chuyển sang "
    "'vietnamese_ai.plugins.finance.quant_indicators' từ v31.2.0; shim cũ sẽ bị gỡ ở v32.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "tinh_sma",
    "tinh_ema",
    "tinh_rsi",
    "tinh_macd",
    "tinh_bollinger_bands",
    "tinh_atr",
    "tinh_pivot_points",
    "tinh_quan_tri_rui_ro",
    "phan_tich_tin_hieu_tong_hop",
]
