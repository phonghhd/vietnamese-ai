"""Shim tương thích ngược cho `vietnamese_ai.plugins.finance.market_feed` (deprecated từ v31.2.0)."""

from __future__ import annotations

import warnings

from vietnamese_ai.plugins.finance.market_feed import (
    CAP_CRYPTO_PHO_BIEN,
    MA_CHUNG_KHOAN_VN_PHO_BIEN,
    BoDieuPhoiThiTruong,
    BoPhatGiaGiaLap,
    BoThuThapCrypto,
    BoThuThapGiaChungKhoan,
    chuan_hoa_ma,
    la_ma_crypto,
)

warnings.warn(
    "Module 'vietnamese_ai.plugin.market_feed' đã được di chuyển sang "
    "'vietnamese_ai.plugins.finance.market_feed' từ v31.2.0; shim cũ sẽ bị gỡ ở v32.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MA_CHUNG_KHOAN_VN_PHO_BIEN",
    "CAP_CRYPTO_PHO_BIEN",
    "la_ma_crypto",
    "chuan_hoa_ma",
    "BoPhatGiaGiaLap",
    "BoThuThapCrypto",
    "BoThuThapGiaChungKhoan",
    "BoDieuPhoiThiTruong",
]
