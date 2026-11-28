"""Shim tương thích ngược cho `vietnamese_ai.plugins.finance.market_predictor` (deprecated từ v31.2.0)."""

from __future__ import annotations

import warnings

from vietnamese_ai.plugins.finance.market_predictor import (
    BoDuDoanXuHuongNganHan,
    MangNoRonViMoPureNumPy,
)

warnings.warn(
    "Module 'vietnamese_ai.plugin.market_predictor' đã được di chuyển sang "
    "'vietnamese_ai.plugins.finance.market_predictor' từ v31.2.0; shim cũ sẽ bị gỡ ở v32.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MangNoRonViMoPureNumPy",
    "BoDuDoanXuHuongNganHan",
]
