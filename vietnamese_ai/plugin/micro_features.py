"""Shim tương thích ngược cho `vietnamese_ai.plugins.finance.micro_features` (deprecated từ v31.2.0)."""

from __future__ import annotations

import warnings

from vietnamese_ai.plugins.finance.micro_features import (
    TEN_24_DAC_TRUNG,
    chuyen_doi_danh_sach_nen_sang_mang,
    trich_xuat_24_dac_trung_vi_mo,
)

warnings.warn(
    "Module 'vietnamese_ai.plugin.micro_features' đã được di chuyển sang "
    "'vietnamese_ai.plugins.finance.micro_features' từ v31.2.0; shim cũ sẽ bị gỡ ở v32.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TEN_24_DAC_TRUNG",
    "chuyen_doi_danh_sach_nen_sang_mang",
    "trich_xuat_24_dac_trung_vi_mo",
]
