"""
V-Plugin 2.0 - Nền tảng mở rộng (Extension & Plugin System) cho Vietnamese AI Framework.

Bao gồm:
- QuanLyPlugin: Bộ quản trị nạp, kiểm duyệt an ninh AST Sandbox và hot-swap plugins.
- PluginCoSo: Lớp cơ sở tổng quát cho mọi plugin.
- PluginCongCu: Chuẩn plugin công cụ nghiệp vụ (Tool Calling) cho Agent.
- PluginTriThuc: Chuẩn plugin tri thức & tài liệu cho RAG Pipeline.
- PluginGiaiPhap: Chuẩn plugin giải pháp nghiệp vụ toàn diện end-to-end.
- PluginManager: Bộ quản trị hook/metric kiểu legacy (tương thích ngược).
"""

import importlib
import warnings
from typing import Any, Dict, Tuple

from .legacy_manager import PluginManager
from .plugin_base import (
    PluginCongCu,
    PluginCoSo,
    PluginGiaiPhap,
    PluginTriThuc,
)
from .plugin_manager import QuanLyPlugin

_LAZY_DOMAIN_PLUGINS: Dict[str, Tuple[str, str]] = {
    "PluginTaiChinh": ("vietnamese_ai.plugins.plugin_taichinh", "PluginTaiChinh"),
    "PluginYTe": ("vietnamese_ai.plugins.plugin_yte", "PluginYTe"),
    "PluginChungKhoanCrypto": ("vietnamese_ai.plugins.plugin_chungkhoan_crypto", "PluginChungKhoanCrypto"),
}

_LAZY_FINANCE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BoDieuPhoiThiTruong": ("vietnamese_ai.plugins.finance.market_feed", "BoDieuPhoiThiTruong"),
    "BoPhatGiaGiaLap": ("vietnamese_ai.plugins.finance.market_feed", "BoPhatGiaGiaLap"),
    "BoDuDoanXuHuongNganHan": ("vietnamese_ai.plugins.finance.market_predictor", "BoDuDoanXuHuongNganHan"),
    "trich_xuat_24_dac_trung_vi_mo": (
        "vietnamese_ai.plugins.finance.micro_features",
        "trich_xuat_24_dac_trung_vi_mo",
    ),
    "phan_tich_tin_hieu_tong_hop": (
        "vietnamese_ai.plugins.finance.quant_indicators",
        "phan_tich_tin_hieu_tong_hop",
    ),
    "tinh_quan_tri_rui_ro": ("vietnamese_ai.plugins.finance.quant_indicators", "tinh_quan_tri_rui_ro"),
}

DANH_MUC_PLUGIN_CHUYEN_NGANH = tuple(_LAZY_DOMAIN_PLUGINS.keys())


def __getattr__(name: str) -> Any:
    """Tải lười các plugin chuyên ngành từ `vietnamese_ai.plugins` và shim tương thích ngược cho nhóm tài chính."""
    if name in _LAZY_DOMAIN_PLUGINS:
        mod_path, attr = _LAZY_DOMAIN_PLUGINS[name]
        return getattr(importlib.import_module(mod_path), attr)
    if name in _LAZY_FINANCE_EXPORTS:
        mod_path, attr = _LAZY_FINANCE_EXPORTS[name]
        warnings.warn(
            f"Nhập '{name}' từ 'vietnamese_ai.plugin' đã lỗi thời từ v31.2.0; "
            f"hãy đổi sang 'from vietnamese_ai.plugins.finance import {name}'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"Module 'vietnamese_ai.plugin' không có thuộc tính '{name}'")


__all__ = [
    "PluginCoSo",
    "PluginCongCu",
    "PluginTriThuc",
    "PluginGiaiPhap",
    "QuanLyPlugin",
    "PluginManager",
    "BoDieuPhoiThiTruong",
    "BoPhatGiaGiaLap",
    "BoDuDoanXuHuongNganHan",
    "trich_xuat_24_dac_trung_vi_mo",
    "phan_tich_tin_hieu_tong_hop",
    "tinh_quan_tri_rui_ro",
    "PluginTaiChinh",
    "PluginYTe",
    "PluginChungKhoanCrypto",
    "DANH_MUC_PLUGIN_CHUYEN_NGANH",
]
