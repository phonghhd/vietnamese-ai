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

from typing import Any, Dict, Tuple

from .legacy_manager import PluginManager
from .market_feed import BoDieuPhoiThiTruong, BoPhatGiaGiaLap
from .market_predictor import BoDuDoanXuHuongNganHan
from .micro_features import trich_xuat_24_dac_trung_vi_mo
from .plugin_base import (
    PluginCongCu,
    PluginCoSo,
    PluginGiaiPhap,
    PluginTriThuc,
)
from .plugin_manager import QuanLyPlugin
from .quant_indicators import phan_tich_tin_hieu_tong_hop, tinh_quan_tri_rui_ro

_LAZY_DOMAIN_PLUGINS: Dict[str, Tuple[str, str]] = {
    "PluginTaiChinh": ("vietnamese_ai.plugins.plugin_taichinh", "PluginTaiChinh"),
    "PluginYTe": ("vietnamese_ai.plugins.plugin_yte", "PluginYTe"),
    "PluginChungKhoanCrypto": ("vietnamese_ai.plugins.plugin_chungkhoan_crypto", "PluginChungKhoanCrypto"),
}

DANH_MUC_PLUGIN_CHUYEN_NGANH = tuple(_LAZY_DOMAIN_PLUGINS.keys())


def __getattr__(name: str) -> Any:
    """Tải lười các plugin chuyên ngành hợp nhất từ `vietnamese_ai.plugins` vào không gian tên chuẩn `vietnamese_ai.plugin`."""
    if name in _LAZY_DOMAIN_PLUGINS:
        import importlib

        mod_path, attr = _LAZY_DOMAIN_PLUGINS[name]
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
