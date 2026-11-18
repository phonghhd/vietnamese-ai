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
]
