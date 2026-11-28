"""Phân hệ Triển khai Di động & Quản lý Năng lượng (`vietnamese_ai.mobile`)."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from .bridge import MobileAppBridge
from .browser_copilot import BrowserCopilot
from .deployment import MobileDeployer, ONNXExporter, TriKhaiDiDong
from .mobile_agent import MobileHybridAgent
from .power_manager import PowerManager
from .sdk_generator import SDKGenerator

_LAZY_DEPRECATED: Dict[str, Tuple[str, str]] = {
    "cong_cu_chup_anh_camera": ("vietnamese_ai.mobile.mobile_tools", "cong_cu_chup_anh_camera"),
    "cong_cu_doc_thong_bao_sms": ("vietnamese_ai.mobile.mobile_tools", "cong_cu_doc_thong_bao_sms"),
    "cong_cu_lay_toa_do_gps": ("vietnamese_ai.mobile.mobile_tools", "cong_cu_lay_toa_do_gps"),
}


def __getattr__(name: str) -> Any:
    """Tải lười các công cụ mobile mô phỏng đã đánh dấu lỗi thời để tránh cảnh báo sớm khi dùng `PowerManager`."""
    if name in _LAZY_DEPRECATED:
        mod_path, attr = _LAZY_DEPRECATED[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"Module 'vietnamese_ai.mobile' không có thuộc tính '{name}'")


__all__ = [
    "MobileDeployer",
    "ONNXExporter",
    "TriKhaiDiDong",
    "PowerManager",
    "MobileHybridAgent",
    "BrowserCopilot",
    "MobileAppBridge",
    "SDKGenerator",
    "cong_cu_chup_anh_camera",
    "cong_cu_doc_thong_bao_sms",
    "cong_cu_lay_toa_do_gps",
]
