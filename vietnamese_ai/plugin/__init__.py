"""
V-Plugin - Nền tảng mở rộng (Extension System) cho EvoNet-Studio.
Hỗ trợ tháo lắp nóng (Hot-Swap) và quét mã nguồn bằng Sandbox.
"""

from .plugin_base import PluginCoSo
from .plugin_manager import QuanLyPlugin

__all__ = ["PluginCoSo", "QuanLyPlugin"]
