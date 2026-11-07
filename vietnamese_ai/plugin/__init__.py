"""
V-Plugin - Nền tảng mở rộng (Extension System) cho EvoNet-Studio.

Hai API song song (mục đích khác nhau):
- `QuanLyPlugin`: hot-swap file .py + quét Sandbox (Studio Super Core).
- `PluginManager`: đăng ký hook/metric kiểu legacy (pre_train, post_train, ...).
"""

from .legacy_manager import PluginManager
from .plugin_base import PluginCoSo
from .plugin_manager import QuanLyPlugin

__all__ = ["PluginCoSo", "QuanLyPlugin", "PluginManager"]
