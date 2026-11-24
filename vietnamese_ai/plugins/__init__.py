"""
Kho Plugin Mẫu Chuyên Ngành cho Vietnamese AI Framework.

Ghi chú hợp nhất không gian tên: tất cả plugin chuyên ngành cũng được xuất thống nhất
từ không gian tên chuẩn `vietnamese_ai.plugin` (ví dụ `from vietnamese_ai.plugin import PluginTaiChinh`);
`vietnamese_ai.plugins` được duy trì đầy đủ để bảo đảm tương thích ngược 100%.
"""

from .plugin_chungkhoan_crypto import PluginChungKhoanCrypto
from .plugin_taichinh import PluginTaiChinh
from .plugin_yte import PluginYTe

__all__ = ["PluginTaiChinh", "PluginYTe", "PluginChungKhoanCrypto"]
