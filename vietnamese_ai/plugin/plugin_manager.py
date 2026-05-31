"""
QuanLyPlugin (Plugin Manager) - Lõi nạp và gỡ bỏ mã Python động (Hot-Swap).
Kết nối với V-Sandbox để quét mã trước khi nạp.
"""

import importlib.util
import inspect
import logging
import os
import sys
from typing import Dict, Optional

from vietnamese_ai.sandbox import LoiAnNinh, PhanTichAST

from .plugin_base import PluginCoSo

logger = logging.getLogger("V-Plugin")


class QuanLyPlugin:
    """Hệ thống Quản trị V-Plugin."""

    def __init__(self):
        # Lưu trữ các đối tượng Plugin đã khởi tạo: { ten_module: object_plugin }
        self.cac_plugin_dang_chay: Dict[str, PluginCoSo] = {}

    def nap_plugin(
        self, duong_dan_file: str, bo_qua_kiem_duyet: bool = False
    ) -> Optional[PluginCoSo]:
        """
        Nạp một file .py bên ngoài vào hệ thống lúc Runtime.

        Args:
            duong_dan_file: Đường dẫn tới file plugin (.py).
            bo_qua_kiem_duyet: Nếu True, bỏ qua V-Sandbox (Dành cho Admin tin cậy).

        Returns:
            Instance của Plugin nếu thành công, None nếu thất bại.
        """
        if not os.path.exists(duong_dan_file):
            logger.error(f"[Plugin] File không tồn tại: {duong_dan_file}")
            return None

        ten_module = os.path.basename(duong_dan_file).replace(".py", "")

        # 1. Kiểm duyệt mã nguồn bằng V-Sandbox (Tĩnh)
        if not bo_qua_kiem_duyet:
            with open(duong_dan_file, "r", encoding="utf-8") as f:
                ma_nguon = f.read()
            try:
                PhanTichAST.kiem_tra(ma_nguon)
            except LoiAnNinh as e:
                logger.error(f"[Plugin] Đã chặn nạp Plugin '{ten_module}' vì lý do an ninh: {e}")
                raise
            except SyntaxError as e:
                logger.error(f"[Plugin] Plugin '{ten_module}' bị lỗi cú pháp: {e}")
                raise

        # 2. Sử dụng importlib để nạp động (Dynamic Loading)
        try:
            # Tạo Spec và Module
            spec = importlib.util.spec_from_file_location(ten_module, duong_dan_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Không thể tạo Spec cho {duong_dan_file}")

            module_nap = importlib.util.module_from_spec(spec)

            # Kích hoạt đăng ký module vào sys (cần thiết cho các tham chiếu chéo)
            sys.modules[ten_module] = module_nap

            # Thực thi file để khởi tạo các biến/lớp bên trong
            spec.loader.exec_module(module_nap)

            # 3. Quét tìm Lớp (Class) kế thừa từ PluginCoSo
            for ten_thanh_phan, kieu_thanh_phan in inspect.getmembers(module_nap, inspect.isclass):
                # Loại trừ chính lớp PluginCoSo ra khỏi kết quả
                if issubclass(kieu_thanh_phan, PluginCoSo) and kieu_thanh_phan is not PluginCoSo:
                    # 4. Khởi tạo và Lưu trữ
                    thuc_the_plugin = kieu_thanh_phan()

                    if thuc_the_plugin.khoi_dong():
                        self.cac_plugin_dang_chay[ten_module] = thuc_the_plugin
                        logger.info(
                            f"[Plugin] Đã nạp thành công: {thuc_the_plugin.ten} v{thuc_the_plugin.phien_ban}"
                        )
                        return thuc_the_plugin
                    else:
                        logger.error(
                            f"[Plugin] Hàm khoi_dong() của '{thuc_the_plugin.ten}' trả về False."
                        )
                        self.go_plugin(ten_module)  # Dọn dẹp
                        return None

            logger.error(
                f"[Plugin] Không tìm thấy Class nào kế thừa PluginCoSo trong file '{duong_dan_file}'"
            )
            self.go_plugin(ten_module)  # Dọn dẹp module rác
            return None

        except Exception as e:
            logger.error(f"[Plugin] Lỗi trong quá trình nạp '{ten_module}': {e}")
            if ten_module in sys.modules:
                del sys.modules[ten_module]
            return None

    def go_plugin(self, ten_module: str) -> bool:
        """
        Gỡ bỏ hoàn toàn một Plugin khỏi RAM để cập nhật hoặc xóa (Hot-Unload).

        Args:
            ten_module: Tên module (tên file không có .py).

        Returns:
            True nếu gỡ thành công.
        """
        # 1. Gọi hàm dọn dẹp của Plugin
        if ten_module in self.cac_plugin_dang_chay:
            plugin = self.cac_plugin_dang_chay[ten_module]
            try:
                plugin.tat()
                logger.info(f"[Plugin] Đã tắt an toàn: {plugin.ten}")
            except Exception as e:
                logger.warning(f"[Plugin] Lỗi khi gọi tat() cho '{plugin.ten}': {e}")
            del self.cac_plugin_dang_chay[ten_module]

        # 2. Xóa khỏi Bộ nhớ Hệ thống Python
        if ten_module in sys.modules:
            del sys.modules[ten_module]
            logger.debug(f"[Plugin] Đã gỡ '{ten_module}' khỏi sys.modules.")

        return True

    def lay_plugin(self, ten_module: str) -> Optional[PluginCoSo]:
        """Lấy instance của plugin đang chạy."""
        return self.cac_plugin_dang_chay.get(ten_module)
