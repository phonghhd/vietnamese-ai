"""
QuanLyPlugin (Plugin Manager) - Lõi nạp, kiểm duyệt và quản trị mã Python động (Hot-Swap).
Tích hợp V-Sandbox (PhanTichAST) quét mã tĩnh an toàn trước khi nạp vào Runtime.
Hỗ trợ phân loại: PluginCongCu (Tools), PluginTriThuc (Knowledge), PluginGiaiPhap (Solution).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from vietnamese_ai.sandbox import LoiAnNinh, PhanTichAST

from .plugin_base import PluginCongCu, PluginCoSo, PluginGiaiPhap, PluginTriThuc

logger = logging.getLogger("V-Plugin")

_BASE_CLASSES = {PluginCoSo, PluginCongCu, PluginTriThuc, PluginGiaiPhap}


class QuanLyPlugin:
    """Hệ thống Quản trị & Điều phối V-Plugin 2.0."""

    def __init__(self) -> None:
        self.cac_plugin_dang_chay: Dict[str, PluginCoSo] = {}
        self._plugin_sys_modules: Dict[str, str] = {}

    @property
    def cac_plugin_cong_cu(self) -> Dict[str, PluginCongCu]:
        """Danh sách các Plugin loại Công cụ (Tools)."""
        return {
            k: v
            for k, v in self.cac_plugin_dang_chay.items()
            if isinstance(v, PluginCongCu)
        }

    @property
    def cac_plugin_tri_thuc(self) -> Dict[str, PluginTriThuc]:
        """Danh sách các Plugin loại Tri thức (Knowledge / RAG)."""
        return {
            k: v
            for k, v in self.cac_plugin_dang_chay.items()
            if isinstance(v, PluginTriThuc)
        }

    @property
    def cac_plugin_giai_phap(self) -> Dict[str, PluginGiaiPhap]:
        """Danh sách các Plugin loại Giải pháp (End-to-End Solutions)."""
        return {
            k: v
            for k, v in self.cac_plugin_dang_chay.items()
            if isinstance(v, PluginGiaiPhap)
        }

    def kiem_tra_an_ninh(self, duong_dan_file: str) -> Tuple[bool, str]:
        """
        Kiểm tra an ninh mã nguồn tệp plugin qua bộ phân tích AST tĩnh.

        Returns:
            (hop_le, ly_do)
        """
        if not os.path.exists(duong_dan_file):
            return False, f"Tệp không tồn tại: {duong_dan_file}"
        try:
            with open(duong_dan_file, "r", encoding="utf-8") as f:
                ma_nguon = f.read()
            PhanTichAST.kiem_tra(ma_nguon)
            return True, "Hợp lệ"
        except LoiAnNinh as e:
            return False, f"Vi phạm chính sách an ninh: {e}"
        except SyntaxError as e:
            return False, f"Lỗi cú pháp Python: {e}"
        except Exception as e:
            return False, f"Lỗi kiểm tra an ninh: {e}"

    def dang_ky_plugin(
        self,
        plugin: PluginCoSo,
        ten_module: Optional[str] = None,
    ) -> bool:
        """
        Đăng ký trực tiếp một instance Plugin đã khởi tạo vào hệ thống.
        """
        if not isinstance(plugin, PluginCoSo):
            logger.error("[Plugin] Đối tượng đăng ký phải kế thừa từ PluginCoSo.")
            return False

        key = ten_module or getattr(plugin, "ten", "unnamed").lower().replace(" ", "_")
        if not plugin.trang_thai_hoat_dong:
            if not plugin.khoi_dong():
                logger.error(f"[Plugin] Không thể khởi động plugin: {plugin.ten}")
                return False

        self.cac_plugin_dang_chay[key] = plugin
        logger.info(f"[Plugin] Đã đăng ký trực tiếp: {plugin.ten} (id: {key})")
        return True

    def nap_plugin(
        self, duong_dan_file: str, bo_qua_kiem_duyet: bool = False
    ) -> Optional[PluginCoSo]:
        """
        Nạp một file .py bên ngoài vào hệ thống lúc Runtime kèm kiểm duyệt V-Sandbox.

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
            hop_le, ly_do = self.kiem_tra_an_ninh(duong_dan_file)
            if not hop_le:
                logger.error(f"[Plugin] Đã chặn nạp Plugin '{ten_module}': {ly_do}")
                raise LoiAnNinh(ly_do)

        path_hash = hashlib.sha256(os.path.abspath(duong_dan_file).encode("utf-8")).hexdigest()[:12]
        ten_sys_module = f"vietnamese_ai._loaded_plugins.{ten_module}_{path_hash}"
        try:
            spec = importlib.util.spec_from_file_location(ten_sys_module, duong_dan_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Không thể tạo Spec cho {duong_dan_file}")

            module_nap = importlib.util.module_from_spec(spec)
            sys.modules[ten_sys_module] = module_nap
            self._plugin_sys_modules[ten_module] = ten_sys_module
            spec.loader.exec_module(module_nap)

            for _, kieu_thanh_phan in inspect.getmembers(module_nap, inspect.isclass):
                if issubclass(kieu_thanh_phan, PluginCoSo) and kieu_thanh_phan not in _BASE_CLASSES:
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
                        self.go_plugin(ten_module)
                        return None

            logger.error(
                f"[Plugin] Không tìm thấy Class nào kế thừa PluginCoSo trong file '{duong_dan_file}'"
            )
            self.go_plugin(ten_module)
            return None

        except Exception as e:
            logger.error(f"[Plugin] Lỗi trong quá trình nạp '{ten_module}': {e}")
            sys.modules.pop(ten_sys_module, None)
            self._plugin_sys_modules.pop(ten_module, None)
            return None

    def go_plugin(self, ten_module: str) -> bool:
        """
        Gỡ bỏ hoàn toàn một Plugin khỏi RAM để cập nhật hoặc xóa (Hot-Unload).
        """
        if ten_module in self.cac_plugin_dang_chay:
            plugin = self.cac_plugin_dang_chay[ten_module]
            try:
                plugin.tat()
                logger.info(f"[Plugin] Đã tắt an toàn: {plugin.ten}")
            except Exception as e:
                logger.warning(f"[Plugin] Lỗi khi gọi tat() cho '{plugin.ten}': {e}")
            del self.cac_plugin_dang_chay[ten_module]

        ten_sys_module = self._plugin_sys_modules.pop(ten_module, None)
        if ten_sys_module and ten_sys_module in sys.modules:
            del sys.modules[ten_sys_module]
            logger.debug(f"[Plugin] Đã gỡ '{ten_sys_module}' khỏi sys.modules.")

        return True

    def lay_plugin(self, ten_module: str) -> Optional[PluginCoSo]:
        """Lấy instance của plugin đang chạy."""
        return self.cac_plugin_dang_chay.get(ten_module)

    def lay_tat_ca_cong_cu(self) -> Dict[str, Callable]:
        """
        Thu thập toàn bộ các hàm công cụ từ tất cả PluginCongCu đang hoạt động.
        """
        cong_cu_tong: Dict[str, Callable] = {}
        for plugin in self.cac_plugin_cong_cu.values():
            for cc in plugin.dang_ky_cong_cu():
                ten_cc = cc.get("ten")
                ham = cc.get("ham")
                if ten_cc and callable(ham):
                    cong_cu_tong[ten_cc] = ham
        return cong_cu_tong

    def lay_schema_tat_ca_cong_cu(self) -> List[Dict[str, Any]]:
        """
        Thu thập toàn bộ metadata/schema của các công cụ phục vụ Tác tử tự hành.
        """
        schema_list: List[Dict[str, Any]] = []
        for plugin in self.cac_plugin_cong_cu.values():
            for cc in plugin.dang_ky_cong_cu():
                schema_list.append(
                    {
                        "plugin": plugin.ten,
                        "ten": cc.get("ten"),
                        "mo_ta": cc.get("mo_ta", ""),
                        "tham_so": cc.get("tham_so", {}),
                        "yeu_cau_xac_nhan": cc.get("yeu_cau_xac_nhan", False),
                    }
                )
        return schema_list

    def lay_tat_ca_tri_thuc(self) -> List[Dict[str, Any]]:
        """
        Thu thập toàn bộ dữ liệu tài liệu từ tất cả PluginTriThuc đang hoạt động.
        """
        tri_thuc_tong: List[Dict[str, Any]] = []
        for plugin in self.cac_plugin_tri_thuc.values():
            tri_thuc_tong.extend(plugin.nap_nguon_tri_thuc())
        return tri_thuc_tong

    def danh_sach_plugin(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tóm tắt toàn bộ plugin đang nạp trong hệ thống.
        """
        return [
            {
                "id": k,
                "ten": v.ten,
                "phien_ban": v.phien_ban,
                "loai": getattr(v, "loai", "co_so"),
                "mo_ta": v.mo_ta,
                "trang_thai": "hoat_dong" if v.trang_thai_hoat_dong else "da_tat",
            }
            for k, v in self.cac_plugin_dang_chay.items()
        ]

    def nap_thu_muc(self, duong_dan_thu_muc: str) -> Dict[str, bool]:
        """
        Quét và nạp tất cả các file plugin (.py) trong một thư mục.
        Trả về kết quả nạp từng module: { ten_module: True/False }.
        """
        ket_qua: Dict[str, bool] = {}
        thu_muc = Path(duong_dan_thu_muc)
        if not thu_muc.exists() or not thu_muc.is_dir():
            logger.warning(f"[Plugin] Thư mục không tồn tại: {duong_dan_thu_muc}")
            return ket_qua

        for file_path in sorted(thu_muc.glob("*.py")):
            if file_path.name.startswith("__"):
                continue
            instance = self.nap_plugin(str(file_path))
            ket_qua[file_path.stem] = instance is not None
        return ket_qua

    def nap_plugin_chuyen_nganh(self) -> Dict[str, bool]:
        """
        Tự động quét và nạp toàn bộ các plugin chuyên ngành tích hợp sẵn (Tài chính, Y tế, Chứng khoán).
        """
        try:
            import vietnamese_ai.plugins as plugins_pkg

            thu_muc_plugins = Path(plugins_pkg.__file__).parent
            return self.nap_thu_muc(str(thu_muc_plugins))
        except Exception as e:
            logger.error(f"[Plugin] Không thể định vị gói plugins chuyên ngành: {e}")
            return {}
