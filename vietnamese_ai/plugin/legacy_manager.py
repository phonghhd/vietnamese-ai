"""PluginManager - Hệ thống plugin mở rộng."""

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List

from vietnamese_ai.utils.logger import Logger


class PluginManager:
    """
    Hệ thống plugin mở rộng cho Vietnamese AI Framework.

    Tính năng:
    - Đăng ký plugin tùy chỉnh (model, preprocessor, metric, ...)
    - Tải plugin từ file Python
    - Quản lý lifecycle plugin
    - Hook system (pre_train, post_train, pre_predict, post_predict)

    Sử dụng:
        >>> pm = PluginManager()

        >>> # Đăng ký plugin tùy chỉnh
        >>> @pm.dang_ky_plugin("my_metric")
        ... def my_metric(y_true, y_pred):
        ...     return float(np.mean(y_true == y_pred))

        >>> # Sử dụng hook
        >>> @pm.hook("pre_train")
        ... def log_before_train(X, y):
        ...     print(f"Training với {len(X)} mẫu")

        >>> # Tải plugin từ file
        >>> pm.tai_plugin("path/to/my_plugin.py")
    """

    def __init__(self):
        self.logger = Logger("PluginManager")
        self._plugins: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_train": [],
            "post_train": [],
            "pre_predict": [],
            "post_predict": [],
            "pre_evaluate": [],
            "post_evaluate": [],
        }

    def dang_ky_plugin(self, ten: str, loai: str = "custom") -> Callable:
        """
        Decorator để đăng ký plugin.

        Args:
            ten: Tên plugin
            Loai: Loại plugin (model, preprocessor, metric, custom)
        """
        def decorator(func):
            self._plugins[ten] = {
                "ten": ten,
                "loai": loai,
                "func": func,
                "hoat_dong": True,
            }
            self.logger.info(f"Đã đăng ký plugin: {ten} ({loai})")
            return func
        return decorator

    def hook(self, ten_hook: str) -> Callable:
        """
        Decorator để đăng ký hook.

        Args:
            ten_hook: Tên hook (pre_train, post_train, pre_predict, post_predict)
        """
        def decorator(func):
            if ten_hook not in self._hooks:
                self._hooks[ten_hook] = []
            self._hooks[ten_hook].append(func)
            self.logger.info(f"Đã đăng ký hook: {ten_hook}")
            return func
        return decorator

    def chay_hook(self, ten_hook: str, **kwargs) -> None:
        """Chạy tất cả hooks của một loại."""
        for hook_func in self._hooks.get(ten_hook, []):
            try:
                hook_func(**kwargs)
            except Exception as e:
                self.logger.warning(f"Hook {ten_hook} lỗi: {e}")

    def lay_plugin(self, ten: str) -> Any:
        """Lấy plugin theo tên."""
        if ten not in self._plugins:
            raise KeyError(f"Không tìm thấy plugin: {ten}")
        return self._plugins[ten]["func"]

    def tai_plugin(self, duong_dan: str) -> None:
        """
        Tải plugin từ file Python.

        File plugin phải có hàm register() để đăng ký các plugins.
        """
        duong_dan = Path(duong_dan)
        if not duong_dan.exists():
            raise FileNotFoundError(f"Không tìm thấy file plugin: {duong_dan}")

        spec = importlib.util.spec_from_file_location("plugin_module", str(duong_dan))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "register"):
            module.register(self)
            self.logger.info(f"Đã tải plugin từ: {duong_dan}")
        else:
            self.logger.warning(f"Plugin không có hàm register(): {duong_dan}")

    def tai_plugin_tu_thu_muc(self, duong_dan: str) -> None:
        """Tải tất cả plugins từ một thư mục."""
        thu_muc = Path(duong_dan)
        if not thu_muc.exists():
            return

        for file in thu_muc.glob("*.py"):
            if file.name.startswith("_"):
                continue
            try:
                self.tai_plugin(str(file))
            except Exception as e:
                self.logger.warning(f"Không thể tải plugin {file}: {e}")

    def danh_sach(self) -> List[Dict]:
        """Liệt kê tất cả plugins."""
        return [
            {"ten": p["ten"], "loai": p["loai"], "hoat_dong": p["hoat_dong"]}
            for p in self._plugins.values()
        ]

    def danh_sach_hooks(self) -> Dict[str, int]:
        """Liệt kê số lượng hooks theo loại."""
        return {ten: len(funcs) for ten, funcs in self._hooks.items()}

    def tat_plugin(self, ten: str) -> None:
        """Tắt plugin."""
        if ten in self._plugins:
            self._plugins[ten]["hoat_dong"] = False

    def bat_plugin(self, ten: str) -> None:
        """Bật plugin."""
        if ten in self._plugins:
            self._plugins[ten]["hoat_dong"] = True

    def xoa_plugin(self, ten: str) -> None:
        """Xóa plugin."""
        if ten in self._plugins:
            del self._plugins[ten]
            self.logger.info(f"Đã xóa plugin: {ten}")

    def xoa_tat_ca_hooks(self) -> None:
        """Xóa tất cả hooks."""
        for ten in self._hooks:
            self._hooks[ten] = []
