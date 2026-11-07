"""Utils module - Các công cụ tiện ích & Quản lý phụ thuộc siêu nhẹ (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import DependencyManager, QuanLyPhuThuoc, tao_lazy_getattr

__all__ = [
    "Logger",
    "Metrics",
    "Validator",
    "LuuTai",
    "QuanLyPhuThuoc",
    "DependencyManager",
    "tao_lazy_getattr",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "Logger": ("vietnamese_ai.utils.logger", "Logger"),
        "Metrics": ("vietnamese_ai.utils.metrics", "Metrics"),
        "Validator": ("vietnamese_ai.utils.validators", "Validator"),
        "LuuTai": ("vietnamese_ai.utils.io_utils", "LuuTai"),
    },
    globals(),
)
