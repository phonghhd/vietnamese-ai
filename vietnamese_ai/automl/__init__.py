"""AutoML module - Tự động hóa học máy và NAS (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "AutoML": ("vietnamese_ai.automl.auto_ml", "AutoML"),
    "TimKiemKienTruc": ("vietnamese_ai.automl.nas", "TimKiemKienTruc"),
}

__all__ = ["AutoML", "TimKiemKienTruc"]
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
