"""Models module - Các mô hình học máy (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "BaseModel": ("vietnamese_ai.models.base", "BaseModel"),
    "PhanLoai": ("vietnamese_ai.models.classifier", "PhanLoai"),
    "HoiQuy": ("vietnamese_ai.models.regression", "HoiQuy"),
    "PhanCum": ("vietnamese_ai.models.clustering", "PhanCum"),
    "MoHinhTapHop": ("vietnamese_ai.models.ensemble", "MoHinhTapHop"),
    "MangNron": ("vietnamese_ai.models.neural_net", "MangNron"),
    "MangSau": ("vietnamese_ai.models.mang_sau", "MangSau"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
