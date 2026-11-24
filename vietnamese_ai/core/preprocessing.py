"""Core Preprocessing - Các bộ tiền xử lý số cốt lõi chỉ dùng NumPy (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "XuLySo": ("vietnamese_ai.preprocessing.numerical", "XuLySo"),
    "TaoDacTrung": ("vietnamese_ai.preprocessing.feature_engineering", "TaoDacTrung"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
