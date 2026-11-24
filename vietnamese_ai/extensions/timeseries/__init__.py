"""Extension TimeSeries - Dự đoán chuỗi thời gian (yêu cầu extra: `ml`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "DuDoanChuoiThoiGian": ("vietnamese_ai.timeseries.forecaster", "DuDoanChuoiThoiGian"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
