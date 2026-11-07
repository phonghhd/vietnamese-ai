"""Core - Engine, Pipeline, Cross-validation, Hyperparameter search (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "Engine": ("vietnamese_ai.core.engine", "Engine"),
    "Pipeline": ("vietnamese_ai.core.pipeline", "Pipeline"),
    "KiemDinhCheo": ("vietnamese_ai.core.cross_validation", "KiemDinhCheo"),
    "TimKiemThamSo": ("vietnamese_ai.core.hyperparameter", "TimKiemThamSo"),
    "QuyTrinhDauCuoi": ("vietnamese_ai.core.end_to_end", "QuyTrinhDauCuoi"),
    "HeSinhThaiAI": ("vietnamese_ai.core.end_to_end", "HeSinhThaiAI"),
    "EndToEndWorkflow": ("vietnamese_ai.core.end_to_end", "EndToEndWorkflow"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
