"""Core - Lõi Engine, Pipeline, Cross-Validation, Hyperparameter Search, Numerical Preprocessing & Zero-GPU Edge Engine (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "Engine": ("vietnamese_ai.core.engine", "Engine"),
    "Pipeline": ("vietnamese_ai.core.pipeline", "Pipeline"),
    "KiemDinhCheo": ("vietnamese_ai.core.cross_validation", "KiemDinhCheo"),
    "TimKiemThamSo": ("vietnamese_ai.core.hyperparameter", "TimKiemThamSo"),
    "QuyTrinhDauCuoi": ("vietnamese_ai.core.end_to_end", "QuyTrinhDauCuoi"),
    "HeSinhThaiAI": ("vietnamese_ai.core.end_to_end", "HeSinhThaiAI"),
    "EndToEndWorkflow": ("vietnamese_ai.core.end_to_end", "EndToEndWorkflow"),
    "XuLySo": ("vietnamese_ai.preprocessing.numerical", "XuLySo"),
    "TaoDacTrung": ("vietnamese_ai.preprocessing.feature_engineering", "TaoDacTrung"),
    "DongCoSuyLuanEdge": ("vietnamese_ai.edge.native_inference", "DongCoSuyLuanEdge"),
    "EdgeInferenceEngine": ("vietnamese_ai.edge.native_inference", "EdgeInferenceEngine"),
    "ZeroGPUInferenceEngine": ("vietnamese_ai.edge.native_inference", "ZeroGPUInferenceEngine"),
    "BoGiaiLuongTuQ4": ("vietnamese_ai.edge.native_inference", "BoGiaiLuongTuQ4"),
    "GGUFMemoryMappedStreamer": (
        "vietnamese_ai.edge.native_inference",
        "GGUFMemoryMappedStreamer",
    ),
    "CauNoiCtypesNative": ("vietnamese_ai.edge.c_bridge", "CauNoiCtypesNative"),
    "NativeLlamaBridge": ("vietnamese_ai.edge.c_bridge", "NativeLlamaBridge"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
