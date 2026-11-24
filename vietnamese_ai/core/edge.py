"""Core Edge Runtime - Các thành phần lõi của Zero-GPU Edge Engine (Lazy PEP 562)."""

from __future__ import annotations

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "DongCoSuyLuanEdge": ("vietnamese_ai.edge.native_inference", "DongCoSuyLuanEdge"),
    "EdgeInferenceEngine": ("vietnamese_ai.edge.native_inference", "EdgeInferenceEngine"),
    "ZeroGPUInferenceEngine": ("vietnamese_ai.edge.native_inference", "ZeroGPUInferenceEngine"),
    "BoGiaiLuongTuQ4": ("vietnamese_ai.edge.native_inference", "BoGiaiLuongTuQ4"),
    "BoGiaiLuongTuGGUF": ("vietnamese_ai.edge.native_inference", "BoGiaiLuongTuGGUF"),
    "BoGiaiMaTransformerEdge": ("vietnamese_ai.edge.native_inference", "BoGiaiMaTransformerEdge"),
    "BoTachTuBPEGGUF": ("vietnamese_ai.edge.native_inference", "BoTachTuBPEGGUF"),
    "GGUFMemoryMappedStreamer": (
        "vietnamese_ai.edge.native_inference",
        "GGUFMemoryMappedStreamer",
    ),
    "THONG_SO_LUONG_TU": ("vietnamese_ai.edge.native_inference", "THONG_SO_LUONG_TU"),
    "KIEN_TRUC_LLM_HO_TRO": ("vietnamese_ai.edge.native_inference", "KIEN_TRUC_LLM_HO_TRO"),
    "CauNoiCtypesNative": ("vietnamese_ai.edge.c_bridge", "CauNoiCtypesNative"),
    "NativeLlamaBridge": ("vietnamese_ai.edge.c_bridge", "NativeLlamaBridge"),
    "CtypesLlamaBridge": ("vietnamese_ai.edge.c_bridge", "CtypesLlamaBridge"),
    "QuanLyMoHinhEdge": ("vietnamese_ai.edge.model_hub", "QuanLyMoHinhEdge"),
    "EdgeModelHub": ("vietnamese_ai.edge.model_hub", "EdgeModelHub"),
    "DANH_MUC_MO_HINH_VIET": ("vietnamese_ai.edge.model_hub", "DANH_MUC_MO_HINH_VIET"),
    "do_ram_he_thong_gb": ("vietnamese_ai.edge.model_hub", "do_ram_he_thong_gb"),
    "dinh_dang_prompt_chat": ("vietnamese_ai.edge.model_hub", "dinh_dang_prompt_chat"),
    "KiemTraMoiTruongNative": ("vietnamese_ai.edge.native_build", "KiemTraMoiTruongNative"),
    "thong_tin_kien_truc_cpu": ("vietnamese_ai.edge.native_build", "thong_tin_kien_truc_cpu"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
