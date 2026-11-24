"""Hệ thống Tính toán Edge & Động cơ Suy luận C++ / Zero-GPU Native (`vietnamese_ai.edge`, Lazy PEP 562)."""

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
    "GGUFMemoryMappedStreamer": ("vietnamese_ai.edge.native_inference", "GGUFMemoryMappedStreamer"),
    "THONG_SO_LUONG_TU": ("vietnamese_ai.edge.native_inference", "THONG_SO_LUONG_TU"),
    "KIEN_TRUC_LLM_HO_TRO": ("vietnamese_ai.edge.native_inference", "KIEN_TRUC_LLM_HO_TRO"),
    "CauNoiCtypesNative": ("vietnamese_ai.edge.c_bridge", "CauNoiCtypesNative"),
    "NativeLlamaBridge": ("vietnamese_ai.edge.c_bridge", "NativeLlamaBridge"),
    "CtypesLlamaBridge": ("vietnamese_ai.edge.c_bridge", "CtypesLlamaBridge"),
    "BienTheGGUF": ("vietnamese_ai.edge.model_hub", "BienTheGGUF"),
    "ThongTinMoHinhEdge": ("vietnamese_ai.edge.model_hub", "ThongTinMoHinhEdge"),
    "DANH_MUC_MO_HINH_VIET": ("vietnamese_ai.edge.model_hub", "DANH_MUC_MO_HINH_VIET"),
    "do_ram_he_thong_gb": ("vietnamese_ai.edge.model_hub", "do_ram_he_thong_gb"),
    "dinh_dang_prompt_chat": ("vietnamese_ai.edge.model_hub", "dinh_dang_prompt_chat"),
    "QuanLyMoHinhEdge": ("vietnamese_ai.edge.model_hub", "QuanLyMoHinhEdge"),
    "EdgeModelHub": ("vietnamese_ai.edge.model_hub", "EdgeModelHub"),
    "VietnameseModelPuller": ("vietnamese_ai.edge.model_hub", "VietnameseModelPuller"),
    "KiemTraMoiTruongNative": ("vietnamese_ai.edge.native_build", "KiemTraMoiTruongNative"),
    "NativeBuildManager": ("vietnamese_ai.edge.native_build", "NativeBuildManager"),
    "thong_tin_kien_truc_cpu": ("vietnamese_ai.edge.native_build", "thong_tin_kien_truc_cpu"),
    "NodeLlamaEngine": ("vietnamese_ai.edge.node_llama", "NodeLlamaEngine"),
    "EdgeRouter": ("vietnamese_ai.edge.intelligent_router", "EdgeRouter"),
    "SecureEdgeNode": ("vietnamese_ai.edge.tee_zkp_node", "SecureEdgeNode"),
    "P2PTracker": ("vietnamese_ai.edge.p2p_network", "P2PTracker"),
    "TokenLedger": ("vietnamese_ai.edge.p2p_network", "TokenLedger"),
    "WebBrowserNode": ("vietnamese_ai.edge.wasm_node", "WebBrowserNode"),
    "WebXRNode": ("vietnamese_ai.edge.webxr_node", "WebXRNode"),
    "NutSwarmCPU": ("vietnamese_ai.edge.swarm_inference", "NutSwarmCPU"),
    "SwarmCPUNode": ("vietnamese_ai.edge.swarm_inference", "SwarmCPUNode"),
    "DieuPhoiSwarmCPU": ("vietnamese_ai.edge.swarm_inference", "DieuPhoiSwarmCPU"),
    "SwarmInferenceOrchestrator": ("vietnamese_ai.edge.swarm_inference", "SwarmInferenceOrchestrator"),
    "tao_ssl_context_server": ("vietnamese_ai.edge.swarm_inference", "tao_ssl_context_server"),
    "tao_ssl_context_client": ("vietnamese_ai.edge.swarm_inference", "tao_ssl_context_client"),
    "XacThucCumSwarm": ("vietnamese_ai.edge.swarm_auth", "XacThucCumSwarm"),
    "phat_hien_ip_tailscale": ("vietnamese_ai.edge.swarm_auth", "phat_hien_ip_tailscale"),
    "tao_ma_pin_ngau_nhien": ("vietnamese_ai.edge.swarm_auth", "tao_ma_pin_ngau_nhien"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
