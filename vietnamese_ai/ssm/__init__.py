"""Module State Space Models (SSMs) — Mamba & Jamba hỗ trợ cả PyTorch và thuần NumPy (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "MambaBlock": ("vietnamese_ai.ssm.mamba_block", "MambaBlock"),
    "SparseMoE": ("vietnamese_ai.ssm.jamba_block", "SparseMoE"),
    "JambaBlock": ("vietnamese_ai.ssm.jamba_block", "JambaBlock"),
}

__all__ = ["MambaBlock", "SparseMoE", "JambaBlock"]
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
