"""Model Compression - nén và tối ưu mô hình."""

from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU
from vietnamese_ai.compression.distillation import HocRutGon
from vietnamese_ai.compression.pruning import CatTiaMoHinh

try:
    from vietnamese_ai.compression.extreme import BitLinear, BitNetTransformerBlock, RMSNorm

    _BITNET = ["BitLinear", "BitNetTransformerBlock", "RMSNorm"]
except ImportError:
    BitLinear = None  # type: ignore
    BitNetTransformerBlock = None  # type: ignore
    RMSNorm = None  # type: ignore
    _BITNET = []

__all__ = ["HocRutGon", "CatTiaMoHinh", "EvoKernelCPU"] + _BITNET

