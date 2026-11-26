"""Model Compression - nén và tối ưu mô hình."""

from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU
from vietnamese_ai.compression.distillation import HocRutGon
from vietnamese_ai.compression.pruning import CatTiaMoHinh

try:
    from vietnamese_ai.compression.extreme import (
        BitLinear,
        BitNetTransformerBlock,
        RMSNorm,
        co_rust_bitnet,
        dong_goi_bitnet_158,
        nhan_ma_tran_bitnet_158,
    )

    _BITNET = [
        "BitLinear",
        "BitNetTransformerBlock",
        "RMSNorm",
        "co_rust_bitnet",
        "dong_goi_bitnet_158",
        "nhan_ma_tran_bitnet_158",
    ]
except ImportError:
    BitLinear = None  # type: ignore
    BitNetTransformerBlock = None  # type: ignore
    RMSNorm = None  # type: ignore
    co_rust_bitnet = None  # type: ignore
    dong_goi_bitnet_158 = None  # type: ignore
    nhan_ma_tran_bitnet_158 = None  # type: ignore
    _BITNET = []

__all__ = ["HocRutGon", "CatTiaMoHinh", "EvoKernelCPU"] + _BITNET

