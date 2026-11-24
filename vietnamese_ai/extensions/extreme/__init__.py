"""Extension Extreme - C++ JIT Compiler và Kernel Fusion (Thử nghiệm, yêu cầu `g++`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "EvoJITCompiler": ("vietnamese_ai.extreme.jit_engine", "EvoJITCompiler"),
    "EvoKernelCPU": ("vietnamese_ai.extreme.cpu_kernel", "EvoKernelCPU"),
    "FusedDenseNormReLU": ("vietnamese_ai.extreme.kernel_fusion", "FusedDenseNormReLU"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
