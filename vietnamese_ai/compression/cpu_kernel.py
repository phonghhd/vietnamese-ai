"""EvoKernelCPU - Nhân ma trận 1.58-bit (BitNet) tối ưu cho CPU (hỗ trợ cả NumPy và PyTorch)."""

import logging
from typing import Any

import numpy as np

try:
    import torch

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False

logger = logging.getLogger("EvoKernelCPU")


class EvoKernelCPU:
    """
    Nhân ma trận lượng tử hoá 1.58-bit (Trọng số chỉ là -1, 0, 1).
    Bằng cách thay thế phép nhân dấu phẩy động bằng phép cộng/trừ thuần túy (Add-only),
    CPU có thể thực thi suy luận cực kỳ tiết kiệm năng lượng và bộ nhớ trên cả NumPy lẫn PyTorch.
    """

    @staticmethod
    def add_only_matmul(x: Any, w_quantized: Any) -> Any:
        """
        Nhân ma trận Add-only cho kiến trúc 1.58-bit:
            Out_j = sum_{i: W_ij = +1} X_i - sum_{i: W_ij = -1} X_i

        Hỗ trợ đầu vào 2D `(B, In)` hoặc 3D `(B, T, In)` trên cả `np.ndarray` và `torch.Tensor`.
        """
        if isinstance(x, np.ndarray) or not _CO_TORCH:
            x_np = np.asarray(x, dtype=np.float64)
            w_np = np.asarray(w_quantized)
            orig_shape = x_np.shape
            x_2d = x_np.reshape(-1, orig_shape[-1])
            out_2d = np.zeros((x_2d.shape[0], w_np.shape[1]), dtype=x_np.dtype)
            for j in range(w_np.shape[1]):
                col = w_np[:, j]
                pos_idx = col == 1
                neg_idx = col == -1
                if np.any(pos_idx):
                    out_2d[:, j] += np.sum(x_2d[:, pos_idx], axis=1)
                if np.any(neg_idx):
                    out_2d[:, j] -= np.sum(x_2d[:, neg_idx], axis=1)
            return out_2d.reshape(*orig_shape[:-1], w_np.shape[1])

        if x.device.type != "cpu" or w_quantized.device.type != "cpu":
            logger.warning("EvoKernelCPU được thiết kế đặc biệt cho CPU, nhưng tensor đang ở GPU.")

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out = torch.zeros(x_2d.size(0), w_quantized.size(1), device=x.device, dtype=x.dtype)

        for j in range(w_quantized.size(1)):
            col = w_quantized[:, j]
            pos_idx = col == 1
            neg_idx = col == -1
            if pos_idx.any():
                out[:, j] += x_2d[:, pos_idx].sum(dim=1)
            if neg_idx.any():
                out[:, j] -= x_2d[:, neg_idx].sum(dim=1)

        return out.reshape(*orig_shape[:-1], w_quantized.size(1))
