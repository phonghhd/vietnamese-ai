"""EvoKernelCPU - Nhân ma trận 1.58-bit (BitNet) tối ưu cho CPU."""

import logging
from typing import Any

try:
    import torch
    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False

logger = logging.getLogger("EvoKernelCPU")

class EvoKernelCPU:
    """
    Nhân ma trận lượng tử hoá 1.58-bit (Trọng số chỉ là -1, 0, 1).
    Bằng cách thay thế phép nhân (Multiplication) bằng phép cộng/trừ (Addition/Subtraction),
    CPU có thể thực thi suy luận cực kỳ tiết kiệm năng lượng và RAM.
    """
    
    @staticmethod
    def add_only_matmul(x: 'torch.Tensor', w_quantized: 'torch.Tensor') -> 'torch.Tensor':
        """
        Nhân ma trận Add-only cho kiến trúc 1.58-bit.
        
        Thay vì dùng phép nhân dấu phẩy động tốn kém, thuật toán sử dụng toán tử:
        Out = sum(X_pos) - sum(X_neg)
        
        Args:
            x: Tensor đầu vào (fp32 hoặc fp16), kích thước (B, In_Features).
            w_quantized: Trọng số đã lượng tử hoá (-1, 0, 1) dạng int8, kích thước (In_Features, Out_Features).
            
        Returns:
            Kết quả của phép nhân ma trận.
        """
        if not _CO_TORCH:
            raise ImportError("Yêu cầu PyTorch để chạy EvoKernelCPU.")
            
        if x.device.type != 'cpu' or w_quantized.device.type != 'cpu':
            logger.warning("EvoKernelCPU được thiết kế đặc biệt cho CPU, nhưng tensor đang ở GPU.")
            
        # Mô phỏng Add-Only thực sự:
        # Out_j = sum_{i: W_ij=1} X_i - sum_{i: W_ij=-1} X_i
        
        out = torch.zeros(x.size(0), w_quantized.size(1), device=x.device, dtype=x.dtype)
        
        # Thuật toán thuần Cộng/Trừ (Add-only)
        # Trong thực tế sản xuất, đoạn này sẽ được compile sang C++/Rust 
        # để tận dụng tối đa thanh ghi (Registers) của CPU.
        for j in range(w_quantized.size(1)):
            col = w_quantized[:, j]
            
            # Tìm vị trí các số 1 và -1 (Không bao giờ có phép nhân)
            pos_idx = (col == 1)
            neg_idx = (col == -1)
            
            if pos_idx.any():
                # Phép cộng (Add)
                out[:, j] += x[:, pos_idx].sum(dim=1)
                
            if neg_idx.any():
                # Phép trừ (Subtract)
                out[:, j] -= x[:, neg_idx].sum(dim=1)
                
        return out
