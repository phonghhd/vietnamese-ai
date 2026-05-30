"""Huấn luyện LoRA trên mạng 1.58-bit (BitLoRA)."""

import math
import torch
import torch.nn as nn
from vietnamese_ai.compression.extreme import BitLinear

class BitLoRALinear(nn.Module):
    """
    Kết hợp BitLinear (1.58-bit) và LoRA (FP32).
    Cho phép huấn luyện mô hình siêu tiết kiệm tài nguyên trên CPU/Edge.
    
    Y = (X @ W_1.58) + (X @ A @ B) * scaling
    """
    def __init__(
        self, 
        bit_linear: BitLinear, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        
        # 1. Base Layer (Đóng băng hoàn toàn)
        self.base_layer = bit_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # Lấy kích thước từ base layer (in_features, out_features)
        # Truy cập thông qua thuộc tính in_features, out_features (của nn.Linear)
        self.in_features = bit_linear.in_features
        self.out_features = bit_linear.out_features
        
        # 2. LoRA Adapters (Mạch phụ FP32 - Có thể huấn luyện)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            
            # Khởi tạo trọng số LoRA
            # lora_A được khởi tạo bằng phân phối chuẩn (Kaiming), lora_B khởi tạo bằng 0
            # Điều này đảm bảo ban đầu Y = X @ W_1.58 (Mạch phụ không ảnh hưởng)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight) 
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass kết hợp 2 luồng tính toán."""
        
        # 1. Luồng chính: Mạng 1.58-bit 
        # (Sẽ tự động kích hoạt EvoKernelCPU nếu đang chạy trên CPU)
        result = self.base_layer(x)
        
        # 2. Luồng phụ: Mạng LoRA Adapter (Mạng siêu nhỏ, tính bằng Float)
        if self.r > 0:
            # Dropout -> Linear A -> Linear B
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x.float())))
            # Scale và cộng gộp kết quả
            result = result + lora_out.type_as(result) * self.scaling
            
        return result
