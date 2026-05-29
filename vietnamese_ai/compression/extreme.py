import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """
    BitLinear layer mô phỏng mạng nơ-ron 1.58-bit (Ternary weights: -1, 0, 1).
    Phỏng theo kiến trúc BitNet b1.58 để tối ưu hóa cực hạn.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = self._quantize_weight_1_58_bit

    def _quantize_weight_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """Lượng tử hóa trọng số về -1, 0, 1 sử dụng Absmean Quantization."""
        scale = weight.abs().mean()

        # Ngăn chia cho 0
        scale = torch.clamp(scale, min=1e-5)

        # Chuẩn hóa trọng số
        scaled_weight = weight / scale

        # Làm tròn về số nguyên (-1, 0, 1)
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1, 1))

        # Straight-Through Estimator (STE) để đạo hàm có thể truyền ngược qua hàm round
        # Phép tính này cho quantized_weight ở forward pass, nhưng đạo hàm bằng đạo hàm của weight ở backward
        return (quantized_weight - weight).detach() + weight

    def _quantize_activation_8_bit(self, x: torch.Tensor) -> torch.Tensor:
        """Lượng tử hóa activation về int8 ([-128, 127]) để tối ưu phép nhân ma trận."""
        scale = x.abs().max() / 127.0
        scale = torch.clamp(scale, min=1e-5)

        scaled_x = x / scale
        quantized_x = torch.round(torch.clamp(scaled_x, -128, 127))

        return (quantized_x - x).detach() + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass với trọng số và activation đã được lượng tử hóa."""
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_x = self._quantize_activation_8_bit(x)

        # Trong thực tế, phép tính này sẽ được thực hiện bằng custom CUDA kernel (chỉ cần phép cộng/trừ)
        # Ở đây dùng hàm tuyến tính của PyTorch để mô phỏng
        output = F.linear(quantized_x, quantized_weight, self.bias)
        return output
