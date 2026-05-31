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
        """Forward pass tự thích nghi (Adaptive Forward): Chọn CPU hoặc GPU Kernel."""
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_x = self._quantize_activation_8_bit(x)

        # Tự động điều phối (Adaptive Dispatch) dựa trên phần cứng thực tế
        if x.device.type == "cpu":
            # Máy yếu hoặc không có GPU: Sử dụng EvoKernelCPU (Add-only MatMul)
            from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU

            # Hàm tuyến tính y = x @ W^T + b. quantized_weight có shape (Out, In)
            # Do đó truyền quantized_weight.t() (shape In, Out) vào add_only_matmul
            output = EvoKernelCPU.add_only_matmul(quantized_x, quantized_weight.t())

            if self.bias is not None:
                output += self.bias
        else:
            # Máy mạnh (GPU): Dùng CUDA native / Tensor Cores siêu tốc
            output = F.linear(quantized_x, quantized_weight, self.bias)

        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Tối ưu hơn LayerNorm, rất phù hợp với kiến trúc Llama/BitNet.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class BitNetTransformerBlock(nn.Module):
    """
    Khối Transformer sử dụng hoàn toàn BitLinear (1.58-bit) cho các lớp dense,
    kết hợp với RMSNorm và SwiGLU activation (giống Llama).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Self-Attention
        self.norm1 = RMSNorm(dim)
        self.q_proj = BitLinear(dim, dim, bias=False)
        self.k_proj = BitLinear(dim, dim, bias=False)
        self.v_proj = BitLinear(dim, dim, bias=False)
        self.o_proj = BitLinear(dim, dim, bias=False)

        # Feed Forward (SwiGLU)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.gate_proj = BitLinear(dim, hidden_dim, bias=False)
        self.up_proj = BitLinear(dim, hidden_dim, bias=False)
        self.down_proj = BitLinear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Attention (giản lược logic QKV computation)
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Mô phỏng self-attention cơ bản
        # (Trong thực tế cần reshape theo num_heads, áp dụng RoPE và FlashAttention)
        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)

        x = x + self.o_proj(context)

        # 2. Feed Forward Network (SwiGLU)
        h = self.norm2(x)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)

        return x
