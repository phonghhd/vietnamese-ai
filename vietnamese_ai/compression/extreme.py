"""Kiến trúc BitNet b1.58 (Ternary Weights -1, 0, +1) hỗ trợ song song PyTorch và thuần NumPy."""

import math
from typing import Any, Tuple

import numpy as np

from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False


if _CO_TORCH:

    class BitLinear(nn.Linear):
        """
        BitLinear layer mạng nơ-ron 1.58-bit (Ternary weights: -1, 0, 1) chuẩn BitNet b1.58.
        """

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__(in_features, out_features, bias)
            self.weight_quantizer = self._quantize_weight_1_58_bit

        def _quantize_weight_1_58_bit(self, weight: "torch.Tensor") -> "torch.Tensor":
            scale = torch.clamp(weight.abs().mean(), min=1e-5)
            quantized_weight = torch.round(torch.clamp(weight / scale, -1, 1))
            return (quantized_weight - weight).detach() + weight

        def _quantize_activation_8_bit(self, x: "torch.Tensor") -> "torch.Tensor":
            scale = torch.clamp(x.abs().max() / 127.0, min=1e-5)
            quantized_x = torch.round(torch.clamp(x / scale, -128, 127))
            return (quantized_x - x).detach() + x

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            quantized_weight = self.weight_quantizer(self.weight)
            quantized_x = self._quantize_activation_8_bit(x)
            if x.device.type == "cpu":
                output = EvoKernelCPU.add_only_matmul(quantized_x, quantized_weight.t())
                if self.bias is not None:
                    output = output + self.bias
            else:
                output = F.linear(quantized_x, quantized_weight, self.bias)
            return output

        def dong_goi_2bit(self) -> bytes:
            """Đóng gói 4 trọng số tam phân `{-1, 0, +1}` vào 1 byte (giảm 16x dung lượng so với FP32)."""
            w_np = self.weight.detach().cpu().numpy()
            scale = max(float(np.mean(np.abs(w_np))), 1e-5)
            q_w = np.clip(np.round(w_np / scale), -1.0, 1.0).astype(np.int8)
            shifted = (q_w.ravel() + 1).astype(np.uint8)
            pad_len = (-len(shifted)) % 4
            if pad_len:
                shifted = np.pad(shifted, (0, pad_len), constant_values=1)
            g = shifted.reshape(-1, 4)
            packed = (g[:, 0] << 6) | (g[:, 1] << 4) | (g[:, 2] << 2) | g[:, 3]
            return packed.tobytes()

    class RMSNorm(nn.Module):
        """Root Mean Square Normalization chuẩn Llama / BitNet."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return normed.type_as(x) * self.weight

    class BitNetTransformerBlock(nn.Module):
        """Khối Transformer sử dụng hoàn toàn BitLinear 1.58-bit + RMSNorm + SwiGLU."""

        def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.norm1 = RMSNorm(dim)
            self.q_proj = BitLinear(dim, dim, bias=False)
            self.k_proj = BitLinear(dim, dim, bias=False)
            self.v_proj = BitLinear(dim, dim, bias=False)
            self.o_proj = BitLinear(dim, dim, bias=False)
            self.norm2 = RMSNorm(dim)
            hidden_dim = int(dim * mlp_ratio)
            self.gate_proj = BitLinear(dim, hidden_dim, bias=False)
            self.up_proj = BitLinear(dim, hidden_dim, bias=False)
            self.down_proj = BitLinear(hidden_dim, dim, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.norm1(x)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            B, T, C = q.size()
            q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.o_proj(context)
            h = self.norm2(x)
            x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
            return x

else:

    class BitLinear:  # type: ignore[no-redef]
        """
        BitLinear 1.58-bit thuần NumPy (không cần PyTorch).
        Lượng tử hóa trọng số về ba mức `{-1, 0, +1}` bằng Absmean Quantization
        và thực thi nhân ma trận chỉ dùng phép cộng/trừ qua `EvoKernelCPU`.
        """

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            self.in_features = in_features
            self.out_features = out_features
            bound = 1.0 / math.sqrt(max(1, in_features))
            self.weight = np.random.uniform(-bound, bound, size=(out_features, in_features))
            self.bias = np.zeros(out_features, dtype=np.float64) if bias else None

        def _quantize_weight_1_58_bit(self, weight: np.ndarray) -> Tuple[np.ndarray, float]:
            scale = max(float(np.mean(np.abs(weight))), 1e-5)
            q_w = np.clip(np.round(weight / scale), -1.0, 1.0).astype(np.int8)
            return q_w, scale

        def forward(self, x: Any) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            scale_w = max(float(np.mean(np.abs(self.weight))), 1e-5)
            q_w = np.clip(np.round(self.weight / scale_w), -1.0, 1.0).astype(np.int8)

            scale_x = max(float(np.max(np.abs(x_arr)) / 127.0), 1e-5)
            q_x = np.clip(np.round(x_arr / scale_x), -128.0, 127.0)

            out = EvoKernelCPU.add_only_matmul(q_x, q_w.T) * (scale_w * scale_x)
            if self.bias is not None:
                out = out + self.bias
            return out

        def dong_goi_2bit(self) -> bytes:
            """Đóng gói 4 trọng số tam phân `{-1, 0, +1}` vào 1 byte (giảm 16x dung lượng so với FP32)."""
            q_w, _ = self._quantize_weight_1_58_bit(self.weight)
            shifted = (q_w.ravel() + 1).astype(np.uint8)
            pad_len = (-len(shifted)) % 4
            if pad_len:
                shifted = np.pad(shifted, (0, pad_len), constant_values=1)
            g = shifted.reshape(-1, 4)
            packed = (g[:, 0] << 6) | (g[:, 1] << 4) | (g[:, 2] << 2) | g[:, 3]
            return packed.tobytes()

        def __call__(self, x: Any) -> np.ndarray:
            return self.forward(x)

    class RMSNorm:  # type: ignore[no-redef]
        """RMSNorm thuần NumPy."""

        def __init__(self, dim: int, eps: float = 1e-6):
            self.dim = dim
            self.eps = eps
            self.weight = np.ones(dim, dtype=np.float64)

        def forward(self, x: Any) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            rms = np.sqrt(np.mean(x_arr**2, axis=-1, keepdims=True) + self.eps)
            return (x_arr / rms) * self.weight

        def __call__(self, x: Any) -> np.ndarray:
            return self.forward(x)

    class BitNetTransformerBlock:  # type: ignore[no-redef]
        """Khối Transformer BitNet b1.58 thuần NumPy (RMSNorm + Ternary Attention + SwiGLU)."""

        def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
            self.dim = dim
            self.num_heads = num_heads
            self.norm1 = RMSNorm(dim)
            self.q_proj = BitLinear(dim, dim, bias=False)
            self.k_proj = BitLinear(dim, dim, bias=False)
            self.v_proj = BitLinear(dim, dim, bias=False)
            self.o_proj = BitLinear(dim, dim, bias=False)
            self.norm2 = RMSNorm(dim)
            hidden_dim = int(dim * mlp_ratio)
            self.gate_proj = BitLinear(dim, hidden_dim, bias=False)
            self.up_proj = BitLinear(dim, hidden_dim, bias=False)
            self.down_proj = BitLinear(hidden_dim, dim, bias=False)

        def forward(self, x: Any) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            h = self.norm1(x_arr)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            B, T, C = q.shape
            d_k = C // self.num_heads
            q = q.reshape(B, T, self.num_heads, d_k).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.num_heads, d_k).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.num_heads, d_k).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(max(1, d_k))
            scores -= np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            context = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            x_arr = x_arr + self.o_proj(context)

            h2 = self.norm2(x_arr)
            gate_in = self.gate_proj(h2)
            silu_gate = gate_in / (1.0 + np.exp(-np.clip(gate_in, -30.0, 30.0)))
            x_arr = x_arr + self.down_proj(silu_gate * self.up_proj(h2))
            return x_arr

        def __call__(self, x: Any) -> np.ndarray:
            return self.forward(x)
