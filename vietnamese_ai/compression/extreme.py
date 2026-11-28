"""Kiến trúc BitNet b1.58 (Ternary Weights -1, 0, +1) hỗ trợ song song PyTorch, thuần NumPy và kernel 2-bit Rust."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU


def _nap_rust_core() -> Any:
    """Nạp mô-đun Rust native ưu tiên bản hỗ trợ kernel BitNet 1.58-bit 2-bit."""
    candidates = []
    try:
        from vietnamese_ai import vietnamese_ai_rust as _pkg_rust  # type: ignore

        candidates.append(_pkg_rust)
    except ImportError:
        pass
    try:
        import vietnamese_ai_rust as _top_rust  # type: ignore

        candidates.append(_top_rust)
    except ImportError:
        pass
    for mod in candidates:
        if hasattr(mod, "fast_bitnet_158_pack_bytes") and hasattr(mod, "fast_bitnet_158_matmul_bytes"):
            return mod
    return candidates[0] if candidates else None


_rust_core: Any = _nap_rust_core()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False


def co_rust_bitnet() -> bool:
    """Kiểm tra lõi Rust có sẵn kernel đóng gói và nhân ma trận BitNet 1.58-bit 2-bit hay không."""
    return (
        _rust_core is not None
        and hasattr(_rust_core, "fast_bitnet_158_pack_bytes")
        and hasattr(_rust_core, "fast_bitnet_158_matmul_bytes")
    )


def dong_goi_bitnet_158(weights: np.ndarray, bat_rust: bool = True) -> Tuple[bytes, float]:
    """Đóng gói ma trận trọng số `(out_features, in_features)` sang 2-bit tam phân theo từng hàng (giảm 16x RAM so với FP32)."""
    w_arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float32))
    if w_arr.ndim != 2:
        raise ValueError(f"Trọng số BitNet phải là ma trận 2 chiều, nhận được shape={w_arr.shape}")
    out_features, in_features = int(w_arr.shape[0]), int(w_arr.shape[1])
    if bat_rust and co_rust_bitnet():
        packed_b, scale = _rust_core.fast_bitnet_158_pack_bytes(
            w_arr.tobytes(), out_features, in_features
        )
        return bytes(packed_b), float(scale)

    scale = max(float(np.mean(np.abs(w_arr))), 1e-5)
    q_w = np.clip(np.round(w_arr / scale), -1.0, 1.0).astype(np.int8)
    shifted = (q_w + 1).astype(np.uint8)
    pad_cols = (-in_features) % 4
    if pad_cols:
        shifted = np.pad(shifted, ((0, 0), (0, pad_cols)), constant_values=1)
    g = shifted.reshape(out_features, -1, 4)
    packed = (g[:, :, 0] << 6) | (g[:, :, 1] << 4) | (g[:, :, 2] << 2) | g[:, :, 3]
    return np.ascontiguousarray(packed, dtype=np.uint8).tobytes(), scale


def nhan_ma_tran_bitnet_158(
    x: np.ndarray,
    packed_w: bytes,
    in_features: int,
    out_features: int,
    scale: float,
    bat_rust: bool = True,
) -> np.ndarray:
    """Thực thi nhân ma trận BitNet 1.58-bit trực tiếp từ trọng số nén 2-bit không dùng phép nhân trọng số."""
    x_arr = np.asarray(x, dtype=np.float32)
    orig_shape = x_arr.shape
    if x_arr.ndim == 1:
        x_2d = x_arr.reshape(1, -1)
    else:
        x_2d = x_arr.reshape(-1, int(orig_shape[-1]))
    if int(x_2d.shape[1]) != int(in_features):
        raise ValueError(
            f"Chiều đầu vào không khớp BitNet ({x_2d.shape[1]} != {in_features})"
        )
    x_c = np.ascontiguousarray(x_2d, dtype=np.float32)
    n_rows = int(x_c.shape[0])

    if bat_rust and co_rust_bitnet():
        raw_out = _rust_core.fast_bitnet_158_matmul_bytes(
            x_c.tobytes(),
            packed_w,
            n_rows,
            int(in_features),
            int(out_features),
            float(scale),
        )
        y_2d = np.frombuffer(bytes(raw_out), dtype=np.float32).reshape(n_rows, int(out_features)).copy()
    else:
        packed_cols = (int(in_features) + 3) // 4
        raw_u8 = np.frombuffer(packed_w, dtype=np.uint8).reshape(int(out_features), packed_cols)
        unpacked = np.empty((int(out_features), packed_cols * 4), dtype=np.int8)
        unpacked[:, 0::4] = ((raw_u8 >> 6) & 0x03).astype(np.int8) - 1
        unpacked[:, 1::4] = ((raw_u8 >> 4) & 0x03).astype(np.int8) - 1
        unpacked[:, 2::4] = ((raw_u8 >> 2) & 0x03).astype(np.int8) - 1
        unpacked[:, 3::4] = (raw_u8 & 0x03).astype(np.int8) - 1
        q_w = unpacked[:, : int(in_features)]
        pos_mask = (q_w == 1).astype(np.float32).T
        neg_mask = (q_w == -1).astype(np.float32).T
        y_2d = ((x_c @ pos_mask) - (x_c @ neg_mask)) * np.float32(scale)

    if len(orig_shape) == 1:
        return y_2d.reshape(int(out_features))
    return y_2d.reshape(*orig_shape[:-1], int(out_features))


if _CO_TORCH:

    class BitLinear(nn.Linear):
        """
        BitLinear layer mạng nơ-ron 1.58-bit (Ternary weights: -1, 0, 1) chuẩn BitNet b1.58.
        """

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__(in_features, out_features, bias)
            self.weight_quantizer = self._quantize_weight_1_58_bit
            self._packed_2bit_cache: Optional[bytes] = None
            self._packed_scale_cache: float = 0.0
            self._packed_weight_version: Optional[int] = None
            self._da_nen_tai_cho: bool = False

        def _quantize_weight_1_58_bit(self, weight: "torch.Tensor") -> "torch.Tensor":
            scale = torch.clamp(weight.abs().mean(), min=1e-5)
            quantized_weight = torch.round(torch.clamp(weight / scale, -1, 1))
            return (quantized_weight - weight).detach() + weight

        def _quantize_activation_8_bit(self, x: "torch.Tensor") -> "torch.Tensor":
            scale = torch.clamp(x.abs().max() / 127.0, min=1e-5)
            quantized_x = torch.round(torch.clamp(x / scale, -128, 127))
            return (quantized_x - x).detach() + x

        def forward(self, x: Any) -> Any:
            la_numpy = isinstance(x, np.ndarray)
            if self._da_nen_tai_cho or self.weight is None:
                out_np = self.suy_luan_2bit(x)
                if la_numpy:
                    return out_np
                out_t = torch.from_numpy(out_np)
                if hasattr(x, "device"):
                    out_t = out_t.to(x.device)
                return out_t
            if la_numpy:
                x_tensor = torch.from_numpy(x).to(dtype=self.weight.dtype)
            else:
                x_tensor = x
            scale_w = torch.clamp(self.weight.abs().mean(), min=1e-5)
            quantized_weight = self.weight_quantizer(self.weight)
            scale_x = torch.clamp(x_tensor.abs().max() / 127.0, min=1e-5)
            quantized_x = self._quantize_activation_8_bit(x_tensor)
            output = F.linear(quantized_x, quantized_weight, None) * (scale_w * scale_x)
            if self.bias is not None:
                output = output + self.bias
            if la_numpy:
                return output.detach().cpu().numpy()
            return output

        def dong_goi_2bit(self) -> bytes:
            """Đóng gói 4 trọng số tam phân `{-1, 0, +1}` vào 1 byte (giảm 16x dung lượng so với FP32)."""
            packed, _ = self.dong_goi_2bit_kem_ty_le()
            return packed

        def dong_goi_2bit_kem_ty_le(self, bat_rust: bool = True) -> Tuple[bytes, float]:
            """Đóng gói ma trận trọng số sang 2-bit theo từng hàng kèm hệ số tỷ lệ absmean (có cache bộ đệm)."""
            if self._da_nen_tai_cho or self.weight is None:
                if self._packed_2bit_cache is None:
                    raise RuntimeError("Trọng số đã bị giải phóng nhưng không tìm thấy bộ đệm 2-bit.")
                return self._packed_2bit_cache, self._packed_scale_cache
            cur_ver = getattr(self.weight, "_version", None)
            if self._packed_2bit_cache is not None and self._packed_weight_version == cur_ver:
                return self._packed_2bit_cache, self._packed_scale_cache
            w_np = self.weight.detach().cpu().numpy()
            packed, scale = dong_goi_bitnet_158(w_np, bat_rust=bat_rust)
            self._packed_2bit_cache = packed
            self._packed_scale_cache = float(scale)
            self._packed_weight_version = cur_ver
            return packed, float(scale)

        def nen_tai_cho(self, bat_rust: bool = True) -> Dict[str, Any]:
            """Đóng gói trọng số sang 2-bit và giải phóng ma trận FP32 `self.weight` để tiết kiệm thực sự 16x RAM suy luận."""
            packed, scale = self.dong_goi_2bit_kem_ty_le(bat_rust=bat_rust)
            self.register_parameter("weight", None)
            self._da_nen_tai_cho = True
            fp32_bytes = int(self.in_features * self.out_features * 4)
            return {
                "da_nen_tai_cho": True,
                "fp32_bytes_giai_phong": fp32_bytes,
                "packed_2bit_bytes": len(packed),
                "ti_le_nen": round(fp32_bytes / max(1, len(packed)), 2),
                "ty_le_absmean": round(scale, 6),
            }

        def suy_luan_2bit(self, x: Any, bat_rust: bool = True) -> np.ndarray:
            """Suy luận trực tiếp trên trọng số nén 2-bit đã cache (không giải nén toàn bộ ma trận)."""
            x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
            packed, scale = self.dong_goi_2bit_kem_ty_le(bat_rust=bat_rust)
            out = nhan_ma_tran_bitnet_158(
                x_np, packed, self.in_features, self.out_features, scale, bat_rust=bat_rust
            )
            if self.bias is not None:
                out = out + self.bias.detach().cpu().numpy().astype(np.float32)
            return out

        def thong_ke_nen(self) -> Dict[str, Any]:
            """Thống kê dung lượng nén 2-bit thực tế so với trọng số FP32 gốc."""
            packed, scale = self.dong_goi_2bit_kem_ty_le()
            fp32_bytes = int(self.in_features * self.out_features * 4)
            return {
                "in_features": int(self.in_features),
                "out_features": int(self.out_features),
                "fp32_bytes": fp32_bytes,
                "packed_2bit_bytes": len(packed),
                "ti_le_nen": round(fp32_bytes / max(1, len(packed)), 2),
                "ty_le_absmean": round(scale, 6),
                "dung_rust": co_rust_bitnet(),
                "da_nen_tai_cho": self._da_nen_tai_cho,
            }

    class RMSNorm(nn.Module):
        """Root Mean Square Normalization chuẩn Llama / BitNet."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any) -> Any:
            la_numpy = isinstance(x, np.ndarray)
            if la_numpy:
                x_tensor = torch.from_numpy(x)
            else:
                x_tensor = x
            normed = x_tensor.float() * torch.rsqrt(x_tensor.float().pow(2).mean(-1, keepdim=True) + self.eps)
            out = normed.type_as(x_tensor) * self.weight
            if la_numpy:
                return out.detach().cpu().numpy()
            return out

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

        def forward(self, x: Any) -> Any:
            la_numpy = isinstance(x, np.ndarray)
            if la_numpy:
                x_tensor = torch.from_numpy(x)
            else:
                x_tensor = x
            h = self.norm1(x_tensor)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            B, T, C = q.size()
            q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
            x_res = x_tensor + self.o_proj(context)
            h2 = self.norm2(x_res)
            out = x_res + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
            if la_numpy:
                return out.detach().cpu().numpy()
            return out

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
            self.weight: Optional[np.ndarray] = np.random.uniform(-bound, bound, size=(out_features, in_features))
            self.bias = np.zeros(out_features, dtype=np.float64) if bias else None
            self._packed_2bit_cache: Optional[bytes] = None
            self._packed_scale_cache: float = 0.0
            self._packed_weight_sig: Optional[bytes] = None
            self._da_nen_tai_cho: bool = False

        def _quantize_weight_1_58_bit(self, weight: np.ndarray) -> Tuple[np.ndarray, float]:
            scale = max(float(np.mean(np.abs(weight))), 1e-5)
            q_w = np.clip(np.round(weight / scale), -1.0, 1.0).astype(np.int8)
            return q_w, scale

        def forward(self, x: Any) -> np.ndarray:
            if self._da_nen_tai_cho or self.weight is None:
                return self.suy_luan_2bit(x)
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
            packed, _ = self.dong_goi_2bit_kem_ty_le()
            return packed

        def dong_goi_2bit_kem_ty_le(self, bat_rust: bool = True) -> Tuple[bytes, float]:
            """Đóng gói ma trận trọng số sang 2-bit theo từng hàng kèm hệ số tỷ lệ absmean (có cache bộ đệm)."""
            if self._da_nen_tai_cho or self.weight is None:
                if self._packed_2bit_cache is None:
                    raise RuntimeError("Trọng số đã bị giải phóng nhưng không tìm thấy bộ đệm 2-bit.")
                return self._packed_2bit_cache, self._packed_scale_cache
            w_f32 = np.ascontiguousarray(self.weight, dtype=np.float32)
            sig = w_f32.tobytes()
            if self._packed_2bit_cache is not None and self._packed_weight_sig == sig:
                return self._packed_2bit_cache, self._packed_scale_cache
            packed, scale = dong_goi_bitnet_158(w_f32, bat_rust=bat_rust)
            self._packed_2bit_cache = packed
            self._packed_scale_cache = float(scale)
            self._packed_weight_sig = sig
            return packed, float(scale)

        def nen_tai_cho(self, bat_rust: bool = True) -> Dict[str, Any]:
            """Đóng gói trọng số sang 2-bit và giải phóng ma trận FP32/FP64 `self.weight` để tiết kiệm thực sự 16x RAM suy luận."""
            packed, scale = self.dong_goi_2bit_kem_ty_le(bat_rust=bat_rust)
            self.weight = None
            self._packed_weight_sig = None
            self._da_nen_tai_cho = True
            fp32_bytes = int(self.in_features * self.out_features * 4)
            return {
                "da_nen_tai_cho": True,
                "fp32_bytes_giai_phong": fp32_bytes,
                "packed_2bit_bytes": len(packed),
                "ti_le_nen": round(fp32_bytes / max(1, len(packed)), 2),
                "ty_le_absmean": round(scale, 6),
            }

        def suy_luan_2bit(self, x: Any, bat_rust: bool = True) -> np.ndarray:
            """Suy luận trực tiếp trên trọng số nén 2-bit đã cache (dùng kernel Rust không phép nhân trọng số khi có)."""
            packed, scale = self.dong_goi_2bit_kem_ty_le(bat_rust=bat_rust)
            out = nhan_ma_tran_bitnet_158(
                np.asarray(x, dtype=np.float32),
                packed,
                self.in_features,
                self.out_features,
                scale,
                bat_rust=bat_rust,
            )
            if self.bias is not None:
                out = out + self.bias.astype(np.float32)
            return out

        def thong_ke_nen(self) -> Dict[str, Any]:
            """Thống kê dung lượng nén 2-bit thực tế so với trọng số FP32 gốc."""
            packed, scale = self.dong_goi_2bit_kem_ty_le()
            fp32_bytes = int(self.in_features * self.out_features * 4)
            return {
                "in_features": int(self.in_features),
                "out_features": int(self.out_features),
                "fp32_bytes": fp32_bytes,
                "packed_2bit_bytes": len(packed),
                "ti_le_nen": round(fp32_bytes / max(1, len(packed)), 2),
                "ty_le_absmean": round(scale, 6),
                "dung_rust": co_rust_bitnet(),
                "da_nen_tai_cho": self._da_nen_tai_cho,
            }

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
