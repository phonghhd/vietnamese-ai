"""Lõi Selective State Space Model (Mamba) O(N) hỗ trợ song song PyTorch và thuần NumPy."""

import math
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


if _CO_TORCH:

    class MambaBlock(nn.Module):
        """
        Module Selective State Space Model (Mamba) với độ phức tạp tuyến tính O(N) theo độ dài chuỗi.
        """

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(self.expand * self.d_model)

            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
            self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
            _, seqlen, _ = hidden_states.shape
            xz = self.in_proj(hidden_states)
            x, z = xz.chunk(2, dim=-1)
            x = self.conv1d(x.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
            x = F.silu(x)

            x_dbl = self.x_proj(x)
            delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta))
            A = -torch.exp(self.A_log.float())

            y = self._selective_scan_pytorch(x, delta, A, B, C)
            y = (y + x * self.D) * F.silu(z)
            return self.out_proj(y)

        @staticmethod
        def _selective_scan_pytorch(x, delta, A, B, C):
            batch, seqlen, d_inner = x.shape
            d_state = A.shape[1]
            h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
            y = torch.zeros(batch, seqlen, d_inner, device=x.device, dtype=x.dtype)
            for t in range(seqlen):
                delta_t = delta[:, t, :].unsqueeze(-1)
                x_t = x[:, t, :].unsqueeze(-1)
                B_t = B[:, t, :].unsqueeze(1)
                C_t = C[:, t, :].unsqueeze(1)
                A_bar = torch.exp(delta_t * A)
                h = A_bar * h + delta_t * B_t * x_t
                y[:, t, :] = torch.sum(h * C_t, dim=-1)
            return y

else:

    class MambaBlock:  # type: ignore[no-redef]
        """
        Selective State Space Model (Mamba) O(N) thuần NumPy — chạy thật 100% không cần cài PyTorch.
        """

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(expand * d_model)

            rng = np.random.RandomState(42)
            s_in = 1.0 / math.sqrt(max(1, d_model))
            s_inner = 1.0 / math.sqrt(max(1, self.d_inner))

            self.W_in = rng.randn(d_model, self.d_inner * 2) * s_in
            self.conv_weight = rng.randn(self.d_inner, d_conv) * (1.0 / math.sqrt(max(1, d_conv)))
            self.conv_bias = np.zeros(self.d_inner, dtype=np.float64)
            self.W_x = rng.randn(self.d_inner, self.d_state * 2 + 1) * s_inner
            self.W_dt = rng.randn(1, self.d_inner) * 0.1
            self.b_dt = np.zeros(self.d_inner, dtype=np.float64)
            A_init = np.tile(np.arange(1, self.d_state + 1, dtype=np.float64), (self.d_inner, 1))
            self.A_log = np.log(A_init)
            self.D = np.ones(self.d_inner, dtype=np.float64)
            self.W_out = rng.randn(self.d_inner, d_model) * s_inner

        def forward(self, hidden_states: Any) -> np.ndarray:
            hs = np.asarray(hidden_states, dtype=np.float64)
            batch, seqlen, _ = hs.shape
            xz = hs @ self.W_in
            x = xz[:, :, : self.d_inner]
            z = xz[:, :, self.d_inner :]

            # Causal Depthwise 1D Convolution
            x_padded = np.pad(x, ((0, 0), (self.d_conv - 1, 0), (0, 0)), mode="constant")
            x_conv = np.zeros_like(x)
            for k in range(self.d_conv):
                w_k = self.conv_weight[:, self.d_conv - 1 - k]  # (d_inner,)
                x_conv += x_padded[:, self.d_conv - 1 - k : self.d_conv - 1 - k + seqlen, :] * w_k
            x_conv = _silu_np(x_conv + self.conv_bias)

            x_dbl = x_conv @ self.W_x
            dt_raw = x_dbl[:, :, :1]
            B = x_dbl[:, :, 1 : 1 + self.d_state]
            C = x_dbl[:, :, 1 + self.d_state :]
            delta = _softplus_np(dt_raw @ self.W_dt + self.b_dt)
            A = -np.exp(self.A_log)

            h = np.zeros((batch, self.d_inner, self.d_state), dtype=np.float64)
            y = np.zeros((batch, seqlen, self.d_inner), dtype=np.float64)
            for t in range(seqlen):
                delta_t = delta[:, t, :, None]
                x_t = x_conv[:, t, :, None]
                B_t = B[:, t, None, :]
                C_t = C[:, t, None, :]
                A_bar = np.exp(np.clip(delta_t * A[None, :, :], -30.0, 5.0))
                h = A_bar * h + delta_t * B_t * x_t
                y[:, t, :] = np.sum(h * C_t, axis=-1)

            y = (y + x_conv * self.D) * _silu_np(z)
            return y @ self.W_out

        def __call__(self, hidden_states: Any) -> np.ndarray:
            return self.forward(hidden_states)
