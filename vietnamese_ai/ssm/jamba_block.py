"""Kiến trúc lai Jamba: Mamba + Transformer + Mixture of Experts (MoE) hỗ trợ PyTorch và thuần NumPy."""

import math
from typing import Any

import numpy as np

from vietnamese_ai.ssm.mamba_block import MambaBlock, _silu_np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False


if _CO_TORCH:

    class SparseMoE(nn.Module):
        """Mạng Mixture of Experts (MoE) định tuyến Top-K."""

        def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 1):
            super().__init__()
            self.d_model = d_model
            self.num_experts = num_experts
            self.top_k = top_k
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
                    )
                    for _ in range(num_experts)
                ]
            )
            self.router = nn.Linear(d_model, num_experts, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            batch, seqlen, d_model = x.shape
            x_flat = x.view(-1, d_model)
            routing_weights = F.softmax(self.router(x_flat), dim=-1)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            final_output = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                expert_mask = selected_experts == i
                if expert_mask.any():
                    token_indices = expert_mask.any(dim=-1)
                    expert_out = expert(x_flat[token_indices])
                    weight_idx = expert_mask[token_indices].nonzero(as_tuple=True)[1]
                    expert_weights = routing_weights[token_indices, weight_idx].unsqueeze(-1)
                    final_output[token_indices] += expert_out * expert_weights
            return final_output.view(batch, seqlen, d_model)

    class JambaBlock(nn.Module):
        """Khối Jamba lai luân phiên giữa Mamba + MLP và Self-Attention + MoE."""

        def __init__(
            self, d_model: int, layer_idx: int, attn_layer_freq: int = 8, num_experts: int = 4
        ):
            super().__init__()
            self.d_model = d_model
            self.layer_idx = layer_idx
            self.attn_layer_freq = attn_layer_freq
            self.is_attn_layer = layer_idx % attn_layer_freq == 0
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            if self.is_attn_layer:
                self.mixer = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
                self.mlp = SparseMoE(d_model=d_model, num_experts=num_experts)
            else:
                self.mixer = MambaBlock(d_model=d_model, d_state=16)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
                )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x_norm = self.norm1(x)
            if self.is_attn_layer:
                mixed_x, _ = self.mixer(x_norm, x_norm, x_norm)
            else:
                mixed_x = self.mixer(x_norm)
            x = x + mixed_x
            return x + self.mlp(self.norm2(x))

else:

    class SparseMoE:  # type: ignore[no-redef]
        """Mạng Mixture of Experts (MoE) Top-K thuần NumPy."""

        def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 1):
            self.d_model = d_model
            self.num_experts = num_experts
            self.top_k = max(1, min(top_k, num_experts))
            rng = np.random.RandomState(42)
            s = 1.0 / math.sqrt(max(1, d_model))
            self.W_router = rng.randn(d_model, num_experts) * s
            self.experts = [
                (
                    rng.randn(d_model, d_model * 2) * s,
                    rng.randn(d_model * 2, d_model) * (1.0 / math.sqrt(max(1, d_model * 2))),
                )
                for _ in range(num_experts)
            ]

        def forward(self, x: Any) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            batch, seqlen, d_model = x_arr.shape
            x_flat = x_arr.reshape(-1, d_model)
            logits = x_flat @ self.W_router
            logits -= np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

            topk_idx = np.argsort(probs, axis=-1)[:, -self.top_k :]
            topk_w = np.take_along_axis(probs, topk_idx, axis=-1)
            topk_w = topk_w / np.sum(topk_w, axis=-1, keepdims=True)

            out_flat = np.zeros_like(x_flat)
            for k_pos in range(self.top_k):
                exp_ids = topk_idx[:, k_pos]
                weights = topk_w[:, k_pos : k_pos + 1]
                for i, (w1, w2) in enumerate(self.experts):
                    mask = exp_ids == i
                    if np.any(mask):
                        h = _silu_np(x_flat[mask] @ w1) @ w2
                        out_flat[mask] += weights[mask] * h
            return out_flat.reshape(batch, seqlen, d_model)

        def __call__(self, x: Any) -> np.ndarray:
            return self.forward(x)

    class JambaBlock:  # type: ignore[no-redef]
        """Khối Jamba lai (Mamba + Self-Attention + SparseMoE) thuần NumPy."""

        def __init__(
            self, d_model: int, layer_idx: int, attn_layer_freq: int = 8, num_experts: int = 4
        ):
            self.d_model = d_model
            self.layer_idx = layer_idx
            self.attn_layer_freq = attn_layer_freq
            self.is_attn_layer = layer_idx % attn_layer_freq == 0
            rng = np.random.RandomState(42 + layer_idx)
            s = 1.0 / math.sqrt(max(1, d_model))
            if self.is_attn_layer:
                self.W_qkv = rng.randn(d_model, d_model * 3) * s
                self.W_o = rng.randn(d_model, d_model) * s
                self.mlp: Any = SparseMoE(d_model=d_model, num_experts=num_experts)
            else:
                self.mixer = MambaBlock(d_model=d_model, d_state=16)
                self.W1 = rng.randn(d_model, d_model * 2) * s
                self.W2 = rng.randn(d_model * 2, d_model) * (1.0 / math.sqrt(max(1, d_model * 2)))

        @staticmethod
        def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps)

        def forward(self, x: Any) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            x_norm = self._layer_norm(x_arr)
            if self.is_attn_layer:
                B, T, C = x_norm.shape
                qkv = x_norm @ self.W_qkv
                q, k, v = np.split(qkv, 3, axis=-1)
                scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(max(1, C))
                scores -= np.max(scores, axis=-1, keepdims=True)
                attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
                mixed = (attn @ v) @ self.W_o
            else:
                mixed = self.mixer(x_norm)
            x_arr = x_arr + mixed

            x_norm2 = self._layer_norm(x_arr)
            if self.is_attn_layer:
                mlp_out = self.mlp(x_norm2)
            else:
                mlp_out = _silu_np(x_norm2 @ self.W1) @ self.W2
            return x_arr + mlp_out

        def __call__(self, x: Any) -> np.ndarray:
            return self.forward(x)
