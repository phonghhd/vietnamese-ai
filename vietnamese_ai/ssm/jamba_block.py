"""Kiến trúc lai Jamba: Mamba + Transformer + Mixture of Experts (MoE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vietnamese_ai.ssm.mamba_block import MambaBlock


class SparseMoE(nn.Module):
    """
    Mạng Mixture of Experts (MoE) Thu gọn.
    Chỉ sử dụng 4 Chuyên gia và định tuyến Top-1 để tối ưu cho CPU/Edge.
    """

    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Mạng nội bộ cho từng chuyên gia (MLP đơn giản)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
                )
                for _ in range(num_experts)
            ]
        )

        # Router (Cổng định tuyến)
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, D_Model)
        """
        batch, seqlen, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (Batch * SeqLen, D_Model)

        # Tính điểm định tuyến
        router_logits = self.router(x_flat)  # (Batch * SeqLen, Num_Experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Top-K Routing
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Tính toán kết quả cho từng token
        final_output = torch.zeros_like(x_flat)

        # Tính MoE tuần tự (Do CPU không song song tốt như GPU với Triton)
        for i, expert in enumerate(self.experts):
            # Tìm các token được gán cho expert i
            expert_mask = selected_experts == i

            if expert_mask.any():
                token_indices = expert_mask.any(dim=-1)
                expert_out = expert(x_flat[token_indices])

                weight_idx = expert_mask[token_indices].nonzero(as_tuple=True)[1]
                expert_weights = routing_weights[token_indices, weight_idx].unsqueeze(-1)

                final_output[token_indices] += expert_out * expert_weights

        return final_output.view(batch, seqlen, d_model)


class JambaBlock(nn.Module):
    """
    Khối Jamba lai.
    Luân phiên giữa:
    - Mamba + MLP (Mặc định)
    - Self-Attention + MoE (Xảy ra ở layer_idx chia hết cho 8)
    """

    def __init__(
        self, d_model: int, layer_idx: int, attn_layer_freq: int = 8, num_experts: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.attn_layer_freq = attn_layer_freq

        self.is_attn_layer = layer_idx % attn_layer_freq == 0

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if self.is_attn_layer:
            # Nhánh Attention + MoE
            self.mixer = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
            self.mlp = SparseMoE(d_model=d_model, num_experts=num_experts)
        else:
            # Nhánh Mamba + MLP thường
            self.mixer = MambaBlock(d_model=d_model, d_state=16)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, D_Model)
        """
        # Bước 1: Mixer (Mamba hoặc Attention)
        residual = x
        x_norm = self.norm1(x)

        if self.is_attn_layer:
            # Attention cần trả về (output, attn_weights)
            mixed_x, _ = self.mixer(x_norm, x_norm, x_norm)
        else:
            mixed_x = self.mixer(x_norm)

        x = residual + mixed_x

        # Bước 2: MLP (Thường hoặc MoE)
        residual = x
        x_norm = self.norm2(x)
        mlp_x = self.mlp(x_norm)
        x = residual + mlp_x

        return x
