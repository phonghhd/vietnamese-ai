"""DeepSeekMoE - Kiến trúc MoE tối ưu của DeepSeek-V3."""

import numpy as np

from vietnamese_ai.models.layers import LopDense


class DeepSeekMoE:
    """
    Mixture of Experts tối ưu từ DeepSeek.
    Khác với MoE thường, DeepSeekMoE có:
    1. Shared Experts: Luôn luôn kích hoạt (cho kiến thức chung).
    2. Routed Experts (Fine-grained): Được chẻ nhỏ ra làm rất nhiều chuyên gia siêu nhỏ.
    Chỉ chọn top_k chuyên gia cho mỗi token.
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int = 1,
        num_routed_experts: int = 16,
        top_k: int = 4,
    ):
        self.d_model = d_model
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        # Mạng chung (luôn chạy)
        self.shared_experts = [
            LopDense(d_model, d_model * 2, ham_kich_hoat="relu") for _ in range(num_shared_experts)
        ]
        self.shared_proj = [LopDense(d_model * 2, d_model) for _ in range(num_shared_experts)]

        # Mạng siêu nhỏ (Routed)
        expert_dim = d_model // 4  # Chuyên gia siêu nhỏ (nhẹ hơn 4 lần)
        self.routed_experts = [
            LopDense(d_model, expert_dim, ham_kich_hoat="relu") for _ in range(num_routed_experts)
        ]
        self.routed_proj = [LopDense(expert_dim, d_model) for _ in range(num_routed_experts)]

        # Cổng Router
        self.router = LopDense(d_model, num_routed_experts, ham_kich_hoat="softmax")

    def tien(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass cho MoE.
        X shape: (batch_size, d_model)
        """
        batch_size = X.shape[0]
        output = np.zeros_like(X)

        # 1. Tính toán Shared Experts (Kiến thức nền)
        for i in range(self.num_shared_experts):
            h_shared = self.shared_experts[i].tien(X)
            output += self.shared_proj[i].tien(h_shared)

        # 2. Tính điểm Routing
        routing_scores = self.router.tien(X)  # (batch_size, num_routed_experts)

        # Chọn top-K chuyên gia cho mỗi batch
        for b in range(batch_size):
            scores_b = routing_scores[b]
            # Lấy index của top K chuyên gia
            top_k_indices = np.argsort(scores_b)[-self.top_k :]

            # Chuẩn hóa lại điểm của Top K (Softmax cục bộ)
            top_k_scores = scores_b[top_k_indices]
            top_k_weights = top_k_scores / (np.sum(top_k_scores) + 1e-9)

            x_b = X[b : b + 1]  # (1, d_model)

            # Chỉ kích hoạt các chuyên gia trong Top K
            for idx, weight in zip(top_k_indices, top_k_weights):
                h_routed = self.routed_experts[idx].tien(x_b)
                out_routed = self.routed_proj[idx].tien(h_routed)
                output[b : b + 1] += weight * out_routed

        return output
