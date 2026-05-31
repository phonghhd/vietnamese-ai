"""MultiHeadAttention - Cơ chế chú ý đa đầu."""

import math
from typing import Optional

import numpy as np


class MultiHeadAttention:
    """
    Multi-Head Attention tự cài đặt (NumPy).

    Cơ chế chú ý đa đầu cho Transformer:
    - Split input thành nhiều heads
    - Tính attention song song trên mỗi head
    - Concatenate và project output

    Sử dụng:
        >>> mha = MultiHeadAttention(d_model=64, so_dau=4)
        >>> output = mha.tien(Q, K, V)
    """

    def __init__(self, d_model: int, so_dau: int, dropout: float = 0.0):
        if d_model % so_dau != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_dau ({so_dau})")

        self.d_model = d_model
        self.so_dau = so_dau
        self.d_k = d_model // so_dau
        self.dropout = dropout

        limit = math.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_k = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_v = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

        self._cache: dict = {}

    def _chia_heads(self, X: np.ndarray) -> np.ndarray:
        """Chia tensor thành nhiều heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        batch, seq, _ = X.shape
        return X.reshape(batch, seq, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

    def _tinh_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Scaled Dot-Product Attention."""
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        if self.dropout > 0:
            drop_mask = (np.random.rand(*attn_weights.shape) > self.dropout).astype(float)
            attn_weights = attn_weights * drop_mask / (1 - self.dropout)

        self._cache["attn_weights"] = attn_weights
        return attn_weights @ V

    def tien(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass Multi-Head Attention.

        Args:
            Q: Query (batch, seq_q, d_model)
            K: Key (batch, seq_k, d_model)
            V: Value (batch, seq_k, d_model)
            mask: Attention mask (tùy chọn)

        Returns:
            Output (batch, seq_q, d_model)
        """
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v

        Q_heads = self._chia_heads(Q_proj)
        K_heads = self._chia_heads(K_proj)
        V_heads = self._chia_heads(V_proj)

        attn_output = self._tinh_attention(Q_heads, K_heads, V_heads, mask)

        batch, _, seq, _ = attn_output.shape
        concat = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        return concat @ self.W_o

    def lay_attention_weights(self) -> Optional[np.ndarray]:
        """Lấy attention weights từ lần forward gần nhất."""
        return self._cache.get("attn_weights")

    def thong_ke(self) -> dict:
        """Thống kê module."""
        return {
            "d_model": self.d_model,
            "so_dau": self.so_dau,
            "d_k": self.d_k,
            "so_tham_so": self.d_model * self.d_model * 4,
        }


class MultiHeadLatentAttention:
    """
    Multi-Head Latent Attention (MLA) - Lấy cảm hứng từ DeepSeek-V2.

    Cơ chế: Nén KV Cache thành một vector ẩn (Latent Vector) siêu nhỏ để tiết kiệm RAM.
    Thay vì lưu K, V kích thước (seq, d_model), ta chỉ lưu C_kv kích thước (seq, latent_dim).

    Sử dụng:
        >>> mla = MultiHeadLatentAttention(d_model=64, so_dau=4, latent_dim=16)
        >>> output = mla.tien(X)
    """

    def __init__(self, d_model: int, so_dau: int, latent_dim: int):
        self.d_model = d_model
        self.so_dau = so_dau
        self.latent_dim = latent_dim
        self.d_k = d_model // so_dau

        limit = math.sqrt(6.0 / (d_model + d_model))
        # Nén KV
        self.W_d_kv = np.random.uniform(-limit, limit, (d_model, latent_dim))
        # Giải nén ra K, V
        limit_latent = math.sqrt(6.0 / (latent_dim + d_model))
        self.W_u_k = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))
        self.W_u_v = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))

        # Tương tự cho Query
        self.W_d_q = np.random.uniform(-limit, limit, (d_model, latent_dim))
        self.W_u_q = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))

        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

    def _chia_heads(self, X: np.ndarray) -> np.ndarray:
        batch, seq, _ = X.shape
        return X.reshape(batch, seq, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

    def tien(self, X: np.ndarray) -> np.ndarray:
        """Forward pass với nén Latent."""
        # 1. Nén input vào Latent Space
        C_q = X @ self.W_d_q
        C_kv = X @ self.W_d_kv  # Đây là cái duy nhất cần lưu vào KV Cache! RAM giảm 75%

        # 2. Giải nén on-the-fly
        Q = C_q @ self.W_u_q
        K = C_kv @ self.W_u_k
        V = C_kv @ self.W_u_v

        Q_heads = self._chia_heads(Q)
        K_heads = self._chia_heads(K)
        V_heads = self._chia_heads(V)

        # Tương đương dot-product thông thường
        scores = Q_heads @ K_heads.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        attn_output = attn_weights @ V_heads

        batch, _, seq, _ = attn_output.shape
        concat = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        return concat @ self.W_o


class MultiHeadFlashAttention:
    """
    Multi-Head FlashAttention tự cài đặt (NumPy).

    Thuật toán Tiling (chia khối) để giảm độ phức tạp RAM từ O(N^2) xuống O(N).
    Tránh cấp phát ma trận (seq, seq) cho Attention Scores.
    """

    def __init__(self, d_model: int, so_dau: int, block_size: int = 128):
        self.d_model = d_model
        self.so_dau = so_dau
        self.d_k = d_model // so_dau
        self.block_size = block_size

        limit = math.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_k = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_v = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

    def _chia_heads(self, X: np.ndarray) -> np.ndarray:
        batch, seq, _ = X.shape
        return X.reshape(batch, seq, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

    def tien(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Forward pass sử dụng Thuật toán FlashAttention (Block-wise)."""
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v

        # (batch, so_dau, seq, d_k)
        Q_h = self._chia_heads(Q_proj)
        K_h = self._chia_heads(K_proj)
        V_h = self._chia_heads(V_proj)

        batch, so_dau, seq_len, d_k = Q_h.shape

        # Biến out_val (Output), m (Max), l_val (Sum) sẽ được lưu trong SRAM (giả lập)
        out_val = np.zeros_like(Q_h)
        m = np.full((batch, so_dau, seq_len, 1), -np.inf)
        l_val = np.zeros((batch, so_dau, seq_len, 1))

        B_c = self.block_size
        B_r = self.block_size

        # Phân rã Softmax theo chuẩn FlashAttention
        # Duyệt theo Key/Value (Outer loop)
        for j in range(0, seq_len, B_c):
            K_j = K_h[:, :, j : j + B_c, :]
            V_j = V_h[:, :, j : j + B_c, :]

            # Duyệt theo Query (Inner loop)
            for i in range(0, seq_len, B_r):
                Q_i = Q_h[:, :, i : i + B_r, :]

                # 1. Tính Scores S_ij (Kích thước B_r x B_c -> Nằm trọn trong L1/L2 Cache)
                S_ij = (Q_i @ K_j.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

                # 2. Tìm max cục bộ
                m_i_prev = m[:, :, i : i + B_r, :]
                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                m_i_new = np.maximum(m_i_prev, m_ij)

                # 3. Mũ hóa P_ij
                P_ij = np.exp(S_ij - m_i_new)

                # 4. Cập nhật l_val (Sum)
                l_i_prev = l_val[:, :, i : i + B_r, :]
                factor = np.exp(m_i_prev - m_i_new)

                # Sửa lỗi cảnh báo np.exp(-inf) => 0
                factor = np.where(m_i_prev == -np.inf, 0.0, factor)

                l_i_new = factor * l_i_prev + np.sum(P_ij, axis=-1, keepdims=True)

                # 5. Cập nhật Output out_val
                out_i_prev = out_val[:, :, i : i + B_r, :]
                out_i_new = (factor * out_i_prev * l_i_prev + P_ij @ V_j) / (l_i_new + 1e-10)

                # Ghi lại trạng thái
                m[:, :, i : i + B_r, :] = m_i_new
                l_val[:, :, i : i + B_r, :] = l_i_new
                out_val[:, :, i : i + B_r, :] = out_i_new

        # Nối output lại
        concat = out_val.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return concat @ self.W_o
