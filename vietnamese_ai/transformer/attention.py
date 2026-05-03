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
