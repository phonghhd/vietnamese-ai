"""MultiHeadAttention - Cơ chế chú ý đa đầu."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


class MultiHeadAttention:
    """
    Multi-Head Attention tự cài đặt (NumPy).

    Cơ chế chú ý đa đầu cho Transformer:
    - Split input thành nhiều heads
    - Tính attention song song trên mỗi head
    - Hỗ trợ mặt nạ nhân quả (causal), KV-Cache suy luận và kiểm soát dropout theo chế độ training/eval
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
        self.training = False

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
        training: Optional[bool] = None,
        causal: bool = False,
        past_len: int = 0,
    ) -> np.ndarray:
        """Scaled Dot-Product Attention."""
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)

        if causal:
            seq_q = Q.shape[2]
            seq_k = K.shape[2]
            q_pos = np.arange(past_len, past_len + seq_q)[:, None]
            k_pos = np.arange(seq_k)[None, :]
            causal_mask = (k_pos <= q_pos)[None, None, :, :]
            scores = np.where(causal_mask, scores, -1e9)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        ap_dung_drop = self.training if training is None else bool(training)
        if ap_dung_drop and self.dropout > 0:
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
        training: Optional[bool] = None,
        causal: bool = False,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Forward pass Multi-Head Attention.

        Args:
            Q: Query (batch, seq_q, d_model)
            K: Key (batch, seq_k, d_model)
            V: Value (batch, seq_k, d_model)
            mask: Attention mask (tùy chọn)
            training: Bật dropout khi huấn luyện; mặc định tắt ở chế độ suy luận.
            causal: Bật mặt nạ nhân quả tự hồi quy.
            kv_cache: Bộ đệm từ điển lưu `k`, `v` đã chiếu qua các bước giải mã.

        Returns:
            Output (batch, seq_q, d_model)
        """
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v

        Q_heads = self._chia_heads(Q_proj)
        K_heads = self._chia_heads(K_proj)
        V_heads = self._chia_heads(V_proj)

        past_len = 0
        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                past_len = int(kv_cache["k"].shape[2])
                K_heads = np.concatenate([kv_cache["k"], K_heads], axis=2)
                V_heads = np.concatenate([kv_cache["v"], V_heads], axis=2)
            kv_cache["k"] = K_heads
            kv_cache["v"] = V_heads

        attn_output = self._tinh_attention(
            Q_heads,
            K_heads,
            V_heads,
            mask=mask,
            training=training,
            causal=causal,
            past_len=past_len,
        )

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
    Thay vì lưu K, V kích thước (seq, d_model), ta chỉ lưu `C_kv` kích thước (seq, latent_dim).

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
        self.W_d_kv = np.random.uniform(-limit, limit, (d_model, latent_dim))
        limit_latent = math.sqrt(6.0 / (latent_dim + d_model))
        self.W_u_k = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))
        self.W_u_v = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))

        self.W_d_q = np.random.uniform(-limit, limit, (d_model, latent_dim))
        self.W_u_q = np.random.uniform(-limit_latent, limit_latent, (latent_dim, d_model))

        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

    def _chia_heads(self, X: np.ndarray) -> np.ndarray:
        batch, seq, _ = X.shape
        return X.reshape(batch, seq, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

    def tien(
        self,
        X: np.ndarray,
        causal: bool = False,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Forward pass với nén Latent và bộ đệm KV-Cache nén `C_kv` thực tế."""
        C_q = X @ self.W_d_q
        C_kv_new = X @ self.W_d_kv

        past_len = 0
        if kv_cache is not None:
            if "c_kv" in kv_cache:
                past_len = int(kv_cache["c_kv"].shape[1])
                C_kv_all = np.concatenate([kv_cache["c_kv"], C_kv_new], axis=1)
            else:
                C_kv_all = C_kv_new
            kv_cache["c_kv"] = C_kv_all
        else:
            C_kv_all = C_kv_new

        Q = C_q @ self.W_u_q
        K = C_kv_all @ self.W_u_k
        V = C_kv_all @ self.W_u_v

        Q_heads = self._chia_heads(Q)
        K_heads = self._chia_heads(K)
        V_heads = self._chia_heads(V)

        scores = Q_heads @ K_heads.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        if causal:
            seq_q = Q_heads.shape[2]
            seq_k = K_heads.shape[2]
            q_pos = np.arange(past_len, past_len + seq_q)[:, None]
            k_pos = np.arange(seq_k)[None, :]
            causal_mask = (k_pos <= q_pos)[None, None, :, :]
            scores = np.where(causal_mask, scores, -1e9)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        attn_output = attn_weights @ V_heads

        batch, _, seq, _ = attn_output.shape
        concat = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        return concat @ self.W_o


class MultiHeadFlashAttention:
    """
    Multi-Head FlashAttention tự cài đặt (NumPy).

    Thuật toán Tiling (chia khối) để giảm độ phức tạp RAM từ O(N^2) xuống O(N),
    hỗ trợ mặt nạ nhân quả theo khối (bỏ qua hoàn toàn các khối tương lai `j >= i + B_r`).
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

    def tien(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
    ) -> np.ndarray:
        """Forward pass sử dụng Thuật toán FlashAttention (Block-wise, tùy chọn causal masking)."""
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v

        Q_h = self._chia_heads(Q_proj)
        K_h = self._chia_heads(K_proj)
        V_h = self._chia_heads(V_proj)

        batch, so_dau, seq_q, d_k = Q_h.shape
        seq_k = K_h.shape[2]

        out_val = np.zeros_like(Q_h)
        m = np.full((batch, so_dau, seq_q, 1), -np.inf)
        l_val = np.zeros((batch, so_dau, seq_q, 1))

        B_c = self.block_size
        B_r = self.block_size

        for j in range(0, seq_k, B_c):
            K_j = K_h[:, :, j : j + B_c, :]
            V_j = V_h[:, :, j : j + B_c, :]
            k_len = K_j.shape[2]

            for i in range(0, seq_q, B_r):
                Q_i = Q_h[:, :, i : i + B_r, :]
                q_len = Q_i.shape[2]
                if causal and j >= i + q_len:
                    continue

                S_ij = (Q_i @ K_j.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
                if causal and j + k_len > i:
                    q_idx = np.arange(i, i + q_len)[:, None]
                    k_idx = np.arange(j, j + k_len)[None, :]
                    c_mask = (k_idx <= q_idx)[None, None, :, :]
                    S_ij = np.where(c_mask, S_ij, -np.inf)

                m_i_prev = m[:, :, i : i + B_r, :]
                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                m_i_new = np.maximum(m_i_prev, m_ij)
                m_safe = np.where(np.isfinite(m_i_new), m_i_new, 0.0)

                P_ij = np.where(np.isfinite(S_ij), np.exp(S_ij - m_safe), 0.0)

                l_i_prev = l_val[:, :, i : i + B_r, :]
                factor = np.where(np.isfinite(m_i_prev), np.exp(m_i_prev - m_safe), 0.0)

                l_i_new = factor * l_i_prev + np.sum(P_ij, axis=-1, keepdims=True)

                out_i_prev = out_val[:, :, i : i + B_r, :]
                out_i_new = (factor * out_i_prev * l_i_prev + P_ij @ V_j) / (l_i_new + 1e-10)

                m[:, :, i : i + B_r, :] = m_i_new
                l_val[:, :, i : i + B_r, :] = l_i_new
                out_val[:, :, i : i + B_r, :] = out_i_new

        concat = out_val.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)
        return concat @ self.W_o
