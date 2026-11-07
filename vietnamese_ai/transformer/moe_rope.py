"""Kiến trúc Transformer Thế hệ mới: Rotary Position Embedding (RoPE), Grouped-Query Attention (GQA) & Sparse Mixture-of-Experts (MoE)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RotaryPositionEmbedding:
    """
    Mã hóa vị trí xoay (Rotary Position Embedding - RoPE) chuẩn Qwen3.8 / Gemma 4 / Llama 4.

    Áp dụng phép quay phức 2D theo tần số góc $\\theta_i = \\text{base}^{-2i / d}$ lên từng cặp chiều
    của các vector Query và Key mà không cần cộng ma trận vị trí cố định.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        do_dai_toi_da: int = 32768,
        theta_base: float = 10000.0,
        d_k: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        actual_dim = dim if dim is not None else (d_k if d_k is not None else 64)
        if max_seq_len is not None:
            do_dai_toi_da = max_seq_len
        if actual_dim % 2 != 0:
            raise ValueError(f"Chiều head_dim ({actual_dim}) phải là số chẵn để áp dụng RoPE.")
        self.dim = actual_dim
        self.d_k = actual_dim
        self.do_dai_toi_da = do_dai_toi_da
        self.theta_base = theta_base

        inv_freq = 1.0 / (theta_base ** (np.arange(0, actual_dim, 2, dtype=np.float32) / actual_dim))
        self.inv_freq = inv_freq

    def _bang_cos_sin(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)
        return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def xoay(self, x: np.ndarray, vi_tri_bat_dau: int = 0) -> np.ndarray:
        """
        Áp dụng phép quay RoPE lên tensor `x` có kích thước `(batch, so_dau, seq_len, head_dim)`
        hoặc `(batch, seq_len, head_dim)`.
        """
        x = np.asarray(x, dtype=np.float32)
        seq_len = x.shape[-2]
        cos, sin = self._bang_cos_sin(vi_tri_bat_dau + seq_len)
        cos = cos[vi_tri_bat_dau : vi_tri_bat_dau + seq_len]
        sin = sin[vi_tri_bat_dau : vi_tri_bat_dau + seq_len]

        if x.ndim == 4:
            cos = cos[np.newaxis, np.newaxis, :, :]
            sin = sin[np.newaxis, np.newaxis, :, :]
        else:
            cos = cos[np.newaxis, :, :]
            sin = sin[np.newaxis, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def ap_dung(
        self, q: np.ndarray, k: np.ndarray, vi_tri_bat_dau: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Xoay đồng thời cả hai tensor Query và Key bằng RoPE."""
        return self.xoay(q, vi_tri_bat_dau=vi_tri_bat_dau), self.xoay(
            k, vi_tri_bat_dau=vi_tri_bat_dau
        )


class GroupedQueryAttention:
    """
    Cơ chế Chú ý Truy vấn Nhóm (Grouped-Query Attention - GQA) kết hợp RoPE.

    Cho phép `so_dau_q` đầu Query dùng chung `so_dau_kv` đầu Key/Value (`so_dau_q % so_dau_kv == 0`),
    giảm kích thước KV Cache gấp `so_dau_q // so_dau_kv` lần so với Multi-Head Attention truyền thống.
    """

    def __init__(
        self,
        d_model: int = 64,
        so_dau_q: int = 8,
        so_dau_kv: int = 2,
        theta_base: float = 10000.0,
    ):
        if d_model % so_dau_q != 0:
            raise ValueError("d_model phải chia hết cho so_dau_q")
        if so_dau_q % so_dau_kv != 0:
            raise ValueError("so_dau_q phải chia hết cho so_dau_kv")

        self.d_model = d_model
        self.so_dau_q = so_dau_q
        self.so_dau_kv = so_dau_kv
        self.so_nhom = so_dau_q // so_dau_kv
        self.head_dim = d_model // so_dau_q
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim phải là số chẵn để dùng RoPE")

        self.rope = RotaryPositionEmbedding(self.head_dim, theta_base=theta_base)
        scale = float(np.sqrt(2.0 / d_model))
        rng = np.random.default_rng(42)
        self.W_q = rng.normal(0, scale, (d_model, so_dau_q * self.head_dim)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, so_dau_kv * self.head_dim)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, so_dau_kv * self.head_dim)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (so_dau_q * self.head_dim, d_model)).astype(np.float32)

    def tien(
        self,
        x: np.ndarray,
        causal_mask: bool = True,
        su_dung_cache: bool = False,
    ) -> np.ndarray:
        """Lan truyền tiến GQA + RoPE trên `x` có kích thước `(batch, seq_len, d_model)`."""
        x = np.asarray(x, dtype=np.float32)
        b, s, _ = x.shape

        q = (x @ self.W_q).reshape(b, s, self.so_dau_q, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.W_k).reshape(b, s, self.so_dau_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.W_v).reshape(b, s, self.so_dau_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope.xoay(q)
        k = self.rope.xoay(k)
        if su_dung_cache:
            self._kv_cache = {"k": k, "v": v}

        # Lặp các đầu KV cho từng nhóm Query
        if self.so_nhom > 1:
            k = np.repeat(k, self.so_nhom, axis=1)
            v = np.repeat(v, self.so_nhom, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(self.head_dim))
        if causal_mask and s > 1:
            mask = np.triu(np.ones((s, s), dtype=bool), k=1)
            scores = np.where(mask[np.newaxis, np.newaxis, :, :], -1e9, scores)

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, self.d_model)
        return (context @ self.W_o).astype(np.float32)

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê cấu hình GQA và tỷ lệ nén bộ nhớ KV Cache."""
        return {
            "d_model": self.d_model,
            "so_dau_q": self.so_dau_q,
            "so_dau_kv": self.so_dau_kv,
            "he_so_nen_kv_cache": self.so_nhom,
        }


class SparseMoELayer:
    r"""
    Tầng Hỗn hợp Chuyên gia Thưa (Sparse Mixture-of-Experts - MoE) chuẩn Qwen3.8-MoE / DeepSeek-V3 / Llama 4.

    Tính năng:
    - Bộ định tuyến cổng Top-$K$ (`W_gate`) chọn $K$ chuyên gia phù hợp nhất cho từng token.
    - Hỗ trợ Chuyên gia dùng chung (`co_chuyen_gia_chung=True` - Shared Expert) luôn hoạt động.
    - Tự động tính hàm mất mát cân bằng tải (**Load Balancing Auxiliary Loss**):
      $$\mathcal{L}_{\text{aux}} = N \sum_{i=1}^{N} f_i \cdot P_i$$
      giúp phân bổ đều lưu lượng token giữa các chuyên gia.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_ff: int = 128,
        so_chuyen_gia: int = 4,
        top_k: int = 2,
        co_chuyen_gia_chung: bool = True,
        he_so_can_bang: float = 0.01,
    ):
        if top_k > so_chuyen_gia or top_k < 1:
            raise ValueError("top_k phải nằm trong đoạn [1, so_chuyen_gia]")

        self.d_model = d_model
        self.d_ff = d_ff
        self.so_chuyen_gia = so_chuyen_gia
        self.top_k = top_k
        self.co_chuyen_gia_chung = co_chuyen_gia_chung
        self.he_so_can_bang = he_so_can_bang
        self.aux_loss: float = 0.0
        self.thong_ke_chuyen_gia: np.ndarray = np.zeros(so_chuyen_gia, dtype=np.float32)

        rng = np.random.default_rng(42)
        scale = float(np.sqrt(2.0 / d_model))
        self.W_gate = rng.normal(0, scale, (d_model, so_chuyen_gia)).astype(np.float32)

        self.chuyen_gia: List[Dict[str, np.ndarray]] = []
        for _ in range(so_chuyen_gia):
            self.chuyen_gia.append(
                {
                    "W1": rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
                    "W2": rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
                }
            )

        if co_chuyen_gia_chung:
            self.chuyen_gia_chung = {
                "W1": rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
                "W2": rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
            }
        else:
            self.chuyen_gia_chung = None

    @staticmethod
    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def _chay_mot_chuyen_gia(self, x_tok: np.ndarray, exp: Dict[str, np.ndarray]) -> np.ndarray:
        h = self._silu(x_tok @ exp["W1"])
        return h @ exp["W2"]

    def tien(self, x: np.ndarray) -> np.ndarray:
        """
        Định tuyến Top-K token qua các chuyên gia và cập nhật `self.aux_loss`.
        """
        x = np.asarray(x, dtype=np.float32)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        n_tokens = x_flat.shape[0]

        # 1. Tính xác suất định tuyến Softmax
        logits = x_flat @ self.W_gate
        logits_shift = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits_shift) / np.sum(np.exp(logits_shift), axis=-1, keepdims=True)

        # 2. Chọn Top-K chuyên gia cho mỗi token
        top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k :]
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)
        top_k_weights = top_k_probs / (np.sum(top_k_probs, axis=-1, keepdims=True) + 1e-9)

        # 3. Tính Load Balancing Auxiliary Loss: N * sum(f_i * P_i)
        P_i = np.mean(probs, axis=0)
        counts = np.zeros(self.so_chuyen_gia, dtype=np.float32)
        for idx in top_k_indices.flatten():
            counts[int(idx)] += 1.0
        f_i = counts / max(1.0, float(n_tokens * self.top_k))
        self.thong_ke_chuyen_gia = f_i
        self.aux_loss = float(self.so_chuyen_gia * np.sum(f_i * P_i))

        # 4. Tính đầu ra kết hợp từ Top-K chuyên gia được kích hoạt
        out_flat = np.zeros_like(x_flat)
        for e_idx in range(self.so_chuyen_gia):
            mask_matrix = top_k_indices == e_idx
            token_rows, k_cols = np.where(mask_matrix)
            if len(token_rows) == 0:
                continue
            x_selected = x_flat[token_rows]
            y_selected = self._chay_mot_chuyen_gia(x_selected, self.chuyen_gia[e_idx])
            w_selected = top_k_weights[token_rows, k_cols][:, np.newaxis]
            np.add.at(out_flat, token_rows, y_selected * w_selected)

        if self.chuyen_gia_chung is not None:
            out_flat = out_flat + 0.5 * self._chay_mot_chuyen_gia(x_flat, self.chuyen_gia_chung)

        return out_flat.reshape(orig_shape).astype(np.float32)

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê phân phối định tuyến chuyên gia và Load Balancing Loss."""
        return {
            "so_chuyen_gia": self.so_chuyen_gia,
            "top_k": self.top_k,
            "aux_loss": self.aux_loss,
            "tan_suat_chon_chuyen_gia": self.thong_ke_chuyen_gia.tolist(),
        }
