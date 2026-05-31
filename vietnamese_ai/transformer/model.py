"""TransformerModel - Mô hình Transformer tự cài đặt."""

import math
from typing import Dict, Optional

import numpy as np

from vietnamese_ai.transformer.attention import MultiHeadAttention
from vietnamese_ai.utils.logger import Logger


class TransformerBlock:
    """Một block Transformer (Self-Attention + FFN + LayerNorm)."""

    def __init__(self, d_model: int, so_dau: int, d_ff: int, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, so_dau, dropout)
        self.d_model = d_model
        self.d_ff = d_ff

        limit1 = math.sqrt(6.0 / (d_model + d_ff))
        self.W1 = np.random.uniform(-limit1, limit1, (d_model, d_ff))
        self.b1 = np.zeros((1, d_ff))
        limit2 = math.sqrt(6.0 / (d_ff + d_model))
        self.W2 = np.random.uniform(-limit2, limit2, (d_ff, d_model))
        self.b2 = np.zeros((1, d_model))

        self.gamma1 = np.ones((1, 1, d_model))
        self.beta1 = np.zeros((1, 1, d_model))
        self.gamma2 = np.ones((1, 1, d_model))
        self.beta2 = np.zeros((1, 1, d_model))

    def _layer_norm(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return gamma * (X - mean) / np.sqrt(var + 1e-6) + beta

    def _relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def tien(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn_out = self.attention.tien(X, X, X, mask)
        X = self._layer_norm(X + attn_out, self.gamma1, self.beta1)

        ff = self._relu(X @ self.W1 + self.b1) @ self.W2 + self.b2
        X = self._layer_norm(X + ff, self.gamma2, self.beta2)

        return X


class TransformerModel:
    """
    Mô hình Transformer tự cài đặt (NumPy).

    Hỗ trợ:
    - Encoder-only (cho classification, embedding)
    - Sinh positional encoding
    - Stack nhiều Transformer blocks
    - Text classification

    Sử dụng:
        >>> model = TransformerModel(d_model=64, so_dau=4, so_lop=8, so_block=2)
        >>> output = model.tien(input_ids)
        >>> pred = model.du_doan(input_ids)
    """

    def __init__(
        self,
        d_model: int = 64,
        so_dau: int = 4,
        d_ff: int = 128,
        so_block: int = 2,
        so_tu_vung: int = 1000,
        do_dai_toi_da: int = 512,
        so_lop: int = 2,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.so_dau = so_dau
        self.d_ff = d_ff
        self.so_block = so_block
        self.so_tu_vung = so_tu_vung
        self.do_dai_toi_da = do_dai_toi_da
        self.so_lop = so_lop
        self.logger = Logger("TransformerModel")

        self.embedding = np.random.randn(so_tu_vung, d_model) * 0.02
        self.positional = self._tao_positional_encoding(do_dai_toi_da, d_model)

        self.blocks = [TransformerBlock(d_model, so_dau, d_ff, dropout) for _ in range(so_block)]

        limit = math.sqrt(6.0 / (d_model + so_lop))
        self.W_classify = np.random.uniform(-limit, limit, (d_model, so_lop))
        self.b_classify = np.zeros((1, so_lop))

    def _tao_positional_encoding(self, do_dai: int, d_model: int) -> np.ndarray:
        """Tạo sinusoidal positional encoding."""
        pe = np.zeros((do_dai, d_model))
        position = np.arange(do_dai).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def tien(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Logits (batch, so_lop)
        """
        batch, seq_len = input_ids.shape
        ids_flat = input_ids.astype(int).flatten()

        valid_mask = (ids_flat >= 0) & (ids_flat < self.so_tu_vung)
        ids_safe = np.where(valid_mask, ids_flat, 0)

        X = self.embedding[ids_safe].reshape(batch, seq_len, self.d_model)
        X = X + self.positional[:seq_len]

        for block in self.blocks:
            X = block.tien(X)

        cls_output = X[:, 0, :]
        logits = cls_output @ self.W_classify + self.b_classify

        return logits

    def du_doan(self, input_ids: np.ndarray) -> np.ndarray:
        """Dự đoán class."""
        logits = self.tien(input_ids)
        return np.argmax(logits, axis=-1)

    def du_doan_xac_suat(self, input_ids: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất."""
        logits = self.tien(input_ids)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def lay_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """Lấy embeddings (hidden states từ CLS token)."""
        batch, seq_len = input_ids.shape
        ids_flat = input_ids.astype(int).flatten()
        valid_mask = (ids_flat >= 0) & (ids_flat < self.so_tu_vung)
        ids_safe = np.where(valid_mask, ids_flat, 0)

        X = self.embedding[ids_safe].reshape(batch, seq_len, self.d_model)
        X = X + self.positional[:seq_len]

        for block in self.blocks:
            X = block.tien(X)

        return X[:, 0, :]

    def thong_ke(self) -> Dict:
        """Thống kê model."""
        so_tham_so = (
            self.so_tu_vung * self.d_model
            + self.so_block * (self.d_model * self.d_ff * 2 + self.d_model * self.d_model * 4)
            + self.d_model * self.so_lop
        )
        return {
            "d_model": self.d_model,
            "so_dau": self.so_dau,
            "d_ff": self.d_ff,
            "so_block": self.so_block,
            "so_tu_vung": self.so_tu_vung,
            "so_lop": self.so_lop,
            "so_tham_so": so_tham_so,
            "so_tham_so_str": f"{so_tham_so / 1e6:.2f}M",
        }
