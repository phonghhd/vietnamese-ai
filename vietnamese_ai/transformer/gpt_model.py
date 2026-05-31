"""GPTModel - GPT-style Decoder-Only Transformer cho tiếng Việt."""

import math
from typing import Dict

import numpy as np

from vietnamese_ai.utils.logger import Logger


class CausalSelfAttention:
    """Causal (masked) self-attention cho decoder."""

    def __init__(self, d_model: int, so_dau: int, dropout: float = 0.1):
        if d_model % so_dau != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_dau ({so_dau})")

        self.d_model = d_model
        self.so_dau = so_dau
        self.d_k = d_model // so_dau
        self.dropout = dropout

        scale = math.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        self._attention_weights = None

    def tien(self, X: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = X.shape

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q = Q.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)

        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores[:, :, causal_mask] = -1e9

        weights = self._softmax(scores)
        self._attention_weights = weights

        output = weights @ V
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return output @ self.W_o

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)


class GPTBlock:
    """GPT Transformer Block: CausalAttention + FFN + LayerNorm."""

    def __init__(self, d_model: int, so_dau: int, d_ff: int, dropout: float = 0.1):
        self.attention = CausalSelfAttention(d_model, so_dau, dropout)
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

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    def tien(self, X: np.ndarray) -> np.ndarray:
        normed = self._layer_norm(X, self.gamma1, self.beta1)
        attn_out = self.attention.tien(normed)
        X = X + attn_out

        normed = self._layer_norm(X, self.gamma2, self.beta2)
        ff = self._gelu(normed @ self.W1 + self.b1) @ self.W2 + self.b2
        X = X + ff

        return X


class GPTModel:
    """
    GPT-style Decoder-Only Transformer.

    Kiến trúc GPT cho text generation tiếng Việt:
    - Causal self-attention (masked)
    - GELU activation
    - Pre-norm LayerNorm
    - Sinusoidal positional encoding
    - Weight tying (embedding ↔ output)

    Sử dụng:
        >>> model = GPTModel(d_model=128, so_dau=4, so_block=4, so_tu_vung=5000)
        >>> logits = model.tien(input_ids)
        >>> next_token = model.sinh_tiep(input_ids, nhiet_do=0.8)
    """

    def __init__(
        self,
        d_model: int = 128,
        so_dau: int = 4,
        d_ff: int = 512,
        so_block: int = 4,
        so_tu_vung: int = 5000,
        do_dai_toi_da: int = 1024,
        dropout: float = 0.1,
    ):
        if d_model % so_dau != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_dau ({so_dau})")

        self.d_model = d_model
        self.so_dau = so_dau
        self.d_ff = d_ff
        self.so_block = so_block
        self.so_tu_vung = so_tu_vung
        self.do_dai_toi_da = do_dai_toi_da
        self.logger = Logger("GPTModel")

        scale = math.sqrt(2.0 / (d_model + so_tu_vung))
        self.embedding = np.random.randn(so_tu_vung, d_model) * scale
        self.positional = self._tao_positional(do_dai_toi_da, d_model)

        self.blocks = [GPTBlock(d_model, so_dau, d_ff, dropout) for _ in range(so_block)]

        self.ln_final_gamma = np.ones((1, 1, d_model))
        self.ln_final_beta = np.zeros((1, 1, d_model))

        self.logger.info(
            f"GPTModel: d_model={d_model}, so_dau={so_dau}, so_block={so_block}, vocab={so_tu_vung}"
        )

    def _tao_positional(self, do_dai: int, d_model: int) -> np.ndarray:
        pe = np.zeros((do_dai, d_model))
        position = np.arange(do_dai).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _layer_norm(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return gamma * (X - mean) / np.sqrt(var + 1e-6) + beta

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def tien(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass - trả về logits cho next token prediction.

        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Logits (batch, seq_len, so_tu_vung)
        """
        batch, seq_len = input_ids.shape
        seq_len = min(seq_len, self.do_dai_toi_da)

        ids_flat = input_ids[:, :seq_len].astype(int).flatten()
        valid_mask = (ids_flat >= 0) & (ids_flat < self.so_tu_vung)
        ids_safe = np.where(valid_mask, ids_flat, 0)

        X = self.embedding[ids_safe].reshape(batch, seq_len, self.d_model)
        X = X + self.positional[:seq_len]

        for block in self.blocks:
            X = block.tien(X)

        X = self._layer_norm(X, self.ln_final_gamma, self.ln_final_beta)

        logits = X @ self.embedding.T

        return logits

    def sinh_tiep(
        self,
        input_ids: np.ndarray,
        so_token: int = 50,
        nhiet_do: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> np.ndarray:
        """
        Sinh token tiếp theo tự hồi quy.

        Args:
            input_ids: Token IDs ban đầu (batch, seq_len)
            so_token: Số token cần sinh
            nhiet_do: Temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Token IDs (batch, seq_len + so_token)
        """
        generated = input_ids.copy()

        for _ in range(so_token):
            logits = self.tien(generated)
            next_logits = logits[:, -1, :] / max(nhiet_do, 1e-8)

            if top_k > 0:
                top_k_clamped = min(top_k, next_logits.shape[-1])
                threshold_val = np.sort(next_logits, axis=-1)[:, -top_k_clamped]
                indices_to_remove = next_logits < threshold_val[:, np.newaxis]
                next_logits[indices_to_remove] = -1e9

            if top_p > 0:
                sorted_indices = np.argsort(-next_logits, axis=-1)
                sorted_logits = np.take_along_axis(next_logits, sorted_indices, axis=-1)
                cumulative_probs = np.cumsum(self._softmax(sorted_logits), axis=-1)
                mask = cumulative_probs - self._softmax(sorted_logits) >= top_p
                sorted_logits[mask] = -1e9
                next_logits = np.take_along_axis(
                    sorted_logits,
                    np.argsort(sorted_indices, axis=-1),
                    axis=-1,
                )

            probs = self._softmax(next_logits)
            next_token = np.array(
                [np.random.choice(self.so_tu_vung, p=probs[i]) for i in range(probs.shape[0])]
            ).reshape(-1, 1)

            generated = np.concatenate([generated, next_token], axis=1)

        return generated

    def tinh_loss(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """Tính cross-entropy loss."""
        logits = self.tien(input_ids)
        batch, seq_len, vocab = logits.shape

        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets[:, :seq_len].flatten().astype(int)

        valid_mask = (flat_targets >= 0) & (flat_targets < vocab)
        flat_logits = flat_logits[valid_mask]
        flat_targets = flat_targets[valid_mask]

        if len(flat_targets) == 0:
            return 0.0

        exp_logits = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        log_probs = np.log(probs[np.arange(len(flat_targets)), flat_targets] + 1e-10)
        return float(-np.mean(log_probs))

    def thong_ke(self) -> Dict:
        so_tham_so = (
            self.so_tu_vung * self.d_model
            + self.so_block
            * (self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2 + self.d_model * 4)
            + self.d_model * 2
        )
        return {
            "loai": "GPT (Decoder-Only)",
            "d_model": self.d_model,
            "so_dau": self.so_dau,
            "d_ff": self.d_ff,
            "so_block": self.so_block,
            "so_tu_vung": self.so_tu_vung,
            "do_dai_toi_da": self.do_dai_toi_da,
            "so_tham_so": so_tham_so,
            "so_tham_so_str": f"{so_tham_so / 1e6:.2f}M",
        }
