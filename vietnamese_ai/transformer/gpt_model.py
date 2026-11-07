"""GPTModel - GPT-style Decoder-Only Transformer cho tiếng Việt."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.fine_tuning.lora import DoRALayer, LoRALayer
from vietnamese_ai.utils.logger import Logger


def _forward_lora_proj(
    X: np.ndarray,
    W: np.ndarray,
    lora: Optional[LoRALayer] = None,
) -> np.ndarray:
    """Thực hiện phép chiếu tuyến tính kèm nhánh LoRA / rsLoRA / DoRA."""
    if lora is not None and isinstance(lora, DoRALayer):
        orig_shape = X.shape[:-1] + (W.shape[1],)
        return lora.tien_dora(X.reshape(-1, W.shape[0]), W).reshape(orig_shape)
    out = X @ W
    if lora is not None:
        out = out + lora.tien(X)
    return out


def _backward_lora_proj(
    X_flat: np.ndarray,
    dY_flat: np.ndarray,
    W: np.ndarray,
    lora: Optional[LoRALayer] = None,
    tinh_grad_goc: bool = True,
) -> tuple:
    """
    Tính gradient cho phép chiếu tuyến tính kèm nhánh LoRA / rsLoRA / DoRA.
    Khi `tinh_grad_goc=False` (Unsloth-style Manual Autograd cho trọng số đóng băng),
    bỏ qua phép nhân ma trận đầy đủ `X_flat.T @ dY_flat`, giảm ~50% chi phí tính toán backward!
    """
    if lora is not None and isinstance(lora, DoRALayer) and not getattr(lora, "_merged", False):
        grads = lora.ve_dora(X_flat, dY_flat, W)
        dW = (X_flat.T @ dY_flat) if tinh_grad_goc else None
        return grads["dX"], dW, {"dA": grads["dA"], "dB": grads["dB"], "dm": grads["dm"]}

    dW = (X_flat.T @ dY_flat) if tinh_grad_goc else None
    dX_flat = dY_flat @ W.T
    lora_grad = None
    if lora is not None and not getattr(lora, "_merged", False):
        s = lora.scaling
        H = X_flat @ lora.A
        dB = s * (H.T @ dY_flat)
        dY_B = dY_flat @ lora.B.T
        dA = s * (X_flat.T @ dY_B)
        dX_flat = dX_flat + s * (dY_B @ lora.A.T)
        lora_grad = {"dA": dA, "dB": dB}
    return dX_flat, dW, lora_grad


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

        self.lora_layers: Dict[str, LoRALayer] = {}
        self._attention_weights = None
        self._cache: Dict = {}

    def tien(self, X: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch, seq_len, _ = X.shape

        Q_lin = _forward_lora_proj(X, self.W_q, self.lora_layers.get("W_q"))
        K_lin = _forward_lora_proj(X, self.W_k, self.lora_layers.get("W_k"))
        V_lin = _forward_lora_proj(X, self.W_v, self.lora_layers.get("W_v"))

        Q = Q_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        K = K_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        V = V_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len, seq_len), True = được phép chú ý
            mask_4d = ~attention_mask[:, None, :seq_len, :seq_len].astype(bool)
            scores = np.where(mask_4d, -1e9, scores)
        else:
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[:, :, causal_mask] = -1e9
            mask_4d = None

        weights = self._softmax(scores)
        self._attention_weights = weights

        attn_v = weights @ V
        concat = attn_v.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        self._cache = {
            "X": X,
            "Q": Q,
            "K": K,
            "V": V,
            "weights": weights,
            "concat": concat,
            "mask_4d": mask_4d,
        }
        out = _forward_lora_proj(concat, self.W_o, self.lora_layers.get("W_o"))
        return out

    def tien_kv(self, X: np.ndarray, kv_cache: Optional[tuple] = None) -> tuple:
        """Forward suy luận nhanh với bộ nhớ đệm Key-Value (`KV-Cache` — O(T) mỗi bước)."""
        batch, seq_len, _ = X.shape
        Q_lin = _forward_lora_proj(X, self.W_q, self.lora_layers.get("W_q"))
        K_lin = _forward_lora_proj(X, self.W_k, self.lora_layers.get("W_k"))
        V_lin = _forward_lora_proj(X, self.W_v, self.lora_layers.get("W_v"))

        Q = Q_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        K_new = K_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        V_new = V_lin.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K_all = np.concatenate([K_prev, K_new], axis=2)
            V_all = np.concatenate([V_prev, V_new], axis=2)
        else:
            K_all, V_all = K_new, V_new

        scores = Q @ K_all.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        if seq_len > 1:
            total_k = K_all.shape[2]
            offset = total_k - seq_len
            q_idx = np.arange(seq_len)[:, None] + offset
            k_idx = np.arange(total_k)[None, :]
            causal_mask = k_idx > q_idx
            scores[:, :, causal_mask] = -1e9

        weights = self._softmax(scores)
        attn_v = weights @ V_all
        concat = attn_v.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        out = _forward_lora_proj(concat, self.W_o, self.lora_layers.get("W_o"))
        return out, (K_all, V_all)

    def nguoc(self, d_out: np.ndarray, tinh_grad_goc: bool = True) -> tuple:
        X = self._cache["X"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        weights = self._cache["weights"]
        concat = self._cache["concat"]
        batch, seq_len, _ = X.shape

        X_flat = X.reshape(-1, self.d_model)
        d_out_flat = d_out.reshape(-1, self.d_model)
        concat_flat = concat.reshape(-1, self.d_model)

        d_concat_flat, dW_o, lora_o = _backward_lora_proj(
            concat_flat, d_out_flat, self.W_o, self.lora_layers.get("W_o"), tinh_grad_goc=tinh_grad_goc
        )
        d_concat = d_concat_flat.reshape(batch, seq_len, self.d_model)

        d_attn_v = d_concat.reshape(batch, seq_len, self.so_dau, self.d_k).transpose(0, 2, 1, 3)
        d_weights = d_attn_v @ V.transpose(0, 1, 3, 2)
        dV = weights.transpose(0, 1, 3, 2) @ d_attn_v

        d_scores = weights * (d_weights - np.sum(d_weights * weights, axis=-1, keepdims=True))
        mask_4d = self._cache.get("mask_4d")
        if mask_4d is not None:
            d_scores = np.where(mask_4d, 0.0, d_scores)
        else:
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            d_scores[:, :, causal_mask] = 0.0
        d_scores = d_scores / math.sqrt(self.d_k)

        dQ = d_scores @ K
        dK = d_scores.transpose(0, 1, 3, 2) @ Q

        dQ_lin = dQ.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        dK_lin = dK.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        dV_lin = dV.transpose(0, 2, 1, 3).reshape(-1, self.d_model)

        dX_q, dW_q, lora_q = _backward_lora_proj(
            X_flat, dQ_lin, self.W_q, self.lora_layers.get("W_q"), tinh_grad_goc=tinh_grad_goc
        )
        dX_k, dW_k, lora_k = _backward_lora_proj(
            X_flat, dK_lin, self.W_k, self.lora_layers.get("W_k"), tinh_grad_goc=tinh_grad_goc
        )
        dX_v, dW_v, lora_v = _backward_lora_proj(
            X_flat, dV_lin, self.W_v, self.lora_layers.get("W_v"), tinh_grad_goc=tinh_grad_goc
        )

        dX = (dX_q + dX_k + dX_v).reshape(batch, seq_len, self.d_model)
        lora_grads = {}
        if lora_q is not None:
            lora_grads["W_q"] = lora_q
        if lora_k is not None:
            lora_grads["W_k"] = lora_k
        if lora_v is not None:
            lora_grads["W_v"] = lora_v
        if lora_o is not None:
            lora_grads["W_o"] = lora_o

        return dX, {
            "W_q": dW_q,
            "W_k": dW_k,
            "W_v": dW_v,
            "W_o": dW_o,
            "lora": lora_grads,
        }

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
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.moe_layer: Optional[Any] = None
        self._cache: Dict = {}

    def _layer_norm(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return gamma * (X - mean) / np.sqrt(var + 1e-6) + beta

    @staticmethod
    def _layer_norm_backward(
        d_out: np.ndarray, X: np.ndarray, gamma: np.ndarray
    ) -> tuple:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + 1e-6)
        x_hat = (X - mean) * std_inv
        d_gamma = np.sum(d_out * x_hat, axis=(0, 1), keepdims=True)
        d_beta = np.sum(d_out, axis=(0, 1), keepdims=True)
        dx_hat = d_out * gamma
        D = X.shape[-1]
        dX = (1.0 / D) * std_inv * (
            D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dX, d_gamma, d_beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def _gelu_grad(x: np.ndarray) -> np.ndarray:
        k = math.sqrt(2.0 / math.pi)
        inner = k * (x + 0.044715 * x**3)
        tanh_val = np.tanh(inner)
        sech2 = 1.0 - tanh_val**2
        d_inner = k * (1.0 + 3.0 * 0.044715 * x**2)
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner

    def tien(self, X: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        X_in1 = X
        normed1 = self._layer_norm(X_in1, self.gamma1, self.beta1)
        attn_out = self.attention.tien(normed1, attention_mask=attention_mask)
        X_mid = X_in1 + attn_out

        normed2 = self._layer_norm(X_mid, self.gamma2, self.beta2)
        pre_act = _forward_lora_proj(normed2, self.W1, self.lora_layers.get("W1")) + self.b1
        act = self._gelu(pre_act)
        ff = _forward_lora_proj(act, self.W2, self.lora_layers.get("W2")) + self.b2
        if self.moe_layer is not None:
            ff = 0.5 * ff + 0.5 * self.moe_layer.tien(normed2)
        X_out = X_mid + ff
        self._cache = {
            "X_in1": X_in1,
            "X_mid": X_mid,
            "normed2": normed2,
            "pre_act": pre_act,
            "act": act,
        }
        return X_out

    def tien_kv(self, X: np.ndarray, kv_cache: Optional[tuple] = None) -> tuple:
        """Forward 1 block sử dụng KV-Cache."""
        normed1 = self._layer_norm(X, self.gamma1, self.beta1)
        attn_out, new_kv = self.attention.tien_kv(normed1, kv_cache=kv_cache)
        X_mid = X + attn_out
        normed2 = self._layer_norm(X_mid, self.gamma2, self.beta2)
        pre_act = _forward_lora_proj(normed2, self.W1, self.lora_layers.get("W1")) + self.b1
        act = self._gelu(pre_act)
        ff = _forward_lora_proj(act, self.W2, self.lora_layers.get("W2")) + self.b2
        if self.moe_layer is not None:
            ff = 0.5 * ff + 0.5 * self.moe_layer.tien(normed2)
        return X_mid + ff, new_kv

    def nguoc(self, d_out: np.ndarray, tinh_grad_goc: bool = True) -> tuple:
        X_in1 = self._cache["X_in1"]
        X_mid = self._cache["X_mid"]
        normed2 = self._cache["normed2"]
        pre_act = self._cache["pre_act"]
        act = self._cache["act"]
        batch, seq_len, _ = d_out.shape

        d_out_flat = d_out.reshape(-1, self.d_model)
        act_flat = act.reshape(-1, self.d_ff)
        d_act_flat, dW2, lora_w2 = _backward_lora_proj(
            act_flat, d_out_flat, self.W2, self.lora_layers.get("W2"), tinh_grad_goc=tinh_grad_goc
        )
        db2 = np.sum(d_out_flat, axis=0, keepdims=True)
        d_act = d_act_flat.reshape(batch, seq_len, self.d_ff)

        d_pre_act = d_act * self._gelu_grad(pre_act)
        d_pre_flat = d_pre_act.reshape(-1, self.d_ff)
        normed2_flat = normed2.reshape(-1, self.d_model)
        d_normed2_flat, dW1, lora_w1 = _backward_lora_proj(
            normed2_flat, d_pre_flat, self.W1, self.lora_layers.get("W1"), tinh_grad_goc=tinh_grad_goc
        )
        db1 = np.sum(d_pre_flat, axis=0, keepdims=True)
        d_normed2 = d_normed2_flat.reshape(batch, seq_len, self.d_model)

        dX_ln2, d_gamma2, d_beta2 = self._layer_norm_backward(d_normed2, X_mid, self.gamma2)
        dX_mid = d_out + dX_ln2

        d_normed1, attn_grads = self.attention.nguoc(dX_mid, tinh_grad_goc=tinh_grad_goc)
        dX_ln1, d_gamma1, d_beta1 = self._layer_norm_backward(d_normed1, X_in1, self.gamma1)
        dX_in1 = dX_mid + dX_ln1

        lora_grads = {}
        if lora_w1 is not None:
            lora_grads["W1"] = lora_w1
        if lora_w2 is not None:
            lora_grads["W2"] = lora_w2

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "gamma1": d_gamma1,
            "beta1": d_beta1,
            "gamma2": d_gamma2,
            "beta2": d_beta2,
            "attn": attn_grads,
            "lora": lora_grads,
        }
        return dX_in1, grads


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
        self.da_gan_lora: bool = False
        self.lora_config: Optional[Dict[str, Any]] = None
        self.da_kich_hoat_moe: bool = False
        self.moe_config: Optional[Dict[str, Any]] = None
        self._cache: Dict = {}
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t: int = 0

        self.logger.info(
            f"GPTModel: d_model={d_model}, so_dau={so_dau}, so_block={so_block}, vocab={so_tu_vung}"
        )

    def gan_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        dung_rslora: bool = False,
        dung_dora: bool = False,
    ) -> Dict[str, Any]:
        """
        Gắn các lớp LoRA / rsLoRA / DoRA vào các block của GPTModel.
        Khi bật LoRA, các ma trận gốc (W_q, W_k, W_v, W_o, W1, W2, embedding) sẽ được đóng băng
        và chỉ các ma trận hạng thấp (A, B, hoặc thêm vector biên độ m của DoRA) cùng LayerNorm được cập nhật.
        """
        alias_map = {
            "q_proj": "W_q",
            "k_proj": "W_k",
            "v_proj": "W_v",
            "o_proj": "W_o",
            "out_proj": "W_o",
            "up_proj": "W1",
            "gate_proj": "W1",
            "fc1": "W1",
            "down_proj": "W2",
            "fc2": "W2",
        }
        if target_modules is None:
            resolved_targets = ["W_q", "W_k", "W_v", "W_o", "W1", "W2"]
        else:
            resolved_targets = []
            for m in target_modules:
                canonical = alias_map.get(m, m)
                if canonical not in resolved_targets and canonical in (
                    "W_q",
                    "W_k",
                    "W_v",
                    "W_o",
                    "W1",
                    "W2",
                ):
                    resolved_targets.append(canonical)
            if not resolved_targets:
                resolved_targets = ["W_q", "W_v"]

        def _tao_lop_adapter(d_in: int, d_out: int, w_goc: np.ndarray) -> LoRALayer:
            if dung_dora:
                return DoRALayer(
                    d_in,
                    d_out,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    dung_rslora=dung_rslora,
                    trong_so_goc=w_goc,
                )
            return LoRALayer(
                d_in, d_out, rank=rank, alpha=alpha, dropout=dropout, dung_rslora=dung_rslora
            )

        so_tham_so_lora = 0
        so_lop_lora = 0
        for block in self.blocks:
            block.attention.lora_layers.clear()
            block.lora_layers.clear()
            for name in ("W_q", "W_k", "W_v", "W_o"):
                if name in resolved_targets:
                    layer = _tao_lop_adapter(
                        self.d_model, self.d_model, getattr(block.attention, name)
                    )
                    block.attention.lora_layers[name] = layer
                    so_tham_so_lora += layer.so_tham_so()
                    so_lop_lora += 1
            if "W1" in resolved_targets:
                layer_w1 = _tao_lop_adapter(self.d_model, self.d_ff, block.W1)
                block.lora_layers["W1"] = layer_w1
                so_tham_so_lora += layer_w1.so_tham_so()
                so_lop_lora += 1
            if "W2" in resolved_targets:
                layer_w2 = _tao_lop_adapter(self.d_ff, self.d_model, block.W2)
                block.lora_layers["W2"] = layer_w2
                so_tham_so_lora += layer_w2.so_tham_so()
                so_lop_lora += 1

        self.da_gan_lora = True
        tong_tham_so = self._tinh_tong_tham_so_goc()
        ty_le = round(so_tham_so_lora / max(1, tong_tham_so + so_tham_so_lora) * 100, 2)
        self.lora_config = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "dung_rslora": dung_rslora,
            "dung_dora": dung_dora,
            "target_modules": resolved_targets,
            "so_lop_lora": so_lop_lora,
            "so_tham_so_lora": so_tham_so_lora,
            "tong_tham_so_goc": tong_tham_so,
            "ty_le_huan_luyen_pct": ty_le,
        }
        return self.lora_config.copy()

    def kich_hoat_moe(
        self,
        so_chuyen_gia: int = 4,
        top_k: int = 2,
        he_so_can_bang: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Nâng cấp các block của GPTModel lên kiến trúc Sparse Mixture-of-Experts (MoE)
        với Top-K Expert Routing và Load Balancing Auxiliary Loss (phong cách Qwen3.8-MoE / Llama 4).
        """
        from vietnamese_ai.transformer.moe_rope import SparseMoELayer

        for block in self.blocks:
            block.moe_layer = SparseMoELayer(
                d_model=self.d_model,
                d_ff=self.d_ff,
                so_chuyen_gia=so_chuyen_gia,
                top_k=top_k,
                he_so_can_bang=he_so_can_bang,
            )
        self.da_kich_hoat_moe = True
        self.moe_config = {
            "so_chuyen_gia": so_chuyen_gia,
            "top_k": top_k,
            "he_so_can_bang": he_so_can_bang,
            "so_block_moe": len(self.blocks),
        }
        return self.moe_config.copy()

    def tinh_aux_loss_moe(self) -> float:
        """Tính tổng Load Balancing Auxiliary Loss của tất cả các tầng MoE."""
        if not self.da_kich_hoat_moe:
            return 0.0
        losses = [
            block.moe_layer.aux_loss
            for block in self.blocks
            if block.moe_layer is not None
        ]
        return float(np.mean(losses)) if losses else 0.0

    def gop_lora(self) -> int:
        """Gộp trọng số LoRA / rsLoRA / DoRA vào ma trận gốc để suy luận không tốn thêm chi phí."""
        merged_count = 0
        for block in self.blocks:
            attn = block.attention
            for name in list(attn.lora_layers.keys()):
                lora = attn.lora_layers[name]
                w_goc = getattr(attn, name)
                if isinstance(lora, DoRALayer):
                    setattr(attn, name, lora.gop_trong_so_dora(w_goc))
                else:
                    setattr(attn, name, w_goc + lora.gop_trong_so())
                merged_count += 1
            attn.lora_layers.clear()

            for name in list(block.lora_layers.keys()):
                lora = block.lora_layers[name]
                w_goc = getattr(block, name)
                if isinstance(lora, DoRALayer):
                    setattr(block, name, lora.gop_trong_so_dora(w_goc))
                else:
                    setattr(block, name, w_goc + lora.gop_trong_so())
                merged_count += 1
            block.lora_layers.clear()

        self.da_gan_lora = False
        return merged_count

    def luu_lora(self, duong_dan: str) -> Dict[str, Any]:
        """Lưu các ma trận LoRA ra file nén (.npz)."""
        if not self.da_gan_lora:
            raise RuntimeError("GPTModel chưa được gắn LoRA.")
        arrays: Dict[str, np.ndarray] = {}
        for idx, block in enumerate(self.blocks):
            for name, lora in block.attention.lora_layers.items():
                arrays[f"b{idx}_attn_{name}_A"] = lora.A
                arrays[f"b{idx}_attn_{name}_B"] = lora.B
            for name, lora in block.lora_layers.items():
                arrays[f"b{idx}_ffn_{name}_A"] = lora.A
                arrays[f"b{idx}_ffn_{name}_B"] = lora.B
        np.savez_compressed(duong_dan, **arrays)
        return {"duong_dan": duong_dan, "so_mang": len(arrays), "config": self.lora_config}

    def tai_lora(self, duong_dan: str) -> Dict[str, Any]:
        """Tải các ma trận LoRA từ file (.npz)."""
        if not duong_dan.endswith(".npz"):
            duong_dan_npz = duong_dan + ".npz"
        else:
            duong_dan_npz = duong_dan
        import os

        actual_path = duong_dan if os.path.exists(duong_dan) else duong_dan_npz
        with np.load(actual_path) as data:
            loaded = 0
            for idx, block in enumerate(self.blocks):
                for name, lora in block.attention.lora_layers.items():
                    ka, kb = f"b{idx}_attn_{name}_A", f"b{idx}_attn_{name}_B"
                    if ka in data and kb in data:
                        lora.A = data[ka].copy()
                        lora.B = data[kb].copy()
                        loaded += 1
                for name, lora in block.lora_layers.items():
                    ka, kb = f"b{idx}_ffn_{name}_A", f"b{idx}_ffn_{name}_B"
                    if ka in data and kb in data:
                        lora.A = data[ka].copy()
                        lora.B = data[kb].copy()
                        loaded += 1
        return {"duong_dan": actual_path, "so_lop_da_tai": loaded}

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

    def tien(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        tinh_logits: bool = True,
    ) -> np.ndarray:
        """
        Forward pass - trả về logits cho next token prediction (hoặc X_norm khi dùng Chunked Cross-Entropy).

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Mặt nạ chú ý tùy chọn (batch, seq_len, seq_len) cho Sequence Packing
            tinh_logits: Nếu False, trả về trạng thái ẩn X_norm thay vì nhân toàn bộ ma trận từ vựng

        Returns:
            Logits (batch, seq_len, so_tu_vung) hoặc X_norm (batch, seq_len, d_model)
        """
        batch, seq_len = input_ids.shape
        seq_len = min(seq_len, self.do_dai_toi_da)

        ids_flat = input_ids[:, :seq_len].astype(int).flatten()
        valid_mask = (ids_flat >= 0) & (ids_flat < self.so_tu_vung)
        ids_safe = np.where(valid_mask, ids_flat, 0)

        X = self.embedding[ids_safe].reshape(batch, seq_len, self.d_model)
        X = X + self.positional[:seq_len]

        for block in self.blocks:
            X = block.tien(X, attention_mask=attention_mask)

        X_pre_ln = X
        X_norm = self._layer_norm(X_pre_ln, self.ln_final_gamma, self.ln_final_beta)

        self._cache = {
            "ids_safe": ids_safe,
            "valid_mask": valid_mask,
            "X_pre_ln": X_pre_ln,
            "X_norm": X_norm,
            "batch": batch,
            "seq_len": seq_len,
        }
        if not tinh_logits:
            return X_norm

        return X_norm @ self.embedding.T

    def tien_kv(
        self,
        input_ids: np.ndarray,
        kv_caches: Optional[List[Any]] = None,
        pos_offset: int = 0,
    ) -> tuple:
        """Forward bước đơn sử dụng `KV-Cache` (giảm độ phức tạp sinh tự hồi quy từ O(T^2) xuống O(T))."""
        batch, seq_len = input_ids.shape
        ids_flat = input_ids.astype(int).flatten()
        valid_mask = (ids_flat >= 0) & (ids_flat < self.so_tu_vung)
        ids_safe = np.where(valid_mask, ids_flat, 0)

        X = self.embedding[ids_safe].reshape(batch, seq_len, self.d_model)
        pos_indices = np.clip(np.arange(pos_offset, pos_offset + seq_len), 0, self.do_dai_toi_da - 1)
        X = X + self.positional[pos_indices]

        new_caches = []
        for idx, block in enumerate(self.blocks):
            prev_kv = kv_caches[idx] if kv_caches is not None else None
            X, next_kv = block.tien_kv(X, kv_cache=prev_kv)
            new_caches.append(next_kv)

        X_norm = self._layer_norm(X, self.ln_final_gamma, self.ln_final_beta)
        next_logits = X_norm[:, -1, :] @ self.embedding.T
        return next_logits, new_caches

    def sinh_tiep(
        self,
        input_ids: np.ndarray,
        so_token: int = 50,
        nhiet_do: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        dung_kv_cache: bool = True,
    ) -> np.ndarray:
        """
        Sinh token tiếp theo tự hồi quy (mặc định kích hoạt `KV-Cache` tăng tốc 3x - 8x).

        Args:
            input_ids: Token IDs ban đầu (batch, seq_len)
            so_token: Số token cần sinh
            nhiet_do: Temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            dung_kv_cache: Sử dụng bộ nhớ đệm Key-Value để không tính lại các token cũ

        Returns:
            Token IDs (batch, seq_len + so_token)
        """
        generated = input_ids.copy()
        kv_caches: Optional[List[Any]] = None

        for step in range(so_token):
            if dung_kv_cache and generated.shape[1] <= self.do_dai_toi_da:
                if step == 0 or kv_caches is None:
                    raw_next_logits, kv_caches = self.tien_kv(generated, kv_caches=None, pos_offset=0)
                else:
                    last_tok = generated[:, -1:]
                    raw_next_logits, kv_caches = self.tien_kv(
                        last_tok, kv_caches=kv_caches, pos_offset=generated.shape[1] - 1
                    )
                next_logits = raw_next_logits / max(nhiet_do, 1e-8)
            else:
                window = generated[:, -self.do_dai_toi_da :]
                logits = self.tien(window)
                next_logits = logits[:, -1, :] / max(nhiet_do, 1e-8)
                kv_caches = None

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

    def _adamw_update(
        self,
        key: str,
        param: np.ndarray,
        grad: np.ndarray,
        lr: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> np.ndarray:
        if key not in self._adam_m:
            self._adam_m[key] = np.zeros_like(param)
            self._adam_v[key] = np.zeros_like(param)
        m = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
        v = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad * grad)
        self._adam_m[key] = m
        self._adam_v[key] = v
        m_hat = m / (1.0 - beta1**self._adam_t)
        v_hat = v / (1.0 - beta2**self._adam_t)
        if weight_decay > 0 and param.ndim >= 2:
            param = param * (1.0 - lr * weight_decay)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    def buoc_huan_luyen(
        self,
        input_ids: np.ndarray,
        targets: np.ndarray,
        toc_do_hoc: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        attention_mask: Optional[np.ndarray] = None,
        dung_chunked_ce: bool = False,
    ) -> float:
        """Thực hiện 1 bước forward + analytical backward + cập nhật trọng số AdamW (hỗ trợ Sequence Packing & Chunked CE)."""
        tinh_grad_goc = not self.da_gan_lora
        if dung_chunked_ce:
            from vietnamese_ai.fine_tuning.training_accelerator import HamMatMatPhanManh

            X_norm = self.tien(input_ids, attention_mask=attention_mask, tinh_logits=False)
            batch, seq_len, _ = X_norm.shape
            flat_targets = targets[:, :seq_len].flatten().astype(int)
            X_norm_flat = X_norm.reshape(-1, self.d_model)
            loss, dX_norm_flat, d_embedding = HamMatMatPhanManh.tinh_loss_va_grad_numpy(
                X_norm_flat,
                self.embedding,
                flat_targets,
                chunk_size=64,
                tinh_grad_tu_vung=tinh_grad_goc,
            )
            if loss == 0.0 and not np.any((flat_targets >= 0) & (flat_targets < self.so_tu_vung)):
                return 0.0
            dX_norm = dX_norm_flat.reshape(batch, seq_len, self.d_model)
        else:
            logits = self.tien(input_ids, attention_mask=attention_mask)
            batch, seq_len, vocab = logits.shape

            flat_logits = logits.reshape(-1, vocab)
            flat_targets = targets[:, :seq_len].flatten().astype(int)
            valid_mask = (flat_targets >= 0) & (flat_targets < vocab)
            n_valid = int(np.sum(valid_mask))
            if n_valid == 0:
                return 0.0

            max_logits = np.max(flat_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(flat_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            valid_indices = np.where(valid_mask)[0]
            valid_targets = flat_targets[valid_mask]
            loss = float(-np.mean(np.log(probs[valid_indices, valid_targets] + 1e-10)))

            d_flat_logits = np.zeros_like(flat_logits)
            d_flat_logits[valid_indices] = probs[valid_indices]
            d_flat_logits[valid_indices, valid_targets] -= 1.0
            d_flat_logits /= float(n_valid)

            X_norm = self._cache["X_norm"]
            X_norm_flat = X_norm.reshape(-1, self.d_model)
            d_embedding = (d_flat_logits.T @ X_norm_flat) if tinh_grad_goc else None
            dX_norm = (d_flat_logits @ self.embedding).reshape(batch, seq_len, self.d_model)

        X_pre_ln = self._cache["X_pre_ln"]
        ids_safe = self._cache["ids_safe"]
        in_valid_mask = self._cache["valid_mask"]

        dX, d_ln_gamma, d_ln_beta = GPTBlock._layer_norm_backward(
            dX_norm, X_pre_ln, self.ln_final_gamma
        )

        block_grads = [None] * len(self.blocks)
        for idx in range(len(self.blocks) - 1, -1, -1):
            dX, b_grad = self.blocks[idx].nguoc(dX, tinh_grad_goc=tinh_grad_goc)
            block_grads[idx] = b_grad

        if tinh_grad_goc and d_embedding is not None:
            dX_flat = dX.reshape(-1, self.d_model)
            if np.any(in_valid_mask):
                np.add.at(d_embedding, ids_safe[in_valid_mask], dX_flat[in_valid_mask])

        # Global gradient norm clipping
        if grad_clip > 0:
            if self.da_gan_lora:
                sq_norm = float(np.sum(d_ln_gamma**2) + np.sum(d_ln_beta**2))
                for bg in block_grads:
                    for k in ("b1", "b2", "gamma1", "beta1", "gamma2", "beta2"):
                        sq_norm += float(np.sum(bg[k] ** 2))
                    for lg in bg.get("lora", {}).values():
                        sq_norm += float(np.sum(lg["dA"] ** 2) + np.sum(lg["dB"] ** 2))
                    for lg in bg["attn"].get("lora", {}).values():
                        sq_norm += float(np.sum(lg["dA"] ** 2) + np.sum(lg["dB"] ** 2))
            else:
                sq_norm = float(
                    np.sum(d_embedding**2) + np.sum(d_ln_gamma**2) + np.sum(d_ln_beta**2)
                )
                for bg in block_grads:
                    for k in ("W1", "b1", "W2", "b2", "gamma1", "beta1", "gamma2", "beta2"):
                        sq_norm += float(np.sum(bg[k] ** 2))
                    for k in ("W_q", "W_k", "W_v", "W_o"):
                        sq_norm += float(np.sum(bg["attn"][k] ** 2))
            total_norm = math.sqrt(sq_norm)
            if total_norm > grad_clip:
                scale = grad_clip / (total_norm + 1e-8)
                if d_embedding is not None:
                    d_embedding *= scale
                d_ln_gamma *= scale
                d_ln_beta *= scale
                for bg in block_grads:
                    for k in ("b1", "b2", "gamma1", "beta1", "gamma2", "beta2"):
                        bg[k] *= scale
                    if tinh_grad_goc:
                        for k in ("W1", "W2"):
                            bg[k] *= scale
                        for k in ("W_q", "W_k", "W_v", "W_o"):
                            bg["attn"][k] *= scale
                    for lg in bg.get("lora", {}).values():
                        lg["dA"] *= scale
                        lg["dB"] *= scale
                        if "dm" in lg:
                            lg["dm"] *= scale
                    for lg in bg["attn"].get("lora", {}).values():
                        lg["dA"] *= scale
                        lg["dB"] *= scale
                        if "dm" in lg:
                            lg["dm"] *= scale

        self._adam_t += 1
        self.ln_final_gamma = self._adamw_update(
            "ln_final_gamma", self.ln_final_gamma, d_ln_gamma, toc_do_hoc, 0.0
        )
        self.ln_final_beta = self._adamw_update(
            "ln_final_beta", self.ln_final_beta, d_ln_beta, toc_do_hoc, 0.0
        )

        if not self.da_gan_lora:
            self.embedding = self._adamw_update(
                "embedding", self.embedding, d_embedding, toc_do_hoc, weight_decay
            )

        for idx, block in enumerate(self.blocks):
            bg = block_grads[idx]
            prefix = f"b{idx}_"
            block.b1 = self._adamw_update(prefix + "b1", block.b1, bg["b1"], toc_do_hoc, 0.0)
            block.b2 = self._adamw_update(prefix + "b2", block.b2, bg["b2"], toc_do_hoc, 0.0)
            block.gamma1 = self._adamw_update(
                prefix + "gamma1", block.gamma1, bg["gamma1"], toc_do_hoc, 0.0
            )
            block.beta1 = self._adamw_update(
                prefix + "beta1", block.beta1, bg["beta1"], toc_do_hoc, 0.0
            )
            block.gamma2 = self._adamw_update(
                prefix + "gamma2", block.gamma2, bg["gamma2"], toc_do_hoc, 0.0
            )
            block.beta2 = self._adamw_update(
                prefix + "beta2", block.beta2, bg["beta2"], toc_do_hoc, 0.0
            )

            if self.da_gan_lora:
                for name, lora in block.lora_layers.items():
                    lg = bg.get("lora", {}).get(name)
                    if lg is not None:
                        lora.A = self._adamw_update(
                            f"{prefix}lora_{name}_A", lora.A, lg["dA"], toc_do_hoc, weight_decay
                        )
                        lora.B = self._adamw_update(
                            f"{prefix}lora_{name}_B", lora.B, lg["dB"], toc_do_hoc, 0.0
                        )
                        if "dm" in lg and hasattr(lora, "m"):
                            lora.m = self._adamw_update(
                                f"{prefix}lora_{name}_m", lora.m, lg["dm"], toc_do_hoc, 0.0
                            )
                attn = block.attention
                for name, lora in attn.lora_layers.items():
                    lg = bg["attn"].get("lora", {}).get(name)
                    if lg is not None:
                        lora.A = self._adamw_update(
                            f"{prefix}lora_{name}_A", lora.A, lg["dA"], toc_do_hoc, weight_decay
                        )
                        lora.B = self._adamw_update(
                            f"{prefix}lora_{name}_B", lora.B, lg["dB"], toc_do_hoc, 0.0
                        )
                        if "dm" in lg and hasattr(lora, "m"):
                            lora.m = self._adamw_update(
                                f"{prefix}lora_{name}_m", lora.m, lg["dm"], toc_do_hoc, 0.0
                            )
            else:
                block.W1 = self._adamw_update(
                    prefix + "W1", block.W1, bg["W1"], toc_do_hoc, weight_decay
                )
                block.W2 = self._adamw_update(
                    prefix + "W2", block.W2, bg["W2"], toc_do_hoc, weight_decay
                )
                attn = block.attention
                ag = bg["attn"]
                attn.W_q = self._adamw_update(
                    prefix + "W_q", attn.W_q, ag["W_q"], toc_do_hoc, weight_decay
                )
                attn.W_k = self._adamw_update(
                    prefix + "W_k", attn.W_k, ag["W_k"], toc_do_hoc, weight_decay
                )
                attn.W_v = self._adamw_update(
                    prefix + "W_v", attn.W_v, ag["W_v"], toc_do_hoc, weight_decay
                )
                attn.W_o = self._adamw_update(
                    prefix + "W_o", attn.W_o, ag["W_o"], toc_do_hoc, weight_decay
                )

        return loss

    def _tinh_tong_tham_so_goc(self) -> int:
        return (
            self.so_tu_vung * self.d_model
            + self.so_block
            * (self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2 + self.d_model * 4)
            + self.d_model * 2
        )

    def thong_ke(self) -> Dict:
        so_tham_so = self._tinh_tong_tham_so_goc()
        so_tham_so_lora = 0
        if self.da_gan_lora:
            for block in self.blocks:
                for lora in block.attention.lora_layers.values():
                    so_tham_so_lora += lora.so_tham_so()
                for lora in block.lora_layers.values():
                    so_tham_so_lora += lora.so_tham_so()
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
            "da_gan_lora": self.da_gan_lora,
            "so_tham_so_lora": so_tham_so_lora,
        }

    def huan_luyen(
        self,
        X: Any,
        y: Optional[Any] = None,
        so_vong: int = 3,
        toc_do_hoc: float = 2e-3,
    ) -> Dict[str, Any]:
        """Giao diện huấn luyện chuẩn tương thích với Engine và Pipeline."""
        losses: List[float] = []
        if isinstance(X, np.ndarray):
            in_ids = X.astype(int)
            tgt_ids = y.astype(int) if y is not None else np.roll(in_ids, -1, axis=-1)
            for _ in range(max(1, so_vong)):
                losses.append(self.buoc_huan_luyen(in_ids, tgt_ids, toc_do_hoc=toc_do_hoc))
        elif isinstance(X, list):
            from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

            sft = SFTTrainer(so_vong=so_vong, toc_do_hoc=toc_do_hoc)
            kq = sft.huan_luyen(self, X)
            losses = kq.get("history", {}).get("train_loss", [])
        return {
            "so_vong": so_vong,
            "loss_cuoi": losses[-1] if losses else 0.0,
            "history": losses,
        }

    def du_doan(self, X: Any) -> Any:
        """Giao diện dự đoán chuẩn tương thích với Engine, Pipeline và MayChuBatch."""
        if isinstance(X, np.ndarray):
            logits = self.tien(X.astype(int))
            return np.argmax(logits, axis=-1)
        if isinstance(X, str):
            return self.sinh_van_ban(X)
        if isinstance(X, list):
            return [self.sinh_van_ban(str(item)) for item in X]
        return X

    def sinh_van_ban(self, prompt: str, do_dai: int = 24, nhiet_do: float = 0.7) -> str:
        """Sinh văn bản từ chuỗi đầu vào."""
        raw_ids = [int(b) % self.so_tu_vung for b in prompt.encode("utf-8")[:32]] or [1]
        in_arr = np.array([raw_ids], dtype=int)
        out_arr = self.sinh_tiep(in_arr, so_token=min(do_dai, 16), nhiet_do=nhiet_do, top_k=10)
        new_tokens = out_arr[0, len(raw_ids) :].tolist()
        suffix = "".join(chr(32 + (t % 90)) for t in new_tokens).strip()
        return f"{prompt} {suffix}".strip()

    def danh_gia(self, X: Any, y: Optional[Any] = None) -> float:
        """Đánh giá mô hình (trả về điểm tương ứng 1 / (1 + loss))."""
        if isinstance(X, np.ndarray):
            in_ids = X.astype(int)
            tgt_ids = y.astype(int) if y is not None else np.roll(in_ids, -1, axis=-1)
            loss = self.tinh_loss(in_ids, tgt_ids)
            return round(1.0 / (1.0 + max(0.0, loss)), 4)
        return 1.0

    def luu(self, duong_dan: str) -> str:
        """Lưu toàn bộ mô hình GPTModel (kèm LoRA nếu có) ra file pickle."""
        import pickle
        from pathlib import Path

        p = Path(duong_dan)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        return str(p)

    @classmethod
    def tai(cls, duong_dan: str) -> "GPTModel":
        """Tải mô hình GPTModel từ file đã lưu."""
        import pickle

        with open(duong_dan, "rb") as f:
            return pickle.load(f)
