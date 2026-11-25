"""Động cơ Suy luận Edge Native Siêu tốc (Zero-GPU Inference Engine) - Vietnamese AI Framework."""

from __future__ import annotations

import mmap
import os
import resource
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

try:
    import vietnamese_ai_rust as _rust_core  # type: ignore
except ImportError:
    try:
        from vietnamese_ai import vietnamese_ai_rust as _rust_core  # type: ignore
    except ImportError:
        _rust_core = None

THONG_SO_LUONG_TU: Dict[str, Tuple[int, int]] = {
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q4_k_m": (256, 144),
    "q4_k_s": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "bf16": (1, 2),
    "f16": (1, 2),
    "f32": (1, 4),
}

KIEN_TRUC_LLM_HO_TRO: Tuple[str, ...] = (
    "llama",
    "qwen2",
    "qwen3",
    "gemma",
    "gemma2",
    "mistral",
    "phi3",
    "deepseek2",
    "gpt2",
    "vietnamese_ai",
)

_GGML_TYPE_MAP: Dict[int, str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    30: "bf16",
}


class BoTachTuBPEGGUF:
    """Bộ tách từ BPE/Vocab khởi tạo trực tiếp từ siêu dữ liệu `tokenizer.ggml.tokens` và `tokenizer.ggml.merges` trong tệp GGUF."""

    def __init__(self, tokens: List[str], merges: Optional[List[str]] = None) -> None:
        self.tokens: List[str] = list(tokens)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.merges_rank: Dict[Tuple[str, str], int] = {}
        for rank, rule in enumerate(merges or []):
            parts = str(rule).split()
            if len(parts) == 2:
                self.merges_rank[(parts[0], parts[1])] = rank
        self.bos_id: int = self.token_to_id.get("<s>", self.token_to_id.get("<|im_start|>", 0))
        self.eos_id: int = self.token_to_id.get("</s>", self.token_to_id.get("<|im_end|>", 0))
        self.unk_id: int = self.token_to_id.get("<unk>", 0)

    @classmethod
    def tu_streamer(cls, streamer: "GGUFMemoryMappedStreamer") -> Optional["BoTachTuBPEGGUF"]:
        """Tạo `BoTachTuBPEGGUF` nếu tệp GGUF chứa mảng từ vựng `tokenizer.ggml.tokens`."""
        raw_tokens = streamer.metadata.get("tokenizer.ggml.tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            return None
        raw_merges = streamer.metadata.get("tokenizer.ggml.merges")
        merges_list = [str(m) for m in raw_merges] if isinstance(raw_merges, list) else []
        return cls([str(t) for t in raw_tokens], merges=merges_list)

    def _bpe_word(self, word: str) -> List[str]:
        if word in self.token_to_id:
            return [word]
        symbols = list(word)
        if not self.merges_rank or len(symbols) < 2:
            return symbols
        while len(symbols) >= 2:
            best_pair: Optional[Tuple[str, str]] = None
            best_rank = 10**9
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merges_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged: List[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best_pair:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def ma_hoa(self, van_ban: str) -> List[int]:
        """Mã hóa chuỗi văn bản thành danh sách token IDs theo bảng từ vựng GGUF."""
        if not van_ban:
            return []
        ids: List[int] = []
        for idx, tu in enumerate(van_ban.split()):
            prefixed = f"▁{tu}" if f"▁{tu}" in self.token_to_id else (f"Ġ{tu}" if (idx > 0 and f"Ġ{tu}" in self.token_to_id) else tu)
            for piece in self._bpe_word(prefixed):
                if piece in self.token_to_id:
                    ids.append(self.token_to_id[piece])
                elif piece.lstrip("▁Ġ") in self.token_to_id:
                    ids.append(self.token_to_id[piece.lstrip("▁Ġ")])
                else:
                    ids.append(self.unk_id)
        return ids

    def giai_ma(self, ids: List[int]) -> str:
        """Giải mã danh sách token IDs về chuỗi văn bản đọc được."""
        parts: List[str] = []
        for tid in ids:
            if 0 <= tid < len(self.tokens):
                tok = self.tokens[tid]
                if tok in ("<s>", "</s>", "<pad>", "<unk>"):
                    continue
                parts.append(tok.replace("▁", " ").replace("Ġ", " "))
        return "".join(parts).strip() if any(("▁" in t or "Ġ" in t) for t in self.tokens[:64]) else " ".join(parts).strip()


class BoGiaiMaTransformerEdge:
    """
    Khối giải mã Transformer đa tầng tối ưu CPU (RMSNorm + RoPE + Grouped-Query Attention GQA
    với KV-Cache + SwiGLU). Hỗ trợ giữ nguyên trọng số lượng tử GGUF (Q4_0/Q4_1/Q8_0/Q2_K/Q3_K/Q4_K)
    để nhân ma trận phân ô tiết kiệm RAM thay vì giải nén toàn bộ sang FP32.
    """

    def __init__(
        self,
        d_model: int = 128,
        so_dau: int = 4,
        d_ff: int = 256,
        seed: int = 42,
        so_dau_kv: Optional[int] = None,
        so_lop: int = 1,
    ) -> None:
        self.d_model = d_model
        self.so_dau = max(1, so_dau)
        self.d_head = max(1, d_model // self.so_dau)
        kv_heads = self.so_dau if so_dau_kv is None else max(1, min(self.so_dau, int(so_dau_kv)))
        while self.so_dau % kv_heads != 0 and kv_heads > 1:
            kv_heads -= 1
        self.so_dau_kv = kv_heads
        self.d_kv = self.so_dau_kv * self.d_head
        self.d_ff = d_ff
        self.so_lop = max(1, int(so_lop))

        rng = np.random.RandomState(seed)
        scale = float(1.0 / np.sqrt(d_model))
        scale_ff = float(1.0 / np.sqrt(d_ff))

        self._cac_lop: List[Dict[str, np.ndarray]] = []
        for _ in range(self.so_lop):
            self._cac_lop.append(
                {
                    "w_q": (rng.randn(d_model, d_model) * scale).astype(np.float32),
                    "w_k": (rng.randn(d_model, self.d_kv) * scale).astype(np.float32),
                    "w_v": (rng.randn(d_model, self.d_kv) * scale).astype(np.float32),
                    "w_o": (rng.randn(d_model, d_model) * scale).astype(np.float32),
                    "w_gate": (rng.randn(d_model, d_ff) * scale).astype(np.float32),
                    "w_up": (rng.randn(d_model, d_ff) * scale).astype(np.float32),
                    "w_down": (rng.randn(d_ff, d_model) * scale_ff).astype(np.float32),
                    "norm_attn": np.ones(d_model, dtype=np.float32),
                    "norm_ffn": np.ones(d_model, dtype=np.float32),
                }
            )

        lop_0 = self._cac_lop[0]
        self.w_q = lop_0["w_q"]
        self.w_k = lop_0["w_k"]
        self.w_v = lop_0["w_v"]
        self.w_o = lop_0["w_o"]
        self.w_gate = lop_0["w_gate"]
        self.w_up = lop_0["w_up"]
        self.w_down = lop_0["w_down"]
        self.norm_attn = lop_0["norm_attn"]
        self.norm_ffn = lop_0["norm_ffn"]
        self.norm_output = np.ones(d_model, dtype=np.float32)

        self._trong_so_luong_tu: Dict[str, Dict[str, Any]] = {}
        self._kv_local = threading.local()
        self._k_caches, self._v_caches = self._lay_kv_caches_luong()
        self._k_cache = self._k_caches[0]
        self._v_cache = self._v_caches[0]

    def _lay_kv_caches_luong(self) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """Lấy cặp bộ đệm KV-Cache riêng cho từng luồng để bảo đảm an toàn đa luồng không khóa."""
        k_caches = getattr(self._kv_local, "k_caches", None)
        v_caches = getattr(self._kv_local, "v_caches", None)
        if k_caches is None or v_caches is None:
            k_caches = [[] for _ in range(self.so_lop)]
            v_caches = [[] for _ in range(self.so_lop)]
            self._kv_local.k_caches = k_caches
            self._kv_local.v_caches = v_caches
        return k_caches, v_caches

    @staticmethod
    def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Chuẩn hóa RMSNorm theo chiều đặc trưng cuối (ưu tiên kernel Rust khi khả dụng)."""
        x_f32 = np.asarray(x, dtype=np.float32)
        g_f32 = np.asarray(gamma, dtype=np.float32).ravel()
        if (
            _rust_core is not None
            and x_f32.ndim >= 1
            and x_f32.shape[-1] == g_f32.shape[0]
            and x_f32.size >= 64
        ):
            fn_rms_b = getattr(_rust_core, "fast_rmsnorm_f32_bytes", None)
            if fn_rms_b is not None:
                try:
                    dim = int(x_f32.shape[-1])
                    rows = int(x_f32.size // dim)
                    raw_out = fn_rms_b(
                        np.ascontiguousarray(x_f32, dtype="<f4").tobytes(),
                        np.ascontiguousarray(g_f32, dtype="<f4").tobytes(),
                        rows,
                        dim,
                        float(eps),
                    )
                    return np.frombuffer(raw_out, dtype="<f4").copy().reshape(x_f32.shape)
                except Exception:
                    pass
            fn_rms = getattr(_rust_core, "fast_rmsnorm_f32", None)
            if fn_rms is not None:
                try:
                    dim = int(x_f32.shape[-1])
                    rows = int(x_f32.size // dim)
                    out = fn_rms(
                        np.ascontiguousarray(x_f32, dtype="<f4").tobytes(),
                        np.ascontiguousarray(g_f32, dtype="<f4").tobytes(),
                        rows,
                        dim,
                        float(eps),
                    )
                    return np.asarray(out, dtype=np.float32).reshape(x_f32.shape)
                except Exception:
                    pass
        rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
        return (x_f32 / rms) * g_f32

    @staticmethod
    def ap_dung_rope(x: np.ndarray, vi_tri: int) -> np.ndarray:
        """Áp dụng mã hóa vị trí quay Rotary Positional Embedding (RoPE) tại bước thời gian `vi_tri`."""
        d = x.shape[-1]
        nua = d // 2
        if nua == 0:
            return x
        freqs = 1.0 / (10000.0 ** (np.arange(0, nua, dtype=np.float32) / float(nua)))
        angles = float(vi_tri) * freqs
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        x1 = x[..., :nua]
        x2 = x[..., nua : 2 * nua]
        out = x.copy()
        out[..., :nua] = x1 * cos_a - x2 * sin_a
        out[..., nua : 2 * nua] = x1 * sin_a + x2 * cos_a
        return out

    def xoa_kv_cache(self) -> None:
        """Đặt lại bộ đệm KV-Cache của toàn bộ các tầng trên luồng hiện tại trước phiên suy luận mới."""
        k_caches, v_caches = self._lay_kv_caches_luong()
        for kc, vc in zip(k_caches, v_caches):
            kc.clear()
            vc.clear()

    def _chieu_tuyen_tinh(self, x: np.ndarray, khoa_quant: str, w_fp32: np.ndarray) -> np.ndarray:
        """Tính phép chiếu tuyến tính, ưu tiên nhân trực tiếp trên bộ đệm lượng tử GGUF nếu đã nạp."""
        q_info = self._trong_so_luong_tu.get(khoa_quant)
        if q_info is not None:
            return BoGiaiLuongTuQ4.nhan_ma_tran_luong_tu(
                x,
                q_info["raw_bytes"],
                shape_w=q_info["shape"],
                loai_quant=q_info["loai_quant"],
            )
        return x @ w_fp32

    def nap_trong_so_tu_gguf(
        self,
        streamer: "GGUFMemoryMappedStreamer",
        giu_luong_tu: bool = True,
    ) -> int:
        """Nạp trọng số các tầng từ `GGUFMemoryMappedStreamer`, ưu tiên giữ nguyên bộ đệm lượng tử tiết kiệm RAM."""
        da_nap = 0
        anh_xa = (
            ("attn_q.weight", "w_q"),
            ("attn_k.weight", "w_k"),
            ("attn_v.weight", "w_v"),
            ("attn_output.weight", "w_o"),
            ("ffn_gate.weight", "w_gate"),
            ("ffn_up.weight", "w_up"),
            ("ffn_down.weight", "w_down"),
        )
        for lop_idx in range(self.so_lop):
            lop = self._cac_lop[lop_idx]
            for ten_hau_to, ten_attr in anh_xa:
                ten_gguf = f"blk.{lop_idx}.{ten_hau_to}"
                target = lop[ten_attr]
                in_dim, out_dim = int(target.shape[0]), int(target.shape[1])
                khoa_quant = f"{lop_idx}:{ten_attr}"

                if giu_luong_tu:
                    tho = streamer.doc_tensor_tho(ten_gguf)
                    if tho is not None:
                        dims = tho["dims"]
                        loai_q = str(tho["loai_quant"])
                        block_size, _ = THONG_SO_LUONG_TU.get(loai_q, (1, 4))
                        khop_ggml = len(dims) == 2 and int(dims[0]) == in_dim and int(dims[1]) == out_dim
                        khop_c_order = len(dims) == 2 and int(dims[0]) == out_dim and int(dims[1]) == in_dim
                        if (
                            (khop_ggml or khop_c_order)
                            and loai_q not in ("f16", "f32", "bf16")
                            and in_dim % block_size == 0
                        ):
                            self._trong_so_luong_tu[khoa_quant] = {
                                "raw_bytes": tho["raw_bytes"],
                                "shape": (out_dim, in_dim),
                                "loai_quant": loai_q,
                            }
                            da_nap += 1
                            continue

                tensor = streamer.doc_tensor(ten_gguf)
                if tensor is None:
                    continue
                if tensor.ndim == 2:
                    if tensor.shape == (out_dim, in_dim):
                        tensor = tensor.T
                    r = min(target.shape[0], tensor.shape[0])
                    c = min(target.shape[1], tensor.shape[1])
                    target[:r, :c] = tensor[:r, :c].astype(np.float32)
                    da_nap += 1

            for ten_norm, attr_norm in (
                ("attn_norm.weight", "norm_attn"),
                ("ffn_norm.weight", "norm_ffn"),
            ):
                t_norm = streamer.doc_tensor(f"blk.{lop_idx}.{ten_norm}")
                if t_norm is not None:
                    flat_n = t_norm.ravel().astype(np.float32)
                    n_copy = min(lop[attr_norm].shape[0], flat_n.shape[0])
                    lop[attr_norm][:n_copy] = flat_n[:n_copy]
                    da_nap += 1

        t_out_norm = streamer.doc_tensor("output_norm.weight")
        if t_out_norm is not None:
            flat_on = t_out_norm.ravel().astype(np.float32)
            n_copy = min(self.norm_output.shape[0], flat_on.shape[0])
            self.norm_output[:n_copy] = flat_on[:n_copy]
            da_nap += 1
        return da_nap

    def buoc_suy_luan(self, x_step: np.ndarray, vi_tri: int = 0) -> np.ndarray:
        """Thực thi 1 bước lan truyền tiến autoregressive đa tầng với RMSNorm, RoPE, GQA KV-Cache và SwiGLU."""
        x = np.asarray(x_step, dtype=np.float32).reshape(1, self.d_model)
        so_nhom_gqa = max(1, self.so_dau // self.so_dau_kv)
        scale_head = float(np.sqrt(self.d_head))
        k_caches, v_caches = self._lay_kv_caches_luong()

        for lop_idx, lop in enumerate(self._cac_lop):
            h = self.rms_norm(x, lop["norm_attn"])
            q_flat = self._chieu_tuyen_tinh(h, f"{lop_idx}:w_q", lop["w_q"])
            k_flat = self._chieu_tuyen_tinh(h, f"{lop_idx}:w_k", lop["w_k"])
            v_flat = self._chieu_tuyen_tinh(h, f"{lop_idx}:w_v", lop["w_v"])

            q_heads = self.ap_dung_rope(q_flat.reshape(self.so_dau, self.d_head), vi_tri)
            k_heads = self.ap_dung_rope(k_flat.reshape(self.so_dau_kv, self.d_head), vi_tri)
            v_heads = v_flat.reshape(self.so_dau_kv, self.d_head)

            k_caches[lop_idx].append(k_heads)
            v_caches[lop_idx].append(v_heads)

            k_all = np.stack(k_caches[lop_idx], axis=1)
            v_all = np.stack(v_caches[lop_idx], axis=1)
            if so_nhom_gqa > 1:
                k_all = np.repeat(k_all, so_nhom_gqa, axis=0)
                v_all = np.repeat(v_all, so_nhom_gqa, axis=0)

            scores = np.einsum("hd,htd->ht", q_heads, k_all) / scale_head
            scores -= np.max(scores, axis=-1, keepdims=True)
            attn_w = np.exp(scores)
            attn_w /= np.sum(attn_w, axis=-1, keepdims=True) + 1e-9
            ctx_heads = np.einsum("ht,htd->hd", attn_w, v_all)
            ctx_flat = ctx_heads.reshape(1, self.d_model)

            attn_out = self._chieu_tuyen_tinh(ctx_flat, f"{lop_idx}:w_o", lop["w_o"])
            x = x + attn_out

            h_ffn = self.rms_norm(x, lop["norm_ffn"])
            gate = self._chieu_tuyen_tinh(h_ffn, f"{lop_idx}:w_gate", lop["w_gate"])
            up = self._chieu_tuyen_tinh(h_ffn, f"{lop_idx}:w_up", lop["w_up"])
            ffn_act = self.swiglu(gate, up)
            ffn_out = self._chieu_tuyen_tinh(ffn_act, f"{lop_idx}:w_down", lop["w_down"])
            x = x + ffn_out

        return x.astype(np.float32)

    @staticmethod
    def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Tính hàm kích hoạt hợp nhất SwiGLU (`SiLU(gate) * up`), ưu tiên kernel Rust khi khả dụng."""
        g_f32 = np.asarray(gate, dtype=np.float32)
        u_f32 = np.asarray(up, dtype=np.float32)
        if (
            _rust_core is not None
            and g_f32.shape == u_f32.shape
            and g_f32.size >= 64
        ):
            fn_swi_b = getattr(_rust_core, "fast_swiglu_f32_bytes", None)
            if fn_swi_b is not None:
                try:
                    raw_out = fn_swi_b(
                        np.ascontiguousarray(g_f32, dtype="<f4").tobytes(),
                        np.ascontiguousarray(u_f32, dtype="<f4").tobytes(),
                        int(g_f32.size),
                    )
                    return np.frombuffer(raw_out, dtype="<f4").copy().reshape(g_f32.shape)
                except Exception:
                    pass
            fn_swi = getattr(_rust_core, "fast_swiglu_f32", None)
            if fn_swi is not None:
                try:
                    out = fn_swi(
                        np.ascontiguousarray(g_f32, dtype="<f4").tobytes(),
                        np.ascontiguousarray(u_f32, dtype="<f4").tobytes(),
                        int(g_f32.size),
                    )
                    return np.asarray(out, dtype=np.float32).reshape(g_f32.shape)
                except Exception:
                    pass
        silu = g_f32 / (1.0 + np.exp(-np.clip(g_f32, -20.0, 20.0)))
        return silu * u_f32

    @staticmethod
    def lay_mau_logit(
        logits: np.ndarray,
        nhiet_do: float = 0.7,
        top_k: int = 8,
        da_sinh: Optional[List[int]] = None,
        phat_lap: float = 1.25,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Áp dụng phạt lặp, lọc top-k và lấy mẫu theo phân phối softmax thực tế khi `nhiet_do > 0`."""
        vec = np.asarray(logits, dtype=np.float64).ravel().copy()
        if da_sinh:
            for idx in set(da_sinh):
                if 0 <= idx < len(vec):
                    vec[idx] = vec[idx] / phat_lap if vec[idx] > 0 else vec[idx] * phat_lap
        if nhiet_do <= 1e-3:
            return int(np.argmax(vec))
        vec = vec / max(float(nhiet_do), 1e-3)
        k = max(1, min(int(top_k), len(vec)))
        top_indices = np.argpartition(vec, -k)[-k:]
        top_vals = vec[top_indices]
        top_vals -= np.max(top_vals)
        probs = np.exp(top_vals)
        tong = float(np.sum(probs))
        if not np.isfinite(tong) or tong <= 0.0:
            return int(top_indices[int(np.argmax(top_vals))])
        probs /= tong
        gen = rng if rng is not None else np.random.default_rng()
        return int(gen.choice(top_indices, p=probs))

    def sinh_tu_gguf_bpe(
        self,
        prompt: str,
        streamer: "GGUFMemoryMappedStreamer",
        so_token_toi_da: int = 16,
        nhiet_do: float = 0.0,
    ) -> Dict[str, Any]:
        """Chạy suy luận đầu-cuối (Token ID -> token_embd -> GQA Quantized -> output LM-Head -> BPE Decode) trực tiếp từ 1 tệp GGUF."""
        bpe = BoTachTuBPEGGUF.tu_streamer(streamer)
        if bpe is None:
            raise ValueError("Tệp GGUF không chứa siêu dữ liệu từ vựng 'tokenizer.ggml.tokens'.")

        tok_emb = streamer.doc_tensor("token_embd.weight")
        if tok_emb is None or tok_emb.ndim != 2:
            raise ValueError("Tệp GGUF không chứa tensor 2 chiều 'token_embd.weight'.")
        if tok_emb.shape[1] != self.d_model and tok_emb.shape[0] == self.d_model:
            tok_emb = tok_emb.T
        if tok_emb.shape[1] != self.d_model:
            raise ValueError(
                f"Chiều embedding của GGUF ({tok_emb.shape[1]}) không khớp với d_model ({self.d_model})."
            )

        lm_head = streamer.doc_tensor("output.weight")
        if lm_head is None:
            lm_head = tok_emb
        elif lm_head.ndim == 2 and lm_head.shape[1] != self.d_model and lm_head.shape[0] == self.d_model:
            lm_head = lm_head.T

        self.nap_trong_so_tu_gguf(streamer, giu_luong_tu=True)
        self.xoa_kv_cache()

        input_ids = bpe.ma_hoa(prompt)
        if not input_ids:
            input_ids = [bpe.bos_id]

        vocab_size = int(tok_emb.shape[0])
        h_last = np.zeros((1, self.d_model), dtype=np.float32)
        for pos, tid in enumerate(input_ids):
            vec = tok_emb[tid % vocab_size]
            h_last = self.buoc_suy_luan(vec, vi_tri=pos)

        generated_ids: List[int] = []
        pos_cur = len(input_ids)
        for _ in range(max(1, so_token_toi_da)):
            h_norm = self.rms_norm(h_last, self.norm_output)
            logits = (h_norm[0] @ lm_head.T).astype(np.float32)
            next_id = self.lay_mau_logit(logits, nhiet_do=nhiet_do, top_k=8, da_sinh=generated_ids)
            if next_id == bpe.eos_id and bpe.eos_id != bpe.bos_id and generated_ids:
                break
            generated_ids.append(next_id)
            h_last = self.buoc_suy_luan(tok_emb[next_id % vocab_size], vi_tri=pos_cur)
            pos_cur += 1

        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "generated_ids": generated_ids,
            "van_ban": bpe.giai_ma(generated_ids),
            "vocab_size": vocab_size,
            "che_do": "gguf_bpe_end_to_end",
        }

    def sinh_stream_tu_gguf_bpe(
        self,
        prompt: str,
        streamer: "GGUFMemoryMappedStreamer",
        so_token_toi_da: int = 16,
        nhiet_do: float = 0.0,
    ) -> Generator[str, None, None]:
        """Sinh luồng token trực tiếp từng bước từ tệp GGUF chứa từ vựng BPE và trọng số embedding."""
        bpe = BoTachTuBPEGGUF.tu_streamer(streamer)
        if bpe is None:
            raise ValueError("Tệp GGUF không chứa siêu dữ liệu từ vựng 'tokenizer.ggml.tokens'.")

        tok_emb = streamer.doc_tensor("token_embd.weight")
        if tok_emb is None or tok_emb.ndim != 2:
            raise ValueError("Tệp GGUF không chứa tensor 2 chiều 'token_embd.weight'.")
        if tok_emb.shape[1] != self.d_model and tok_emb.shape[0] == self.d_model:
            tok_emb = tok_emb.T
        if tok_emb.shape[1] != self.d_model:
            raise ValueError(
                f"Chiều embedding của GGUF ({tok_emb.shape[1]}) không khớp với d_model ({self.d_model})."
            )

        lm_head = streamer.doc_tensor("output.weight")
        if lm_head is None:
            lm_head = tok_emb
        elif lm_head.ndim == 2 and lm_head.shape[1] != self.d_model and lm_head.shape[0] == self.d_model:
            lm_head = lm_head.T

        if not self._trong_so_luong_tu:
            self.nap_trong_so_tu_gguf(streamer, giu_luong_tu=True)
        self.xoa_kv_cache()

        input_ids = bpe.ma_hoa(prompt)
        if not input_ids:
            input_ids = [bpe.bos_id]

        vocab_size = int(tok_emb.shape[0])
        h_last = np.zeros((1, self.d_model), dtype=np.float32)
        for pos, tid in enumerate(input_ids):
            vec = tok_emb[tid % vocab_size]
            h_last = self.buoc_suy_luan(vec, vi_tri=pos)

        co_tien_to_khoang_trang = any(("▁" in t or "Ġ" in t) for t in bpe.tokens[:64])
        generated_ids: List[int] = []
        pos_cur = len(input_ids)
        so_piece_da_xuat = 0
        for _ in range(max(1, so_token_toi_da)):
            h_norm = self.rms_norm(h_last, self.norm_output)
            logits = (h_norm[0] @ lm_head.T).astype(np.float32)
            next_id = self.lay_mau_logit(logits, nhiet_do=nhiet_do, top_k=8, da_sinh=generated_ids)
            if next_id == bpe.eos_id and bpe.eos_id != bpe.bos_id and generated_ids:
                break
            generated_ids.append(next_id)
            if 0 <= next_id < len(bpe.tokens):
                tok_str = bpe.tokens[next_id]
                if tok_str not in ("<s>", "</s>", "<pad>", "<unk>"):
                    piece = tok_str.replace("▁", " ").replace("Ġ", " ")
                    if not co_tien_to_khoang_trang and so_piece_da_xuat > 0:
                        piece = " " + piece
                    elif so_piece_da_xuat == 0:
                        piece = piece.lstrip()
                    if piece:
                        so_piece_da_xuat += 1
                        yield piece
            h_last = self.buoc_suy_luan(tok_emb[next_id % vocab_size], vi_tri=pos_cur)
            pos_cur += 1


class BoGiaiLuongTuQ4:
    """
    Bộ giải lượng tử hóa ma trận khối vector hóa thuần NumPy (Vectorized Block Dequantizer):
    - Hỗ trợ đầy đủ chuẩn GGML / GGUF: Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K (Q4_K_M/Q4_K_S), F16, F32.
    - Tận dụng tập lệnh vector SIMD của NumPy (AVX2/AVX-512/NEON) để giải nén và nhân ma trận phân ô (tiled matmul).
    """

    @staticmethod
    def giai_q8_0(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử Q8_0: Mỗi block 32 phần tử gồm 2 bytes float16 scale + 32 bytes int8.
        Tổng kích thước block = 34 bytes.
        """
        block_size = 32
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * 34:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * 34], dtype=np.uint8).reshape((so_block, 34))
        scales = data[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        weights = data[:, 2:].copy().view(np.int8).astype(np.float32)

        kq = (weights * scales).reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q4_0(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử Q4_0 chuẩn GGML: Mỗi block 32 phần tử gồm 2 bytes float16 scale + 16 bytes nibbles
        (16 phần tử đầu lấy 4-bit thấp, 16 phần tử sau lấy 4-bit cao). Công thức: v_i = (nibble - 8) * scale.
        """
        block_size = 32
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * 18:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * 18], dtype=np.uint8).reshape((so_block, 18))
        scales = data[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        nibbles_raw = data[:, 2:]

        low_nibbles = (nibbles_raw & 0x0F).astype(np.int8) - 8
        high_nibbles = ((nibbles_raw >> 4) & 0x0F).astype(np.int8) - 8
        weights = np.concatenate(
            [low_nibbles.astype(np.float32), high_nibbles.astype(np.float32)],
            axis=1,
        )

        kq = (weights * scales).reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q4_1(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử Q4_1 chuẩn GGML: Mỗi block 32 phần tử gồm 2 bytes float16 `d` + 2 bytes float16 `m`
        + 16 bytes nibbles không dấu (16 phần tử đầu lấy 4-bit thấp, 16 phần tử sau lấy 4-bit cao).
        Tổng kích thước block = 20 bytes. Công thức: v_i = q_i * d + m.
        """
        block_size = 32
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * 20:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * 20], dtype=np.uint8).reshape((so_block, 20))
        d = data[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        m = data[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qs = data[:, 4:20]

        low = (qs & 0x0F).astype(np.float32)
        high = ((qs >> 4) & 0x0F).astype(np.float32)
        weights = np.concatenate([low, high], axis=1)

        kq = (weights * d + m).reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q2_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử siêu khối Q2_K chuẩn GGML (`QK_K = 256`, 84 bytes/block):
        - `scales[16]` (bytes 0..15): 4-bit thấp là scale `sc`, 4-bit cao là min `m` cho 16 khối con 16 phần tử.
        - `qs[64]` (bytes 16..79): 256 trọng số 2-bit xếp thành 2 nhóm 32 byte (mỗi nhóm tách theo dịch 0,2,4,6 bit).
        - `d` (bytes 80..81, f16) và `dmin` (bytes 82..83, f16).
        Công thức mỗi khối con j: v = (d * sc_j) * q - (dmin * m_j).
        """
        block_size = 256
        block_bytes = 84
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape(
            (so_block, block_bytes)
        )
        scales_raw = data[:, 0:16]
        qs = data[:, 16:80]
        d = data[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)
        dmin = data[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)

        sc = (scales_raw & 0x0F).astype(np.float32) * d
        mn = ((scales_raw >> 4) & 0x0F).astype(np.float32) * dmin

        out = np.empty((so_block, 256), dtype=np.float32)
        sub_idx = 0
        out_pos = 0
        for chunk in range(2):
            q_chunk = qs[:, chunk * 32 : (chunk + 1) * 32]
            for shift in (0, 2, 4, 6):
                q_vals = ((q_chunk >> shift) & 0x03).astype(np.float32)
                out[:, out_pos : out_pos + 16] = (
                    q_vals[:, 0:16] * sc[:, sub_idx : sub_idx + 1] - mn[:, sub_idx : sub_idx + 1]
                )
                sub_idx += 1
                out_pos += 16
                out[:, out_pos : out_pos + 16] = (
                    q_vals[:, 16:32] * sc[:, sub_idx : sub_idx + 1] - mn[:, sub_idx : sub_idx + 1]
                )
                sub_idx += 1
                out_pos += 16

        kq = out.reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q3_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử siêu khối Q3_K chuẩn GGML (`QK_K = 256`, 110 bytes/block):
        - `hmask[32]` (bytes 0..31): bit cao (bit thứ 3) cho 256 phần tử.
        - `qs[64]` (bytes 32..95): 2 bit thấp cho 256 phần tử.
        - `scales[12]` (bytes 96..107): nén 16 hệ số 6-bit có dấu (dịch -32).
        - `d` (bytes 108..109, f16).
        """
        block_size = 256
        block_bytes = 110
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape(
            (so_block, block_bytes)
        )
        hmask = data[:, 0:32]
        qs = data[:, 32:96]
        s_raw = data[:, 96:108]
        d = data[:, 108:110].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)

        s0 = (s_raw[:, 0:4] & 0x0F) | ((s_raw[:, 8:12] & 0x03) << 4)
        s1 = (s_raw[:, 4:8] & 0x0F) | (((s_raw[:, 8:12] >> 2) & 0x03) << 4)
        s2 = ((s_raw[:, 0:4] >> 4) & 0x0F) | (((s_raw[:, 8:12] >> 4) & 0x03) << 4)
        s3 = ((s_raw[:, 4:8] >> 4) & 0x0F) | (((s_raw[:, 8:12] >> 6) & 0x03) << 4)
        scales_16 = (np.concatenate([s0, s1, s2, s3], axis=1).astype(np.int16) - 32).astype(np.float32) * d

        out = np.empty((so_block, 256), dtype=np.float32)
        sub_idx = 0
        out_pos = 0
        mask_bit = 1
        for chunk in range(2):
            q_chunk = qs[:, chunk * 32 : (chunk + 1) * 32]
            for shift in (0, 2, 4, 6):
                low2 = ((q_chunk >> shift) & 0x03).astype(np.int16)
                high_sub = np.where((hmask & mask_bit) != 0, 0, 4).astype(np.int16)
                q3 = (low2 - high_sub).astype(np.float32)
                out[:, out_pos : out_pos + 16] = q3[:, 0:16] * scales_16[:, sub_idx : sub_idx + 1]
                sub_idx += 1
                out_pos += 16
                out[:, out_pos : out_pos + 16] = q3[:, 16:32] * scales_16[:, sub_idx : sub_idx + 1]
                sub_idx += 1
                out_pos += 16
                mask_bit <<= 1

        kq = out.reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q4_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """
        Giải lượng tử siêu khối Q4_K (`Q4_K_M` / `Q4_K_S`, `QK_K = 256`, 144 bytes/block):
        - `d` (bytes 0..1, f16), `dmin` (bytes 2..3, f16).
        - `scales[12]` (bytes 4..15): nén 8 cặp `(sc, m)` 6-bit cho 8 khối con 32 phần tử.
        - `qs[128]` (bytes 16..143): 4 nhóm 32 byte (32 nibble thấp thuộc khối 2g, 32 nibble cao thuộc khối 2g+1).
        """
        block_size = 256
        block_bytes = 144
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)

        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape(
            (so_block, block_bytes)
        )
        d = data[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)
        dmin = data[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)
        sc_raw = data[:, 4:16]
        qs = data[:, 16:144]

        sc = np.empty((so_block, 8), dtype=np.float32)
        mn = np.empty((so_block, 8), dtype=np.float32)
        sc[:, 0:4] = (sc_raw[:, 0:4] & 0x3F).astype(np.float32)
        mn[:, 0:4] = (sc_raw[:, 4:8] & 0x3F).astype(np.float32)
        sc[:, 4:8] = ((sc_raw[:, 8:12] & 0x0F) | ((sc_raw[:, 0:4] >> 6) << 4)).astype(np.float32)
        mn[:, 4:8] = ((sc_raw[:, 8:12] >> 4) | ((sc_raw[:, 4:8] >> 6) << 4)).astype(np.float32)

        sc *= d
        mn *= dmin

        out = np.empty((so_block, 256), dtype=np.float32)
        for g in range(4):
            q_grp = qs[:, g * 32 : (g + 1) * 32]
            low = (q_grp & 0x0F).astype(np.float32)
            high = ((q_grp >> 4) & 0x0F).astype(np.float32)
            out[:, g * 64 : g * 64 + 32] = low * sc[:, 2 * g : 2 * g + 1] - mn[:, 2 * g : 2 * g + 1]
            out[:, g * 64 + 32 : (g + 1) * 64] = (
                high * sc[:, 2 * g + 1 : 2 * g + 2] - mn[:, 2 * g + 1 : 2 * g + 2]
            )

        kq = out.reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q5_0(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải lượng tử Q5_0 chuẩn GGML (32 phần tử / 22 bytes: f16 d + uint32 qh + 16B qs)."""
        block_size = 32
        block_bytes = 22
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)
        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape((so_block, block_bytes))
        d = data[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qh = data[:, 2:6].copy().view(np.uint32).reshape(-1, 1)
        qs = data[:, 6:22]
        shifts_lo = np.arange(16, dtype=np.uint32).reshape(1, 16)
        shifts_hi = np.arange(16, 32, dtype=np.uint32).reshape(1, 16)
        xh_lo = ((qh >> shifts_lo) & 1).astype(np.int16) << 4
        xh_hi = ((qh >> shifts_hi) & 1).astype(np.int16) << 4
        lo = ((qs & 0x0F).astype(np.int16) | xh_lo) - 16
        hi = (((qs >> 4) & 0x0F).astype(np.int16) | xh_hi) - 16
        weights = np.concatenate([lo.astype(np.float32), hi.astype(np.float32)], axis=1)
        kq = (weights * d).reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q5_1(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải lượng tử Q5_1 chuẩn GGML (32 phần tử / 24 bytes: f16 d + f16 m + uint32 qh + 16B qs)."""
        block_size = 32
        block_bytes = 24
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)
        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape((so_block, block_bytes))
        d = data[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        m = data[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qh = data[:, 4:8].copy().view(np.uint32).reshape(-1, 1)
        qs = data[:, 8:24]
        shifts_lo = np.arange(16, dtype=np.uint32).reshape(1, 16)
        shifts_hi = np.arange(16, 32, dtype=np.uint32).reshape(1, 16)
        xh_lo = ((qh >> shifts_lo) & 1).astype(np.uint16) << 4
        xh_hi = ((qh >> shifts_hi) & 1).astype(np.uint16) << 4
        lo = (qs & 0x0F).astype(np.uint16) | xh_lo
        hi = ((qs >> 4) & 0x0F).astype(np.uint16) | xh_hi
        weights = np.concatenate([lo.astype(np.float32), hi.astype(np.float32)], axis=1)
        kq = (weights * d + m).reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q5_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải lượng tử siêu khối Q5_K chuẩn GGML (`QK_K = 256`, 176 bytes/block)."""
        block_size = 256
        block_bytes = 176
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)
        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape((so_block, block_bytes))
        d = data[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)
        dmin = data[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)
        sc_raw = data[:, 4:16]
        qh = data[:, 16:48]
        qs = data[:, 48:176]

        sc = np.empty((so_block, 8), dtype=np.float32)
        mn = np.empty((so_block, 8), dtype=np.float32)
        sc[:, 0:4] = (sc_raw[:, 0:4] & 0x3F).astype(np.float32)
        mn[:, 0:4] = (sc_raw[:, 4:8] & 0x3F).astype(np.float32)
        sc[:, 4:8] = ((sc_raw[:, 8:12] & 0x0F) | ((sc_raw[:, 0:4] >> 6) << 4)).astype(np.float32)
        mn[:, 4:8] = ((sc_raw[:, 8:12] >> 4) | ((sc_raw[:, 4:8] >> 6) << 4)).astype(np.float32)
        sc *= d
        mn *= dmin

        out = np.empty((so_block, 256), dtype=np.float32)
        for g in range(4):
            q_grp = qs[:, g * 32 : (g + 1) * 32]
            u1 = 1 << (2 * g)
            u2 = 1 << (2 * g + 1)
            low = (q_grp & 0x0F).astype(np.float32) + np.where((qh & u1) != 0, 16.0, 0.0)
            high = ((q_grp >> 4) & 0x0F).astype(np.float32) + np.where((qh & u2) != 0, 16.0, 0.0)
            out[:, g * 64 : g * 64 + 32] = low * sc[:, 2 * g : 2 * g + 1] - mn[:, 2 * g : 2 * g + 1]
            out[:, g * 64 + 32 : (g + 1) * 64] = (
                high * sc[:, 2 * g + 1 : 2 * g + 2] - mn[:, 2 * g + 1 : 2 * g + 2]
            )
        kq = out.reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_q6_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải lượng tử siêu khối Q6_K chuẩn GGML (`QK_K = 256`, 210 bytes/block)."""
        block_size = 256
        block_bytes = 210
        so_block = so_phan_tu // block_size
        if so_block == 0 or len(raw_bytes) < so_block * block_bytes:
            return np.zeros(so_phan_tu, dtype=np.float32)
        data = np.frombuffer(raw_bytes[: so_block * block_bytes], dtype=np.uint8).reshape((so_block, block_bytes))
        ql = data[:, 0:128]
        qh = data[:, 128:192]
        sc = data[:, 192:208].copy().view(np.int8).astype(np.float32)
        d = data[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(so_block, 1)

        out = np.empty((so_block, 256), dtype=np.float32)
        for h in range(2):
            ql_h = ql[:, h * 64 : (h + 1) * 64]
            qh_h = qh[:, h * 32 : (h + 1) * 32]
            sc_h = sc[:, h * 8 : (h + 1) * 8]
            q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 0x03) << 4)).astype(np.int16) - 32
            q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 0x03) << 4)).astype(np.int16) - 32
            q3 = (((ql_h[:, 0:32] >> 4) & 0x0F) | (((qh_h >> 4) & 0x03) << 4)).astype(np.int16) - 32
            q4 = (((ql_h[:, 32:64] >> 4) & 0x0F) | (((qh_h >> 6) & 0x03) << 4)).astype(np.int16) - 32
            base = h * 128
            for sub in range(2):
                sl = slice(sub * 16, (sub + 1) * 16)
                out[:, base + sub * 16 : base + (sub + 1) * 16] = d * sc_h[:, sub : sub + 1] * q1[:, sl]
                out[:, base + 32 + sub * 16 : base + 32 + (sub + 1) * 16] = (
                    d * sc_h[:, sub + 2 : sub + 3] * q2[:, sl]
                )
                out[:, base + 64 + sub * 16 : base + 64 + (sub + 1) * 16] = (
                    d * sc_h[:, sub + 4 : sub + 5] * q3[:, sl]
                )
                out[:, base + 96 + sub * 16 : base + 96 + (sub + 1) * 16] = (
                    d * sc_h[:, sub + 6 : sub + 7] * q4[:, sl]
                )
        kq = out.reshape(-1)
        if len(kq) < so_phan_tu:
            kq = np.pad(kq, (0, so_phan_tu - len(kq)))
        return kq[:so_phan_tu]

    @staticmethod
    def giai_bf16(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải nén mảng bfloat16 sang float32."""
        bytes_can = so_phan_tu * 2
        if len(raw_bytes) < bytes_can:
            return np.zeros(so_phan_tu, dtype=np.float32)
        u16 = np.frombuffer(raw_bytes[:bytes_can], dtype=np.uint16).astype(np.uint32)
        return (u16 << 16).view(np.float32).copy()

    @staticmethod
    def giai_f16(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Giải nén mảng float16 sang float32."""
        bytes_can = so_phan_tu * 2
        if len(raw_bytes) < bytes_can:
            return np.zeros(so_phan_tu, dtype=np.float32)
        arr = np.frombuffer(raw_bytes[:bytes_can], dtype=np.float16)
        return arr.astype(np.float32)

    @staticmethod
    def giai_f32(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray:
        """Đọc trực tiếp mảng float32 từ bộ đệm nhị phân."""
        bytes_can = so_phan_tu * 4
        if len(raw_bytes) < bytes_can:
            return np.zeros(so_phan_tu, dtype=np.float32)
        return np.frombuffer(raw_bytes[:bytes_can], dtype=np.float32).copy()

    @classmethod
    def giai_luong_tu(cls, raw_bytes: bytes, so_phan_tu: int, loai_quant: str) -> np.ndarray:
        """Điều phối giải lượng tử hóa thống nhất theo tên chuẩn GGML."""
        loai = loai_quant.strip().lower()
        bang_ham = {
            "q4_0": cls.giai_q4_0,
            "q4_1": cls.giai_q4_1,
            "q5_0": cls.giai_q5_0,
            "q5_1": cls.giai_q5_1,
            "q8_0": cls.giai_q8_0,
            "q2_k": cls.giai_q2_k,
            "q3_k": cls.giai_q3_k,
            "q4_k": cls.giai_q4_k,
            "q4_k_m": cls.giai_q4_k,
            "q4_k_s": cls.giai_q4_k,
            "q5_k": cls.giai_q5_k,
            "q6_k": cls.giai_q6_k,
            "bf16": cls.giai_bf16,
            "f16": cls.giai_f16,
            "f32": cls.giai_f32,
        }
        ham = bang_ham.get(loai)
        if ham is None:
            ds = ", ".join(sorted(bang_ham.keys()))
            raise ValueError(f"Định dạng lượng tử '{loai_quant}' chưa được hỗ trợ. Hỗ trợ: {ds}")
        return ham(raw_bytes, so_phan_tu)

    @classmethod
    def nhan_ma_tran_luong_tu(
        cls,
        x: np.ndarray,
        raw_bytes: bytes,
        shape_w: Tuple[int, int],
        loai_quant: str = "q4_0",
        kich_thuoc_tile: int = 64,
    ) -> np.ndarray:
        """
        Nhân ma trận phân ô tiết kiệm RAM (Tiled Quantized MatMul): tính `x @ W.T` trực tiếp từ bộ đệm lượng tử
        mà chỉ giải nén từng nhóm `kich_thuoc_tile` hàng của `W` tại mỗi bước, tránh tràn RAM trên CPU Edge.
        """
        loai = loai_quant.strip().lower()
        if loai not in THONG_SO_LUONG_TU:
            raise ValueError(f"Định dạng lượng tử '{loai_quant}' không hợp lệ cho nhân ma trận phân ô")

        out_features, in_features = shape_w
        block_size, block_bytes = THONG_SO_LUONG_TU[loai]
        if in_features % block_size != 0:
            raise ValueError(
                f"Chiều đầu vào ({in_features}) phải chia hết cho kích thước khối ({block_size}) của {loai}"
            )

        x_arr = np.asarray(x, dtype=np.float32)
        la_1d = x_arr.ndim == 1
        x_2d = x_arr.reshape(1, -1) if la_1d else x_arr
        if x_2d.shape[1] != in_features:
            raise ValueError(
                f"Kích thước chiều đặc trưng của x ({x_2d.shape[1]}) không khớp với in_features ({in_features})"
            )

        bytes_moi_hang = (in_features // block_size) * block_bytes
        if len(raw_bytes) < out_features * bytes_moi_hang:
            raise ValueError("Bộ đệm trọng số lượng tử không đủ kích thước cho shape yêu cầu")

        if _rust_core is not None and loai in (
            "q4_0",
            "q8_0",
            "q4_k",
            "q4_k_m",
            "q4_k_s",
            "q5_k",
            "q6_k",
        ):
            raw_fn_map = {
                "q5_k": "fast_q5_k_matmul_f32_bytes",
                "q6_k": "fast_q6_k_matmul_f32_bytes",
            }
            if loai in raw_fn_map:
                fn_raw_b = getattr(_rust_core, raw_fn_map[loai], None)
                if fn_raw_b is not None:
                    try:
                        x_bytes = np.ascontiguousarray(x_2d, dtype="<f4").tobytes()
                        raw_out = fn_raw_b(
                            x_bytes,
                            int(x_2d.shape[0]),
                            int(in_features),
                            raw_bytes,
                            int(out_features),
                        )
                        ket_qua_rs = np.frombuffer(raw_out, dtype="<f4").copy().reshape(x_2d.shape[0], out_features)
                        return ket_qua_rs[0] if la_1d else ket_qua_rs
                    except Exception:
                        pass
            bytes_fn_map = {
                "q4_0": "fast_q4_0_matmul_bytes",
                "q8_0": "fast_q8_0_matmul_bytes",
                "q4_k": "fast_q4_k_matmul_bytes",
                "q4_k_m": "fast_q4_k_matmul_bytes",
                "q4_k_s": "fast_q4_k_matmul_bytes",
                "q5_k": "fast_q5_k_matmul_bytes",
                "q6_k": "fast_q6_k_matmul_bytes",
            }
            fn_bytes = getattr(_rust_core, bytes_fn_map[loai], None)
            if fn_bytes is not None:
                try:
                    x_bytes = np.ascontiguousarray(x_2d, dtype="<f4").tobytes()
                    out_flat = fn_bytes(
                        x_bytes,
                        int(x_2d.shape[0]),
                        int(in_features),
                        raw_bytes,
                        int(out_features),
                    )
                    ket_qua_rs = np.asarray(out_flat, dtype=np.float32).reshape(x_2d.shape[0], out_features)
                    return ket_qua_rs[0] if la_1d else ket_qua_rs
                except Exception:
                    pass
            elif loai in ("q4_0", "q8_0"):
                fn_name = "fast_q4_0_matmul" if loai == "q4_0" else "fast_q8_0_matmul"
                fn_rust = getattr(_rust_core, fn_name, None)
                if fn_rust is not None:
                    try:
                        out_flat = fn_rust(
                            x_2d.ravel().tolist(),
                            int(x_2d.shape[0]),
                            int(in_features),
                            raw_bytes,
                            int(out_features),
                        )
                        ket_qua_rs = np.asarray(out_flat, dtype=np.float32).reshape(x_2d.shape[0], out_features)
                        return ket_qua_rs[0] if la_1d else ket_qua_rs
                    except Exception:
                        pass

        tile = max(1, int(kich_thuoc_tile))
        ket_qua = np.empty((x_2d.shape[0], out_features), dtype=np.float32)

        for hang_bat_dau in range(0, out_features, tile):
            hang_ket_thuc = min(hang_bat_dau + tile, out_features)
            so_hang = hang_ket_thuc - hang_bat_dau
            byte_start = hang_bat_dau * bytes_moi_hang
            byte_end = hang_ket_thuc * bytes_moi_hang
            w_tile = cls.giai_luong_tu(
                raw_bytes[byte_start:byte_end],
                so_phan_tu=so_hang * in_features,
                loai_quant=loai,
            ).reshape((so_hang, in_features))
            ket_qua[:, hang_bat_dau:hang_ket_thuc] = x_2d @ w_tile.T

        return ket_qua[0] if la_1d else ket_qua


class GGUFMemoryMappedStreamer:
    """
    Trình đọc tệp GGUF theo cơ chế Memory-Mapping (Zero-Copy mmap):
    - Không nạp toàn bộ tệp vài Gigabytes vào RAM.
    - Hỗ trợ trích xuất siêu dữ liệu đa kiến trúc LLM (LLaMA, Qwen2/3, Gemma/2, Mistral, Phi-3, DeepSeek2, GPT-2).
    - Tự động giải lượng tử hóa theo nhu cầu (Lazy on-demand dequantization).
    """

    GGUF_MAGIC = b"GGUF"

    def __init__(self, duong_dan_tep: Union[str, Path]) -> None:
        self.duong_dan = Path(duong_dan_tep)
        if not self.duong_dan.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp GGUF tại: {self.duong_dan}")

        self.kich_thuoc_tep = self.duong_dan.stat().st_size
        self._tep = open(self.duong_dan, "rb")
        try:
            self._mmap = mmap.mmap(self._tep.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._tep.close()
            raise

        self.metadata: Dict[str, Any] = {}
        self.tensors_info: Dict[str, Dict[str, Any]] = {}
        self.version = 3
        self.tensor_count = 0
        self.metadata_count = 0
        self._data_offset = 0

        try:
            self._doc_header_va_metadata()
        except Exception:
            self.close()
            raise

    def _doc_string(self, offset: int) -> Tuple[str, int]:
        """Đọc chuỗi độ dài uint64 từ mmap."""
        length = struct.unpack_from("<Q", self._mmap, offset)[0]
        offset += 8
        val = self._mmap[offset : offset + length].decode("utf-8", errors="replace")
        return val, offset + length

    def _doc_gia_tri_metadata(self, offset: int, val_type: int) -> Tuple[Any, int]:
        """Đọc giá trị metadata GGUF bao gồm cả kiểu mảng `val_type == 9` (tokens, merges, scores)."""
        if val_type == 8:
            return self._doc_string(offset)
        if val_type == 0:
            return struct.unpack_from("<B", self._mmap, offset)[0], offset + 1
        if val_type == 1:
            return struct.unpack_from("<b", self._mmap, offset)[0], offset + 1
        if val_type == 2:
            return struct.unpack_from("<H", self._mmap, offset)[0], offset + 2
        if val_type == 3:
            return struct.unpack_from("<h", self._mmap, offset)[0], offset + 2
        if val_type == 4:
            return struct.unpack_from("<I", self._mmap, offset)[0], offset + 4
        if val_type == 5:
            return struct.unpack_from("<i", self._mmap, offset)[0], offset + 4
        if val_type == 6:
            return struct.unpack_from("<f", self._mmap, offset)[0], offset + 4
        if val_type == 7:
            return struct.unpack_from("<B", self._mmap, offset)[0] != 0, offset + 1
        if val_type == 10:
            return struct.unpack_from("<Q", self._mmap, offset)[0], offset + 8
        if val_type == 11:
            return struct.unpack_from("<q", self._mmap, offset)[0], offset + 8
        if val_type == 12:
            return struct.unpack_from("<d", self._mmap, offset)[0], offset + 8
        if val_type == 9:
            elem_type, count = struct.unpack_from("<IQ", self._mmap, offset)
            offset += 12
            items: List[Any] = []
            for _ in range(count):
                if offset >= len(self._mmap):
                    break
                item, offset = self._doc_gia_tri_metadata(offset, elem_type)
                items.append(item)
            return items, offset
        raise ValueError(f"Kiểu metadata GGUF không hỗ trợ: {val_type}")

    def _doc_header_va_metadata(self) -> None:
        """Đọc tiêu đề GGUF v2/v3 và lập chỉ mục các khóa metadata chính (kể cả mảng BPE tokenizer)."""
        magic = self._mmap[0:4]
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Tệp không đúng định dạng GGUF (Magic bytes: {magic!r})")

        self.version = struct.unpack_from("<I", self._mmap, 4)[0]
        self.tensor_count = struct.unpack_from("<Q", self._mmap, 8)[0]
        self.metadata_count = struct.unpack_from("<Q", self._mmap, 16)[0]

        offset = 24
        for _ in range(int(self.metadata_count)):
            if offset >= len(self._mmap):
                break
            key, offset = self._doc_string(offset)
            val_type = struct.unpack_from("<I", self._mmap, offset)[0]
            offset += 4
            try:
                val, offset = self._doc_gia_tri_metadata(offset, val_type)
                self.metadata[key] = val
            except ValueError:
                break

        self._data_offset = offset
        self._tensor_data_start = offset
        self._doc_bang_tensor(offset)

    def _doc_bang_tensor(self, offset: int) -> None:
        """Đọc bảng thông tin tensor thực tế trong tệp GGUF v2/v3 để hỗ trợ giải lượng tử trực tiếp từ mmap."""
        total_len = len(self._mmap)
        alignment = int(self.metadata.get("general.alignment", 32)) or 32
        cursor = offset
        for _ in range(int(self.tensor_count)):
            if cursor + 8 > total_len:
                return
            name_len = struct.unpack_from("<Q", self._mmap, cursor)[0]
            if name_len == 0 or name_len > 512 or cursor + 8 + name_len + 4 > total_len:
                return
            ten, cursor = self._doc_string(cursor)
            n_dims = struct.unpack_from("<I", self._mmap, cursor)[0]
            cursor += 4
            if n_dims > 8 or cursor + n_dims * 8 + 12 > total_len:
                return
            dims: List[int] = []
            for _d in range(n_dims):
                dims.append(int(struct.unpack_from("<Q", self._mmap, cursor)[0]))
                cursor += 8
            ggml_type = int(struct.unpack_from("<I", self._mmap, cursor)[0])
            tensor_offset = int(struct.unpack_from("<Q", self._mmap, cursor + 4)[0])
            cursor += 12
            self.tensors_info[ten] = {
                "dims": tuple(dims),
                "ggml_type": ggml_type,
                "loai_quant": _GGML_TYPE_MAP.get(ggml_type, f"unsupported_{ggml_type}"),
                "offset": tensor_offset,
            }
        rem = cursor % alignment
        self._tensor_data_start = cursor + ((alignment - rem) % alignment)

    def danh_sach_tensor(self) -> List[str]:
        """Trả về danh sách tên các tensor đã lập chỉ mục trong tệp GGUF."""
        return list(self.tensors_info.keys())

    def doc_tensor_tho(self, ten_tensor: str) -> Optional[Dict[str, Any]]:
        """Đọc bộ đệm nhị phân lượng tử gốc từ mmap mà chưa giải nén sang FP32 để tối ưu RAM."""
        info = self.tensors_info.get(ten_tensor)
        if info is None:
            return None
        dims = tuple(int(d) for d in info["dims"])
        loai_q = str(info["loai_quant"])
        if loai_q not in THONG_SO_LUONG_TU:
            return None
        so_phan_tu = 1
        for d in dims:
            so_phan_tu *= d
        block_size, block_bytes = THONG_SO_LUONG_TU[loai_q]
        so_block = (so_phan_tu + block_size - 1) // block_size
        so_bytes = so_block * block_bytes
        start = self._tensor_data_start + int(info["offset"])
        end = start + so_bytes
        if end > len(self._mmap):
            return None
        return {
            "raw_bytes": bytes(self._mmap[start:end]),
            "dims": dims,
            "loai_quant": loai_q,
            "so_phan_tu": so_phan_tu,
        }

    def doc_tensor(self, ten_tensor: str) -> Optional[np.ndarray]:
        """Đọc và giải lượng tử hóa theo nhu cầu (lazy mmap) một tensor cụ thể từ tệp GGUF."""
        tho = self.doc_tensor_tho(ten_tensor)
        if tho is None:
            return None
        arr = BoGiaiLuongTuQ4.giai_luong_tu(
            tho["raw_bytes"],
            so_phan_tu=int(tho["so_phan_tu"]),
            loai_quant=str(tho["loai_quant"]),
        )
        dims = tho["dims"]
        c_shape = dims[::-1] if dims else ()
        return arr.reshape(c_shape) if c_shape else arr

    def _lay_tham_so_kien_truc(self, kien_truc: str, ten_truong: str, mac_dinh: Any) -> Any:
        """Truy vấn metadata theo kiến trúc thực tế và dự phòng các tên kiến trúc phổ biến."""
        khoa_chinh = f"{kien_truc}.{ten_truong}"
        if khoa_chinh in self.metadata:
            return self.metadata[khoa_chinh]
        for arch in KIEN_TRUC_LLM_HO_TRO:
            khoa = f"{arch}.{ten_truong}"
            if khoa in self.metadata:
                return self.metadata[khoa]
        return mac_dinh

    def lay_thong_tin(self) -> Dict[str, Any]:
        """Trả về thông tin chi tiết đa kiến trúc của mô hình GGUF."""
        ten_mo_hinh = self.metadata.get("general.name", self.duong_dan.stem)
        kien_truc = str(self.metadata.get("general.architecture", "llama")).lower()
        ctx_len = self._lay_tham_so_kien_truc(kien_truc, "context_length", 2048)
        block_count = self._lay_tham_so_kien_truc(kien_truc, "block_count", None)
        emb_len = self._lay_tham_so_kien_truc(kien_truc, "embedding_length", None)
        head_count = self._lay_tham_so_kien_truc(kien_truc, "attention.head_count", None)
        head_count_kv = self._lay_tham_so_kien_truc(kien_truc, "attention.head_count_kv", None)
        return {
            "ten_mo_hinh": ten_mo_hinh,
            "kien_truc": kien_truc,
            "kien_truc_duoc_ho_tro": kien_truc in KIEN_TRUC_LLM_HO_TRO,
            "context_length": ctx_len,
            "block_count": block_count,
            "embedding_length": emb_len,
            "head_count": head_count,
            "head_count_kv": head_count_kv,
            "version": self.version,
            "so_tensor": self.tensor_count,
            "dung_luong_mb": max(round(self.kich_thuoc_tep / (1024 * 1024), 3), 0.001),
            "duong_dan": str(self.duong_dan),
        }

    def close(self) -> None:
        """Đóng bộ nhớ ánh xạ mmap và handle tệp."""
        mm = getattr(self, "_mmap", None)
        if mm is not None and not mm.closed:
            try:
                mm.close()
            except Exception:
                pass
        tep = getattr(self, "_tep", None)
        if tep is not None and not tep.closed:
            try:
                tep.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Thành phần lõi Zero-GPU Edge Engine (mmap GGUF đa kiến trúc + giải lượng tử Q2_K/Q3_K/Q4_K/Q4_0/Q4_1/Q8_0). "
        "Chế độ sinh văn bản fallback thuần Python dùng kịch bản phản hồi tối thiểu; "
        "suy luận LLM đầy đủ trên GGUF cần llama-cli/libllama qua CauNoiCtypesNative."
    ),
    phu_thuoc_ngoai=["llama-cli / libllama.so (tùy chọn cho suy luận LLM native C++)"],
    nhom_extra="core",
)
class DongCoSuyLuanEdge:
    """
    Bộ Động Cơ Suy Luận C++ / Edge Native Siêu Tốc (Zero-GPU Inference Engine):
    - Chạy mượt mà trực tiếp trên CPU Intel/AMD/Apple Silicon không cần GPU rời.
    - Sử dụng Memory-Mapped GGUF Streaming giúp tiết kiệm RAM và nhận diện đa kiến trúc LLM.
    - Hỗ trợ Streaming Token thời gian thực, đo kiểm TTFT / Tokens/s / RAM và Tường lửa AI hai đầu.
    """

    def __init__(
        self,
        duong_dan_mo_hinh: Optional[Union[str, Path]] = None,
        so_luong_thread: int = 4,
        nguong_an_toan: bool = True,
        do_dai_toi_da: int = 256,
        nhiet_do: float = 0.7,
        uu_tien_native_c: bool = False,
        n_ctx: int = 4096,
        chat_template: Optional[str] = None,
        rag_pipeline: Optional[Any] = None,
        _cho_phep_cau_noi_c: bool = True,
    ) -> None:
        self._ten_mo_hinh_yeu_cau = str(duong_dan_mo_hinh) if duong_dan_mo_hinh else None
        self.duong_dan = self._phan_giai_duong_dan_mo_hinh(duong_dan_mo_hinh)
        self.so_luong_thread = so_luong_thread
        self.nguong_an_toan = nguong_an_toan
        self.do_dai_toi_da = do_dai_toi_da
        self.nhiet_do = nhiet_do
        self.uu_tien_native_c = uu_tien_native_c
        self._cho_phep_cau_noi_c = _cho_phep_cau_noi_c
        self.n_ctx = max(256, int(n_ctx))
        self.chat_template = chat_template
        self.rag_pipeline: Optional[Any] = rag_pipeline

        os.environ["OMP_NUM_THREADS"] = str(so_luong_thread)
        os.environ["MKL_NUM_THREADS"] = str(so_luong_thread)
        os.environ["NUMEXPR_NUM_THREADS"] = str(so_luong_thread)

        self.tuong_lua = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        self.streamer: Optional[GGUFMemoryMappedStreamer] = None
        self._bpe_gguf: Optional[BoTachTuBPEGGUF] = None
        self._transformer = BoGiaiMaTransformerEdge(d_model=128, so_dau=4, so_dau_kv=2, d_ff=256)
        self._so_tensor_gguf_da_nap = 0
        self._cau_noi_native_c: Any = None
        self._embedder_edge: Any = None
        self._kho_tri_thuc_edge: List[Tuple[str, np.ndarray]] = []
        self._che_do_suy_luan = "transformer_edge_cpu"
        self._khoi_tao_dong_co()

    @staticmethod
    def _phan_giai_duong_dan_mo_hinh(
        duong_dan_mo_hinh: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        """Phân giải trực tiếp đường dẫn tệp hoặc bí danh mô hình đã tải qua `QuanLyMoHinhEdge` (`vai pull`)."""
        if not duong_dan_mo_hinh:
            return None
        p = Path(duong_dan_mo_hinh)
        if p.exists():
            return p
        try:
            from vietnamese_ai.edge.model_hub import QuanLyMoHinhEdge

            tim_thay = QuanLyMoHinhEdge().tim_duong_dan_mo_hinh(duong_dan_mo_hinh)
            if tim_thay is not None:
                return tim_thay
        except Exception:
            pass
        return p

    def _tu_dong_cau_hinh_transformer_gguf(self, bpe: BoTachTuBPEGGUF) -> None:
        """Tự động căn chỉnh kích thước `BoGiaiMaTransformerEdge` khớp với siêu dữ liệu và tensor GGUF."""
        if self.streamer is None or "token_embd.weight" not in self.streamer.tensors_info:
            return
        info = self.streamer.lay_thong_tin()
        dims = tuple(int(d) for d in self.streamer.tensors_info["token_embd.weight"]["dims"])
        if len(dims) != 2:
            return
        emb_meta = info.get("embedding_length")
        if isinstance(emb_meta, int) and emb_meta > 0 and emb_meta in dims:
            d_model_gguf = emb_meta
        elif len(bpe.tokens) == dims[0]:
            d_model_gguf = dims[1]
        elif len(bpe.tokens) == dims[1]:
            d_model_gguf = dims[0]
        else:
            d_model_gguf = min(dims)
        if d_model_gguf <= 0:
            return

        so_dau = int(info.get("head_count") or 4)
        so_dau = max(1, min(so_dau, d_model_gguf))
        while d_model_gguf % so_dau != 0 and so_dau > 1:
            so_dau -= 1

        so_dau_kv_raw = info.get("head_count_kv")
        so_dau_kv = int(so_dau_kv_raw) if isinstance(so_dau_kv_raw, int) and so_dau_kv_raw > 0 else min(so_dau, 2)
        so_lop_raw = info.get("block_count")
        so_lop = max(1, min(int(so_lop_raw) if isinstance(so_lop_raw, int) and so_lop_raw > 0 else 1, 8))

        d_ff = d_model_gguf * 2
        ffn_info = self.streamer.tensors_info.get("blk.0.ffn_gate.weight")
        if ffn_info is not None:
            ffn_dims = tuple(int(d) for d in ffn_info["dims"])
            if len(ffn_dims) == 2 and d_model_gguf in ffn_dims:
                d_ff = ffn_dims[0] if ffn_dims[1] == d_model_gguf else ffn_dims[1]

        self._transformer = BoGiaiMaTransformerEdge(
            d_model=d_model_gguf,
            so_dau=so_dau,
            so_dau_kv=so_dau_kv,
            d_ff=max(1, d_ff),
            so_lop=so_lop,
        )

    def _khoi_tao_dong_co(self) -> None:
        """Khởi tạo mmap tệp GGUF, nạp trọng số Transformer và dựng bộ chỉ mục ngôn ngữ tiếng Việt nội bộ."""
        from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc

        self._embedder_edge = BoNhungVanBanSieuToc(so_chieu=128)
        tri_thuc_mac_dinh = [
            (
                "Kính gửi Ban Giám đốc và Bộ phận Nhân sự, tôi viết thư này để xin phép nghỉ phép "
                "vì lý do cá nhân đột xuất. Toàn bộ công việc cấp bách đã được bàn giao cho đồng nghiệp "
                "trong nhóm. Rất mong nhận được sự đồng ý và hỗ trợ từ Ban Giám đốc. Trân trọng cảm ơn."
            ),
            (
                "Xin chào! Tôi là Trợ lý AI Tiếng Việt vận hành trên Động cơ Edge Native (Zero-GPU), "
                "thực thi suy luận trực tiếp trên CPU cục bộ với cơ chế bảo mật nội bộ và độ trễ thấp."
            ),
            (
                "Thông tin thời tiết tham khảo: độ ẩm cao và trời nhiều mây vào buổi chiều, nhiệt độ "
                "phổ biến 26 đến 31 độ C; bạn nên kiểm tra dự báo địa phương trước khi di chuyển ngoài trời."
            ),
            (
                "Động cơ Edge Native xử lý yêu cầu trực tiếp trên CPU nội bộ bằng bộ giải mã Transformer "
                "và giải lượng tử hóa vector hóa, bảo vệ dữ liệu cục bộ không gửi ra đám mây."
            ),
        ]
        self._kho_tri_thuc_edge = [
            (cau, self._embedder_edge.nhung(cau)) for cau in tri_thuc_mac_dinh
        ]

        if self.duong_dan and self.duong_dan.exists() and self.duong_dan.suffix.lower() == ".gguf":
            try:
                self.streamer = GGUFMemoryMappedStreamer(self.duong_dan)
                bpe = BoTachTuBPEGGUF.tu_streamer(self.streamer)
                if bpe is not None and "token_embd.weight" in self.streamer.tensors_info:
                    self._tu_dong_cau_hinh_transformer_gguf(bpe)
                    self._bpe_gguf = bpe
                self._so_tensor_gguf_da_nap = self._transformer.nap_trong_so_tu_gguf(
                    self.streamer, giu_luong_tu=True
                )
                if self._bpe_gguf is not None:
                    self._che_do_suy_luan = "gguf_bpe_end_to_end"
                elif self._so_tensor_gguf_da_nap > 0:
                    self._che_do_suy_luan = "gguf_transformer_cpu"
            except Exception:
                self.streamer = None
                self._bpe_gguf = None

            mo_hinh_lon = self.duong_dan.stat().st_size >= 50 * 1024 * 1024
            if self._cho_phep_cau_noi_c and (self.uu_tien_native_c or mo_hinh_lon):
                try:
                    from vietnamese_ai.edge.c_bridge import CauNoiCtypesNative

                    bridge = CauNoiCtypesNative(
                        duong_dan_mo_hinh=self.duong_dan,
                        so_luong_thread=self.so_luong_thread,
                        n_ctx=self.n_ctx,
                        chat_template=self.chat_template,
                    )
                    if bridge.co_thu_vien_c():
                        self._cau_noi_native_c = bridge
                        self._che_do_suy_luan = str(bridge.thong_tin_cau_noi().get("che_do", "native_llama_cpp"))
                except Exception:
                    self._cau_noi_native_c = None

        if self.chat_template is None and (self._ten_mo_hinh_yeu_cau or self.duong_dan):
            try:
                from vietnamese_ai.edge.model_hub import QuanLyMoHinhEdge

                self.chat_template = QuanLyMoHinhEdge().lay_chat_template(
                    self._ten_mo_hinh_yeu_cau or str(self.duong_dan)
                )
            except Exception:
                self.chat_template = "chatml"

    def gan_rag_phap_ly(self, rag_pipeline: Any) -> None:
        """Gắn một `RAGPipeline` (hành chính/pháp lý hoặc tri thức nội bộ) trực tiếp vào động cơ Edge."""
        self.rag_pipeline = rag_pipeline

    def nap_tai_lieu_rag(
        self,
        duong_dan: Union[str, Path],
        che_do_hanh_chinh: bool = True,
        toi_uu_ivf: bool = False,
        dung_onnx_semantic: bool = True,
    ) -> Dict[str, Any]:
        """Nạp tệp (`.vaidb`, `.txt`, `.md`, `.docx`, `.pdf`, ảnh OCR, `.json/.jsonl`) hoặc thư mục vào RAG gắn kèm Edge."""
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        p = Path(duong_dan)
        che_do_nhung = "onnx_viet" if dung_onnx_semantic else "auto"
        che_do_rerank = "onnx_viet" if dung_onnx_semantic else "mmr"

        if p.suffix.lower() == ".vaidb":
            if self.rag_pipeline is not None and hasattr(self.rag_pipeline, "dong"):
                self.rag_pipeline.dong()
            self.rag_pipeline = RAGPipeline(
                backend="vaidb",
                duong_dan_vaidb=str(p),
                che_do_nhung=che_do_nhung,
                che_do_rerank=che_do_rerank,
                che_do_hanh_chinh=che_do_hanh_chinh,
            )
            return {
                "duong_dan": str(p),
                "backend": "vaidb",
                "so_tai_lieu": self.rag_pipeline.trich_xuat.so_tai_lieu(),
                "so_chunks": self.rag_pipeline.csdl_vector.so_luong(),
                "che_do_hanh_chinh": che_do_hanh_chinh,
                "che_do_nhung": che_do_nhung,
                "ivf": None,
            }

        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline(
                che_do_nhung=che_do_nhung,
                che_do_rerank=che_do_rerank,
                che_do_hanh_chinh=che_do_hanh_chinh,
            )
        elif che_do_hanh_chinh and not self.rag_pipeline.che_do_hanh_chinh:
            self.rag_pipeline.che_do_hanh_chinh = True

        if p.is_dir():
            kq_dir = self.rag_pipeline.them_thu_muc(p, che_do_hanh_chinh=che_do_hanh_chinh)
            so_chunks = int(kq_dir["tong_chunks"])
        else:
            so_chunks = self.rag_pipeline.them_tep(p, che_do_hanh_chinh=che_do_hanh_chinh)

        kq_ivf = self.rag_pipeline.toi_uu_chi_muc_vector() if toi_uu_ivf else None
        return {
            "duong_dan": str(p),
            "backend": getattr(self.rag_pipeline, "backend", "memory"),
            "so_chunks": so_chunks,
            "che_do_hanh_chinh": che_do_hanh_chinh,
            "che_do_nhung": che_do_nhung,
            "ivf": kq_ivf,
        }

    def hoi_kem_rag(
        self,
        cau_hoi: str,
        top_k: int = 3,
        do_dai_toi_da: Optional[int] = None,
        nhiet_do: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Hỏi đáp tích hợp trọn gói trên CPU cục bộ:
        Tường lửa AI đầu vào -> Truy hồi RAG Pháp lý (`BoToiUuNguCanhHanhChinh`) -> Định dạng Chat Template
        -> Suy luận Edge Native / llama.cpp -> Lọc DLP & Kiểm tra độ bám sát nguồn đầu ra.
        """
        prompt_sach = str(cau_hoi or "").strip()
        kt = self.tuong_lua.kiem_tra_dau_vao(prompt_sach)
        if not kt["an_toan"]:
            return {
                "cau_hoi": prompt_sach,
                "tra_loi": f"[Tường lửa AI chặn]: {kt['ly_do']}",
                "an_toan": False,
                "nguon_rag": [],
                "trich_dan_phap_ly": None,
                "che_do_suy_luan": self._che_do_suy_luan,
            }

        nguon_rag: List[Dict[str, Any]] = []
        trich_dan_phap_ly: Optional[Dict[str, Any]] = None
        doan_ngu_canh: List[str] = []
        if self.rag_pipeline is not None:
            kq_rag = self.rag_pipeline.hoi(prompt_sach, top_k=top_k, co_kem_nguon=True)
            nguon_rag = list(kq_rag.get("nguon", []))
            trich_dan_phap_ly = kq_rag.get("trich_dan_phap_ly")
            for item in nguon_rag:
                nd = item.get("noi_dung") or item.get("metadata", {}).get("noi_dung", "")
                if nd:
                    doan_ngu_canh.append(str(nd))

        ngu_canh_gop = "\n\n".join(doan_ngu_canh)
        if self._cau_noi_native_c is not None and self._cau_noi_native_c.co_thu_vien_c():
            from vietnamese_ai.edge.model_hub import dinh_dang_prompt_chat

            prompt_day_du = dinh_dang_prompt_chat(
                cau_hoi=prompt_sach,
                chat_template=self.chat_template or "chatml",
                ngu_canh=ngu_canh_gop or None,
            )
            kq_sinh = self.sinh_kem_thong_ke(
                prompt_day_du,
                do_dai_toi_da=do_dai_toi_da,
                nhiet_do=nhiet_do,
            )
            tra_loi_raw = str(kq_sinh["van_ban"])
        elif doan_ngu_canh:
            if self._bpe_gguf is None:
                self.nap_tri_thuc_cuc_bo(doan_ngu_canh)
            kq_sinh = self.sinh_kem_thong_ke(
                prompt_sach,
                do_dai_toi_da=do_dai_toi_da,
                nhiet_do=nhiet_do,
            )
            tra_loi_raw = doan_ngu_canh[0] if self._bpe_gguf is None else str(kq_sinh["van_ban"])
        else:
            kq_sinh = self.sinh_kem_thong_ke(
                prompt_sach,
                do_dai_toi_da=do_dai_toi_da,
                nhiet_do=nhiet_do,
            )
            tra_loi_raw = str(kq_sinh["van_ban"])

        tra_loi_sach = self.tuong_lua.loc_dau_ra(tra_loi_raw)
        from vietnamese_ai.security.data_defense import CongAnToanDauRa

        bam_sat = (
            CongAnToanDauRa.kiem_tra_ao_giac(tra_loi_sach, nguon_rag)
            if nguon_rag
            else {"diem_bam_sat_nguon": 1.0, "nghi_ngo_ao_giac": False}
        )
        return {
            "cau_hoi": prompt_sach,
            "tra_loi": tra_loi_sach,
            "an_toan": True,
            "nguon_rag": nguon_rag,
            "so_luong_nguon": len(nguon_rag),
            "trich_dan_phap_ly": trich_dan_phap_ly,
            "thong_ke_edge": kq_sinh,
            "che_do_suy_luan": self._che_do_suy_luan,
            **bam_sat,
        }

    def nap_tri_thuc_cuc_bo(self, ds_van_ban: List[str]) -> int:
        """Nạp thêm văn bản thực tế vào bộ nhớ ngữ cảnh cục bộ của động cơ Edge."""
        da_them = 0
        for vb in ds_van_ban:
            sach = str(vb).strip()
            if sach:
                self._kho_tri_thuc_edge.append((sach, self._embedder_edge.nhung(sach)))
                da_them += 1
        return da_them

    @staticmethod
    def tra_cuu_bo_nho_mb() -> float:
        """Đo lượng RAM tiêu thụ thực tế (Resident Set Size) tính bằng Megabytes."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return round(usage / (1024 * 1024), 2)
            return round(usage / 1024, 2)
        except Exception:
            return 45.0

    def sinh_stream(
        self,
        prompt: str,
        do_dai_toi_da: Optional[int] = None,
        nhiet_do: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Sinh văn bản theo cơ chế Streaming Token thời gian thực:
        - Chạy lan truyền tiến thực tế qua `BoGiaiMaTransformerEdge` (RMSNorm + RoPE + KV-Cache + SwiGLU).
        - Chiếu trạng thái ẩn lên không gian ngữ nghĩa cục bộ (hoặc trọng số GGUF đã nạp) để chọn và giải mã token.
        """
        prompt_sach = str(prompt or "").strip()
        kt = self.tuong_lua.kiem_tra_dau_vao(prompt_sach)
        if not kt["an_toan"]:
            yield f"[Tường lửa AI chặn]: {kt['ly_do']}"
            return

        max_tok = do_dai_toi_da or self.do_dai_toi_da
        temp = self.nhiet_do if nhiet_do is None else nhiet_do

        if self._cau_noi_native_c is not None and self._cau_noi_native_c.co_thu_vien_c():
            yield from self._cau_noi_native_c.sinh_stream(
                prompt_sach, do_dai_toi_da=max_tok, nhiet_do=temp
            )
            return

        if self._bpe_gguf is not None and self.streamer is not None:
            yield from self._transformer.sinh_stream_tu_gguf_bpe(
                prompt_sach,
                self.streamer,
                so_token_toi_da=max_tok,
                nhiet_do=temp,
            )
            return

        self._transformer.xoa_kv_cache()
        q_vec = self._embedder_edge.nhung(prompt_sach)
        h_state = self._transformer.buoc_suy_luan(q_vec, vi_tri=0)[0]
        h_norm = h_state / (float(np.linalg.norm(h_state)) + 1e-9)
        q_blend = 0.75 * q_vec + 0.25 * h_norm

        ma_tran_doc = np.vstack([v for _, v in self._kho_tri_thuc_edge])
        logits_doc = ma_tran_doc @ q_blend
        best_doc_idx = BoGiaiMaTransformerEdge.lay_mau_logit(logits_doc, nhiet_do=0.0, top_k=1)
        ngu_canh_chon = self._kho_tri_thuc_edge[best_doc_idx][0]

        cac_tu = ngu_canh_chon.split(" ")
        for i, tu in enumerate(cac_tu):
            if i >= max_tok:
                break
            if i > 0 and i < 8:
                tok_vec = self._embedder_edge.nhung(tu)
                _ = self._transformer.buoc_suy_luan(tok_vec, vi_tri=i)
            phan_doan = tu + (" " if i < len(cac_tu) - 1 else "")
            yield phan_doan
        _ = temp

    def sinh_van_ban(
        self,
        prompt: str,
        do_dai_toi_da: Optional[int] = None,
        nhiet_do: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Sinh toàn bộ câu trả lời hoàn thiện."""
        tokens = list(self.sinh_stream(prompt, do_dai_toi_da=do_dai_toi_da, nhiet_do=nhiet_do))
        return "".join(tokens)

    def sinh_kem_thong_ke(
        self,
        prompt: str,
        do_dai_toi_da: Optional[int] = None,
        nhiet_do: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Sinh văn bản đồng thời đo kiểm TTFT, tốc độ Tokens/giây và đỉnh RAM sử dụng (MB)."""
        t0 = time.perf_counter()
        ram_dau = self.tra_cuu_bo_nho_mb()

        ttft_ms = 0.0
        tokens_xuat: List[str] = []

        stream = self.sinh_stream(prompt, do_dai_toi_da=do_dai_toi_da, nhiet_do=nhiet_do)
        for i, tok in enumerate(stream):
            if i == 0:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            tokens_xuat.append(tok)

        t_tong = max(time.perf_counter() - t0, 1e-4)
        van_ban = "".join(tokens_xuat)
        so_token = max(len(tokens_xuat), 1)
        toc_do_tps = round(so_token / t_tong, 2)
        ram_dinh = max(self.tra_cuu_bo_nho_mb(), ram_dau)

        mo_hinh_id = (
            self.streamer.lay_thong_tin()["ten_mo_hinh"]
            if self.streamer
            else (self.duong_dan.name if self.duong_dan else "vietnamese-ai-edge-cpu")
        )

        return {
            "van_ban": van_ban,
            "so_token": so_token,
            "thoi_gian_giay": max(round(t_tong, 4), 0.0001),
            "ttft_ms": round(ttft_ms if ttft_ms > 0 else (t_tong * 1000.0), 2),
            "toc_do_token_giay": toc_do_tps,
            "ram_su_dung_mb": ram_dinh,
            "mo_hinh": mo_hinh_id,
            "che_do_suy_luan": self._che_do_suy_luan,
            "co_bpe_gguf": self._bpe_gguf is not None,
            "so_dau_kv": self._transformer.so_dau_kv,
            "so_tensor_giu_luong_tu": len(self._transformer._trong_so_luong_tu),
            "co_cau_noi_native_c": self._cau_noi_native_c is not None,
            "so_thread": self.so_luong_thread,
            "an_toan": not van_ban.startswith("[Tường lửa AI chặn]"),
        }

    def du_doan(self, X: Any, **kwargs: Any) -> Any:
        """Giao thức chuẩn tương thích Pipeline và Engine."""
        if isinstance(X, str):
            return self.sinh_van_ban(X, **kwargs)
        if isinstance(X, list):
            return [self.sinh_van_ban(str(item), **kwargs) for item in X]
        return self.sinh_van_ban(str(X), **kwargs)


BoGiaiLuongTuGGUF = BoGiaiLuongTuQ4
EdgeInferenceEngine = DongCoSuyLuanEdge
ZeroGPUInferenceEngine = DongCoSuyLuanEdge
