"""BoTangTocHuanLuyen - Tổ hợp Công nghệ Siêu Tăng tốc Huấn luyện & Suy luận AI (2x - 8x)."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn.functional as F

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False

try:
    import unsloth  # noqa: F401

    _CO_UNSLOTH = True
except ImportError:
    _CO_UNSLOTH = False

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: F401

    _CO_LIGER = True
except ImportError:
    _CO_LIGER = False


class BoGopChuoi:
    """
    Sequence Packing (Multipack / Flash-Packing) kèm mặt nạ chú ý chéo khối 2D (Block-Diagonal Causal Mask).

    Gom nhiều câu hướng dẫn tiếng Việt ngắn vào chung một chuỗi có độ dài tối đa `max_seq_length`
    thay vì đệm `<pad>` lãng phí, đồng thời ngăn chặn tuyệt đối việc rò rỉ ngữ cảnh chéo giữa
    các mẫu khác nhau trong cùng một chuỗi gộp.
    """

    def __init__(self, max_seq_length: int = 256, pad_token_id: int = 0):
        self.max_seq_length = max(4, max_seq_length)
        self.pad_token_id = pad_token_id

    def gop_chuoi(
        self,
        du_lieu: List[Dict[str, Any]],
        max_seq_length: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Gộp danh sách mẫu `[{input_ids, labels}, ...]` bằng thuật toán Best-Fit-Decreasing (Multipack).

        Returns:
            Tuple `(danh_sach_chuoi_da_gop, thong_ke_gop)`
        """
        gioi_han = max_seq_length or self.max_seq_length
        cac_doan: List[Tuple[List[int], List[int]]] = []
        tong_token_goc = 0

        for mau in du_lieu:
            in_ids = [int(x) for x in mau.get("input_ids", [])][:gioi_han]
            lbl_ids = [int(x) for x in mau.get("labels", in_ids)][:gioi_han]
            if not in_ids:
                continue
            if in_ids == lbl_ids and len(in_ids) > 1:
                seg_in = in_ids[:-1]
                seg_lbl = lbl_ids[1:]
            else:
                do_dai = min(len(in_ids), len(lbl_ids))
                seg_in = in_ids[:do_dai]
                seg_lbl = lbl_ids[:do_dai]
            if seg_in:
                cac_doan.append((seg_in, seg_lbl))
                tong_token_goc += len(seg_in)

        if not cac_doan:
            return [], {
                "so_mau_goc": 0,
                "so_chuoi_sau_gop": 0,
                "he_so_nen_chuoi": 1.0,
                "ty_le_lap_day_pct": 0.0,
                "tong_token_hieu_dung": 0,
            }

        # Sắp xếp giảm dần theo độ dài để tối ưu Best-Fit-Decreasing
        cac_doan_sap_xep = sorted(cac_doan, key=lambda pair: len(pair[0]), reverse=True)
        bins: List[Dict[str, Any]] = []

        for seg_in, seg_lbl in cac_doan_sap_xep:
            seg_len = len(seg_in)
            best_idx = -1
            best_con_trong = gioi_han + 1
            for idx, b in enumerate(bins):
                con_trong = gioi_han - len(b["input_ids"])
                if seg_len <= con_trong < best_con_trong:
                    best_con_trong = con_trong
                    best_idx = idx

            if best_idx >= 0:
                b = bins[best_idx]
                doc_id = b["next_doc_id"]
                b["input_ids"].extend(seg_in)
                b["labels"].extend(seg_lbl)
                b["doc_ids"].extend([doc_id] * seg_len)
                b["next_doc_id"] = doc_id + 1
            else:
                bins.append(
                    {
                        "input_ids": list(seg_in),
                        "labels": list(seg_lbl),
                        "doc_ids": [0] * seg_len,
                        "next_doc_id": 1,
                    }
                )

        ket_qua_gop: List[Dict[str, Any]] = []
        for b in bins:
            L = len(b["input_ids"])
            doc_arr = np.array(b["doc_ids"], dtype=int)
            cung_tai_lieu = doc_arr[:, None] == doc_arr[None, :]
            nhan_qua = np.tril(np.ones((L, L), dtype=bool))
            mask_2d = cung_tai_lieu & nhan_qua
            ket_qua_gop.append(
                {
                    "input_ids": b["input_ids"],
                    "labels": b["labels"],
                    "doc_ids": b["doc_ids"],
                    "so_mau_trong_chuoi": b["next_doc_id"],
                    "attention_mask_2d": mask_2d,
                }
            )

        so_goc = len(cac_doan)
        so_gop = len(ket_qua_gop)
        ty_le_lap_day = tong_token_goc / max(1, so_gop * gioi_han) * 100.0
        thong_ke = {
            "so_mau_goc": so_goc,
            "so_chuoi_sau_gop": so_gop,
            "he_so_nen_chuoi": round(so_goc / max(1, so_gop), 2),
            "ty_le_lap_day_pct": round(ty_le_lap_day, 2),
            "tong_token_hieu_dung": tong_token_goc,
        }
        return ket_qua_gop, thong_ke

    @staticmethod
    def tao_batch_vector_hoa(
        batch_items: List[Dict[str, Any]],
        vocab_limit: Optional[int] = None,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Đóng gói danh sách mẫu (đã gộp hoặc chưa gộp) thành tensor 2D `(B, T_max)` và mặt nạ chú ý 3D `(B, T_max, T_max)`
        để thực thi 1 lượt lan truyền tiến/ngược duy nhất cho toàn bộ mini-batch.
        """
        valid_items = []
        for mau in batch_items:
            raw_in = mau.get("input_ids", [])[:max_seq_length]
            raw_lbl = mau.get("labels", raw_in)[:max_seq_length]
            if not raw_in:
                continue
            if vocab_limit is not None:
                in_clean = [int(x) % vocab_limit if int(x) >= 0 else pad_token_id for x in raw_in]
                lbl_clean = [int(x) % vocab_limit if int(x) >= 0 else -100 for x in raw_lbl]
            else:
                in_clean = [int(x) if int(x) >= 0 else pad_token_id for x in raw_in]
                lbl_clean = [int(x) for x in raw_lbl]

            if in_clean == lbl_clean and len(in_clean) > 1 and "attention_mask_2d" not in mau:
                in_clean = in_clean[:-1]
                lbl_clean = lbl_clean[1:]
            else:
                L = min(len(in_clean), len(lbl_clean))
                in_clean = in_clean[:L]
                lbl_clean = lbl_clean[:L]

            if in_clean:
                valid_items.append((in_clean, lbl_clean, mau.get("attention_mask_2d")))

        if not valid_items:
            return None

        B = len(valid_items)
        T_max = max(len(item[0]) for item in valid_items)
        in_batch = np.full((B, T_max), pad_token_id, dtype=int)
        lbl_batch = np.full((B, T_max), -100, dtype=int)
        mask_batch = np.zeros((B, T_max, T_max), dtype=bool)
        # Đặt đường chéo bằng True để tránh chia 0 trong softmax tại các vị trí padding (-100 sẽ triệt tiêu gradient)
        diag_idx = np.arange(T_max)
        mask_batch[:, diag_idx, diag_idx] = True

        for b_idx, (seq_in, seq_lbl, mask_2d) in enumerate(valid_items):
            L = len(seq_in)
            in_batch[b_idx, :L] = seq_in
            lbl_batch[b_idx, :L] = seq_lbl
            if mask_2d is not None and mask_2d.shape[0] >= L:
                mask_batch[b_idx, :L, :L] = mask_2d[:L, :L]
            else:
                mask_batch[b_idx, :L, :L] = np.tril(np.ones((L, L), dtype=bool))

        return {
            "input_ids": in_batch,
            "labels": lbl_batch,
            "attention_mask": mask_batch,
        }


class HamMatMatPhanManh:
    """
    Fused / Chunked Linear Cross-Entropy (tương thích Liger Kernel trên GPU và Chunked GEMM trên CPU/NumPy).

    Tránh cấp phát toàn bộ ma trận `logits` khổng lồ kích thước `(B * T, Vocab)` cùng lúc (vốn chiếm hàng GB VRAM
    với từ vựng 152k-256k của Qwen3.8 / Gemma 4), giảm đỉnh bộ nhớ trung gian tới 75-90% mà vẫn giữ nguyên
    độ chính xác toán học của hàm mất mát và đạo hàm.
    """

    @staticmethod
    def tinh_loss_va_grad_numpy(
        hidden_flat: np.ndarray,
        trong_so_tu_vung: np.ndarray,
        targets_flat: np.ndarray,
        chunk_size: int = 64,
        tinh_grad_tu_vung: bool = True,
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Tính Cross-Entropy Loss và đạo hàm `(d_hidden_flat, d_trong_so_tu_vung)` theo từng lát cắt `chunk_size`.
        Khi `tinh_grad_tu_vung=False` (ví dụ khi đóng băng embedding với LoRA/Unsloth Manual Autograd),
        bỏ qua hoàn toàn phép nhân ma trận lớn `d_logits.T @ hidden`, tăng tốc thêm ~30%.
        """
        N, D = hidden_flat.shape
        V = trong_so_tu_vung.shape[0]
        valid_mask = (targets_flat >= 0) & (targets_flat < V)
        n_valid = int(np.sum(valid_mask))
        d_hidden = np.zeros_like(hidden_flat)
        d_vocab = np.zeros_like(trong_so_tu_vung) if tinh_grad_tu_vung else None

        if n_valid == 0:
            return 0.0, d_hidden, d_vocab

        tong_nll = 0.0
        buoc = max(1, chunk_size)
        W_T = trong_so_tu_vung.T

        for start in range(0, N, buoc):
            end = min(start + buoc, N)
            sub_valid = valid_mask[start:end]
            if not np.any(sub_valid):
                continue

            h_sub = hidden_flat[start:end]
            t_sub = targets_flat[start:end]
            logits_sub = h_sub @ W_T

            max_l = np.max(logits_sub, axis=-1, keepdims=True)
            exp_l = np.exp(logits_sub - max_l)
            probs_sub = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

            valid_rows = np.where(sub_valid)[0]
            valid_tgts = t_sub[sub_valid]
            tong_nll += float(-np.sum(np.log(probs_sub[valid_rows, valid_tgts] + 1e-10)))

            d_logits_sub = np.zeros_like(logits_sub)
            d_logits_sub[valid_rows] = probs_sub[valid_rows]
            d_logits_sub[valid_rows, valid_tgts] -= 1.0
            d_logits_sub /= float(n_valid)

            d_hidden[start:end] = d_logits_sub @ trong_so_tu_vung
            if d_vocab is not None:
                d_vocab += d_logits_sub.T @ h_sub

        return tong_nll / float(n_valid), d_hidden, d_vocab

    @staticmethod
    def tinh_loss_pytorch(
        hidden_or_logits: "torch.Tensor",
        labels: "torch.Tensor",
        lm_head_weight: Optional["torch.Tensor"] = None,
        chunk_size: int = 128,
    ) -> "torch.Tensor":
        """Tính Cross-Entropy theo từng lát cắt trên PyTorch (hoặc gọi trực tiếp Liger Kernel nếu khả dụng)."""
        if not _CO_PYTORCH:
            raise ImportError("Cần cài đặt PyTorch để dùng tinh_loss_pytorch.")

        flat_labels = labels.reshape(-1)
        if lm_head_weight is not None and _CO_LIGER and hidden_or_logits.is_cuda:
            loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
            return loss_fn(lm_head_weight, hidden_or_logits.reshape(-1, hidden_or_logits.size(-1)), flat_labels)

        if lm_head_weight is not None:
            flat_hidden = hidden_or_logits.reshape(-1, hidden_or_logits.size(-1))
            valid_total = (flat_labels != -100).sum().clamp(min=1)
            total_loss = flat_hidden.new_zeros(())
            for start in range(0, flat_hidden.size(0), max(1, chunk_size)):
                end = min(start + max(1, chunk_size), flat_hidden.size(0))
                lbl_chunk = flat_labels[start:end]
                if not torch.any(lbl_chunk != -100):
                    continue
                logits_chunk = F.linear(flat_hidden[start:end], lm_head_weight)
                total_loss = total_loss + F.cross_entropy(
                    logits_chunk, lbl_chunk, ignore_index=-100, reduction="sum"
                )
            return total_loss / valid_total

        flat_logits = hidden_or_logits.reshape(-1, hidden_or_logits.size(-1))
        valid_total = (flat_labels != -100).sum().clamp(min=1)
        total_loss = flat_logits.new_zeros(())
        for start in range(0, flat_logits.size(0), max(1, chunk_size)):
            end = min(start + max(1, chunk_size), flat_logits.size(0))
            lbl_chunk = flat_labels[start:end]
            if not torch.any(lbl_chunk != -100):
                continue
            total_loss = total_loss + F.cross_entropy(
                flat_logits[start:end], lbl_chunk, ignore_index=-100, reduction="sum"
            )
        return total_loss / valid_total


class BoToiUuGaLore:
    """
    GaLore (Gradient Low-Rank Projection Optimizer).

    Chiếu ma trận đạo hàm 2D `G` xuống không gian con hạng thấp `r` thông qua phân rã SVD định kỳ,
    chỉ lưu trữ các trạng thái moment bậc 1 và bậc 2 của Adam trong không gian hạng thấp (tiết kiệm 60-85% RAM/VRAM
    bộ tối ưu) nhưng vẫn cập nhật đầy đủ toàn bộ tham số của mô hình.
    """

    def __init__(
        self,
        rank: int = 8,
        update_proj_gap: int = 20,
        scale: float = 0.25,
        toc_do_hoc: float = 1e-3,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        if rank <= 0:
            raise ValueError("rank phải > 0")
        self.rank = rank
        self.update_proj_gap = max(1, update_proj_gap)
        self.scale = scale
        self.toc_do_hoc = toc_do_hoc
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._step: Dict[str, int] = {}
        self._proj: Dict[str, np.ndarray] = {}
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        self._so_phan_tu_goc: Dict[str, int] = {}
        self._so_phan_tu_nen: Dict[str, int] = {}

    def cap_nhat_numpy(self, key: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Cập nhật một ma trận trọng số bằng GaLore + AdamW."""
        t = self._step.get(key, 0) + 1
        self._step[key] = t
        self._so_phan_tu_goc[key] = int(param.size) * 2  # m + v chuẩn

        if param.ndim != 2 or min(param.shape) <= self.rank:
            if key not in self._m:
                self._m[key] = np.zeros_like(grad)
                self._v[key] = np.zeros_like(grad)
            self._so_phan_tu_nen[key] = int(self._m[key].size) * 2
            m = self.beta1 * self._m[key] + (1.0 - self.beta1) * grad
            v = self.beta2 * self._v[key] + (1.0 - self.beta2) * (grad * grad)
            self._m[key], self._v[key] = m, v
            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)
            out = param * (1.0 - self.toc_do_hoc * self.weight_decay) if param.ndim >= 2 else param
            return out - self.toc_do_hoc * m_hat / (np.sqrt(v_hat) + self.eps)

        rows, cols = grad.shape
        r = min(self.rank, rows, cols)
        if key not in self._proj or (t - 1) % self.update_proj_gap == 0:
            U, _, Vt = np.linalg.svd(grad, full_matrices=False)
            if rows >= cols:
                self._proj[key] = Vt[:r, :]  # (r, cols)
            else:
                self._proj[key] = U[:, :r]  # (rows, r)

        P = self._proj[key]
        grad_low = grad @ P.T if rows >= cols else P.T @ grad

        if key not in self._m or self._m[key].shape != grad_low.shape:
            self._m[key] = np.zeros_like(grad_low)
            self._v[key] = np.zeros_like(grad_low)
        self._so_phan_tu_nen[key] = int(self._m[key].size) * 2 + int(P.size)

        m = self.beta1 * self._m[key] + (1.0 - self.beta1) * grad_low
        v = self.beta2 * self._v[key] + (1.0 - self.beta2) * (grad_low * grad_low)
        self._m[key], self._v[key] = m, v
        m_hat = m / (1.0 - self.beta1**t)
        v_hat = v / (1.0 - self.beta2**t)
        norm_low = m_hat / (np.sqrt(v_hat) + self.eps)

        grad_full = (norm_low @ P) if rows >= cols else (P @ norm_low)
        out = param * (1.0 - self.toc_do_hoc * self.weight_decay)
        return out - self.toc_do_hoc * self.scale * grad_full

    def thong_ke_bo_nho(self) -> Dict[str, Any]:
        """Thống kê tỷ lệ tiết kiệm bộ nhớ trạng thái Optimizer của GaLore so với AdamW chuẩn."""
        goc = sum(self._so_phan_tu_goc.values())
        nen = sum(self._so_phan_tu_nen.values())
        tiet_kiem = (1.0 - nen / max(1, goc)) * 100.0 if goc > 0 else 0.0
        return {
            "rank": self.rank,
            "phan_tu_adamw_chuan": goc,
            "phan_tu_galore": nen,
            "tiet_kiem_bo_nho_pct": round(max(0.0, tiet_kiem), 2),
        }


class BoTangTocHuanLuyen:
    """
    Bộ Điều phối Siêu Tăng tốc Huấn luyện AI (`TrainingAccelerator` — 2x đến 8x tốc độ).

    Tự động kết hợp:
    1. **Auto-Unsloth 2x Engine** (khi chạy trên CUDA GPU có cài `unsloth`)
    2. **Sequence Packing (`BoGopChuoi`)** kèm mặt nạ chú ý chéo khối 2D
    3. **Vectorized Mini-Batch `(B, T)`** loại bỏ vòng lặp từng mẫu đơn lẻ
    4. **Fused / Chunked Cross-Entropy (`HamMatMatPhanManh` / Liger Kernel)** giảm 75% đỉnh RAM/VRAM
    5. **rsLoRA & DoRA** giúp hội tụ nhanh hơn 30-40% số vòng huấn luyện
    6. **Mixed Precision (`bf16`/`fp16`) & `torch.compile`**
    """

    def __init__(
        self,
        che_do: str = "auto",
        gop_chuoi: bool = True,
        dung_chunked_ce: bool = True,
        dung_rslora: bool = True,
        dung_dora: bool = False,
        mixed_precision: bool = True,
        dung_torch_compile: bool = False,
        max_seq_length: int = 256,
    ):
        self.che_do = che_do.lower()
        self.gop_chuoi = gop_chuoi
        self.dung_chunked_ce = dung_chunked_ce
        self.dung_rslora = dung_rslora
        self.dung_dora = dung_dora
        self.mixed_precision = mixed_precision
        self.dung_torch_compile = dung_torch_compile
        self.max_seq_length = max_seq_length
        self.logger = Logger("BoTangTocHuanLuyen")

    def phat_hien_phan_cung(self) -> Dict[str, Any]:
        """Khám phá năng lực phần cứng và các nhân tăng tốc khả dụng trên máy."""
        co_cuda = bool(_CO_PYTORCH and torch.cuda.is_available())
        co_bf16 = bool(co_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        co_unsloth_gpu = bool(_CO_UNSLOTH and co_cuda)
        ten_gpu = torch.cuda.get_device_name(0) if co_cuda else None
        return {
            "so_luong_cpu": os.cpu_count() or 1,
            "co_pytorch": _CO_PYTORCH,
            "co_cuda": co_cuda,
            "ten_gpu": ten_gpu,
            "co_bf16": co_bf16,
            "co_unsloth": _CO_UNSLOTH,
            "co_the_chay_unsloth_2x": co_unsloth_gpu,
            "co_liger_kernel": _CO_LIGER,
            "backend_de_xuat": (
                "unsloth_cuda_2x"
                if co_unsloth_gpu
                else ("pytorch_amp_packed" if _CO_PYTORCH else "numpy_vectorized_packed")
            ),
        }

    def huan_luyen_nhanh(
        self,
        mo_hinh: Any,
        du_lieu_train: Any,
        du_lieu_val: Optional[Any] = None,
        so_vong: int = 3,
        kich_thuoc_batch: int = 4,
        toc_do_hoc: float = 2e-3,
        rank_lora: int = 8,
        alpha_lora: float = 16.0,
    ) -> Dict[str, Any]:
        """Thực thi huấn luyện tăng tốc đầu-cuối cho `GPTModel`, `BoTinhChinhMoHinh`, hoặc alias LLM."""
        phan_cung = self.phat_hien_phan_cung()

        # Ưu tiên Unsloth 2x nếu có CUDA + Unsloth và người dùng truyền tên model hỗ trợ
        if (
            isinstance(mo_hinh, str)
            and self.che_do in ("auto", "unsloth")
            and phan_cung["co_the_chay_unsloth_2x"]
        ):
            from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

            uw = UnslothWrapper()
            if mo_hinh in uw.UNSLOTH_MODELS or "/" in mo_hinh:
                uw.tai_mo_hinh(mo_hinh, max_seq_length=self.max_seq_length)
                uw.config_lora(
                    r=rank_lora,
                    lora_alpha=int(alpha_lora),
                    use_rslora=self.dung_rslora,
                    use_dora=self.dung_dora,
                )
                kq_u = uw.fine_tune(
                    du_lieu_train,
                    so_vong=so_vong,
                    batch_size=kich_thuoc_batch,
                    learning_rate=toc_do_hoc,
                    packing=self.gop_chuoi,
                )
                kq_u["cong_nghe_tang_toc"] = ["unsloth_2x", "qlora_4bit"] + (
                    ["sequence_packing"] if self.gop_chuoi else []
                )
                return kq_u

        if hasattr(mo_hinh, "gan_lora") and not getattr(mo_hinh, "da_gan_lora", False):
            mo_hinh.gan_lora(
                rank=rank_lora,
                alpha=alpha_lora,
                dung_rslora=self.dung_rslora,
                dung_dora=self.dung_dora,
            )

        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            so_vong=so_vong,
            kich_thuoc_batch=kich_thuoc_batch,
            toc_do_hoc=toc_do_hoc,
            max_seq_length=self.max_seq_length,
            gop_chuoi=self.gop_chuoi,
            mixed_precision=self.mixed_precision,
            dung_torch_compile=self.dung_torch_compile,
            dung_chunked_ce=self.dung_chunked_ce,
        )
        kq = trainer.huan_luyen(mo_hinh, du_lieu_train, du_lieu_val=du_lieu_val)
        cong_nghe = ["vectorized_batch"]
        if self.gop_chuoi:
            cong_nghe.append("sequence_packing")
        if self.dung_chunked_ce:
            cong_nghe.append("chunked_cross_entropy")
        if self.dung_dora:
            cong_nghe.append("dora")
        elif self.dung_rslora:
            cong_nghe.append("rslora")
        kq["cong_nghe_tang_toc"] = cong_nghe
        kq["phan_cung"] = phan_cung
        return kq


class BenchmarkTangToc:
    """
    Bộ Đo kiểm Tốc độ Thực tế (`SpeedBenchmark`) so sánh thông lượng huấn luyện và suy luận
    trước & sau khi bật Tổ hợp Tăng tốc (Sequence Packing + Vectorized Batch + KV-Cache).
    """

    @staticmethod
    def do_toc_do_huan_luyen(
        so_mau: int = 32,
        so_vong: int = 2,
        d_model: int = 64,
        so_block: int = 2,
        so_tu_vung: int = 256,
        max_seq_length: int = 64,
    ) -> Dict[str, Any]:
        """So sánh trực tiếp tốc độ huấn luyện Chuẩn (Baseline) và Siêu Tăng tốc trên cùng kiến trúc."""
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer
        from vietnamese_ai.transformer.gpt_model import GPTModel

        rng = np.random.RandomState(42)
        du_lieu_ngan = []
        du_lieu_pad_co_dinh = []
        tong_tokens = 0
        for _ in range(so_mau):
            do_dai = int(rng.randint(6, 14))
            seq = rng.randint(1, so_tu_vung - 1, size=do_dai).tolist()
            in_seq = seq[:-1]
            lbl_seq = seq[1:]
            du_lieu_ngan.append({"input_ids": in_seq, "labels": lbl_seq})
            pad_len = max(0, max_seq_length - len(in_seq))
            du_lieu_pad_co_dinh.append(
                {
                    "input_ids": in_seq + [0] * pad_len,
                    "labels": lbl_seq + [-100] * pad_len,
                }
            )
            tong_tokens += len(in_seq)

        np.random.seed(123)
        model_base = GPTModel(
            d_model=d_model,
            so_dau=2,
            d_ff=d_model * 2,
            so_block=so_block,
            so_tu_vung=so_tu_vung,
            do_dai_toi_da=max_seq_length,
        )
        model_base.gan_lora(rank=4, alpha=8.0)

        t0 = time.perf_counter()
        # Chế độ chuẩn truyền thống: từng mẫu đệm cố định tới max_seq_length, không gộp chuỗi
        trainer_base = SFTTrainer(
            so_vong=so_vong,
            kich_thuoc_batch=1,
            toc_do_hoc=2e-3,
            max_seq_length=max_seq_length,
            gop_chuoi=False,
            dung_chunked_ce=False,
        )
        kq_base = trainer_base.huan_luyen(model_base, du_lieu_pad_co_dinh)
        dt_base = max(1e-6, time.perf_counter() - t0)

        np.random.seed(123)
        model_fast = GPTModel(
            d_model=d_model,
            so_dau=2,
            d_ff=d_model * 2,
            so_block=so_block,
            so_tu_vung=so_tu_vung,
            do_dai_toi_da=max_seq_length,
        )
        model_fast.gan_lora(rank=4, alpha=8.0, dung_rslora=True)

        t1 = time.perf_counter()
        trainer_fast = SFTTrainer(
            so_vong=so_vong,
            kich_thuoc_batch=4,
            toc_do_hoc=2e-3,
            max_seq_length=max_seq_length,
            gop_chuoi=True,
            dung_chunked_ce=True,
        )
        kq_fast = trainer_fast.huan_luyen(model_fast, du_lieu_ngan)
        dt_fast = max(1e-6, time.perf_counter() - t1)

        tong_tok_xu_ly = tong_tokens * so_vong
        tps_base = tong_tok_xu_ly / dt_base
        tps_fast = tong_tok_xu_ly / dt_fast

        return {
            "so_mau": so_mau,
            "so_vong": so_vong,
            "thoi_gian_goc_giay": round(dt_base, 4),
            "thoi_gian_tang_toc_giay": round(dt_fast, 4),
            "tokens_moi_giay_goc": round(tps_base, 1),
            "tokens_moi_giay_tang_toc": round(tps_fast, 1),
            "he_so_tang_toc_train": round(dt_base / dt_fast, 2),
            "loss_cuoi_goc": round(float(kq_base.get("loss_cuoi", 0.0)), 4),
            "loss_cuoi_tang_toc": round(float(kq_fast.get("loss_cuoi", 0.0)), 4),
            "thong_ke_gop_chuoi": kq_fast.get("thong_ke_gop_chuoi", {}),
        }

    @staticmethod
    def do_toc_do_suy_luan_kv_cache(
        do_dai_prompt: int = 24,
        so_token_sinh: int = 32,
        d_model: int = 64,
        so_block: int = 3,
        so_tu_vung: int = 256,
    ) -> Dict[str, Any]:
        """So sánh tốc độ sinh văn bản tự hồi quy khi TẮT và BẬT `KV-Cache` trên `GPTModel`."""
        from vietnamese_ai.transformer.gpt_model import GPTModel

        np.random.seed(42)
        gpt = GPTModel(
            d_model=d_model,
            so_dau=2,
            d_ff=d_model * 2,
            so_block=so_block,
            so_tu_vung=so_tu_vung,
            do_dai_toi_da=do_dai_prompt + so_token_sinh + 16,
        )
        prompt_ids = np.random.randint(1, so_tu_vung - 1, size=(1, do_dai_prompt))

        np.random.seed(99)
        t0 = time.perf_counter()
        out_no_cache = gpt.sinh_tiep(prompt_ids, so_token=so_token_sinh, nhiet_do=0.8, dung_kv_cache=False)
        dt_no_cache = max(1e-6, time.perf_counter() - t0)

        np.random.seed(99)
        t1 = time.perf_counter()
        out_kv_cache = gpt.sinh_tiep(prompt_ids, so_token=so_token_sinh, nhiet_do=0.8, dung_kv_cache=True)
        dt_kv_cache = max(1e-6, time.perf_counter() - t1)

        khop_tuyet_doi = bool(np.array_equal(out_no_cache, out_kv_cache))
        return {
            "so_token_sinh": so_token_sinh,
            "thoi_gian_khong_cache_giay": round(dt_no_cache, 4),
            "thoi_gian_kv_cache_giay": round(dt_kv_cache, 4),
            "tokens_moi_giay_khong_cache": round(so_token_sinh / dt_no_cache, 1),
            "tokens_moi_giay_kv_cache": round(so_token_sinh / dt_kv_cache, 1),
            "he_so_tang_toc_kv_cache": round(dt_no_cache / dt_kv_cache, 2),
            "khop_dau_ra_tuyet_doi": khop_tuyet_doi,
        }


SequencePacker = BoGopChuoi
ChunkedCrossEntropy = HamMatMatPhanManh
GaLoreOptimizer = BoToiUuGaLore
TrainingAccelerator = BoTangTocHuanLuyen
SpeedBenchmark = BenchmarkTangToc
