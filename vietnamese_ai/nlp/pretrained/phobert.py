"""PhoBERTWrapper - Tích hợp chuyên sâu PhoBERT tiếng Việt với tách từ tự động, gộp vector và đầu phân loại."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.utils.logger import Logger


class PhoBERTWrapper:
    """
    Tích hợp PhoBERT (`vinai/phobert-base`, `vinai/phobert-base-v2`) cho tiếng Việt.

    Tính năng nổi bật:
    - Tự động tách từ ghép tiếng Việt (`trí_tuệ_nhân_tạo`) qua `BoTachTuTiengVietSieuToc` theo đúng
      yêu cầu tiền xử lý BPE của VinAI PhoBERT.
    - Hỗ trợ chọn chiến lược pooling (`mean`, `cls`, `max`), chọn thiết bị (`auto`, `cpu`, `cuda`)
      và chuẩn hóa L2 vector đầu ra.
    - Cung cấp đầu phân loại tuyến tính chỉnh quy L2 (`huan_luyen_dau_phan_loai` / `du_doan_phan_loai`)
      ngay trên biểu diễn PhoBERT (hoạt động cả chế độ Transformer thật và chế độ offline nhẹ).
    - Khi chưa cài `transformers`/`torch` hoặc `che_do_mo_phong=True`, tự động dùng biểu diễn hash
      xác định trên token đã tách từ tiếng Việt (768 chiều).
    """

    PHUONG_PHAP_GOP_HOP_LE = ("mean", "cls", "max")

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        che_do_mo_phong: bool = False,
        kich_thuoc_an: int = 768,
        tu_dong_tach_tu: bool = True,
        phuong_phap_gop: str = "mean",
        thiet_bi: str = "auto",
        do_dai_toi_da: int = 256,
    ) -> None:
        if phuong_phap_gop not in self.PHUONG_PHAP_GOP_HOP_LE:
            raise ValueError(
                f"phuong_phap_gop phải thuộc {self.PHUONG_PHAP_GOP_HOP_LE}, nhận được '{phuong_phap_gop}'"
            )
        self.model_name = model_name
        self.che_do_mo_phong = che_do_mo_phong
        self.kich_thuoc_an = kich_thuoc_an
        self.tu_dong_tach_tu = tu_dong_tach_tu
        self.phuong_phap_gop = phuong_phap_gop
        self.thiet_bi = thiet_bi
        self.do_dai_toi_da = do_dai_toi_da
        self.logger = Logger("PhoBERT")
        self._vi_tokenizer = BoTachTuTiengVietSieuToc()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._da_tai = False
        self._trong_so_phan_loai: Optional[np.ndarray] = None
        self._ds_nhan: List[str] = []

    def tien_xu_ly_tieng_viet(self, van_ban: str) -> str:
        """Chuẩn hóa dấu thanh và tách từ ghép có gạch nối `_` chuẩn định dạng đầu vào PhoBERT."""
        if not van_ban:
            return ""
        if not self.tu_dong_tach_tu:
            return self._vi_tokenizer.chuan_hoa_dau_thanh(van_ban)
        tokens = self._vi_tokenizer.tach_tu(van_ban, noi_tu_ghep=True, ky_tu_noi="_")
        return " ".join(tokens)

    def tai_mo_hinh(self) -> None:
        """Tải mô hình PhoBERT từ HuggingFace hoặc kích hoạt chế độ mô phỏng offline."""
        if self.che_do_mo_phong:
            self._da_tai = True
            self.logger.info("PhoBERT chạy chế độ mô phỏng offline (Vietnamese segmented hash embedding)")
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.thiet_bi == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.thiet_bi

            self.logger.info(f"Đang tải mô hình: {self.model_name} trên thiết bị {self._device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            self._da_tai = True
            self.logger.info("Tải mô hình PhoBERT hoàn tất")
        except ImportError:
            self.che_do_mo_phong = True
            self._da_tai = True
            self.logger.warning(
                "Thiếu transformers/torch — chuyển sang chế độ mô phỏng. "
                "Cài đặt: pip install vietnamese-ai[dl]"
            )

    def _ma_hoa_mo_phong(self, cac_van_ban: List[str]) -> np.ndarray:
        """Mã hóa vector xác định 768 chiều dựa trên từ ghép tiếng Việt, bigram và ký tự n-gram."""
        dim = self.kich_thuoc_an
        rows: List[np.ndarray] = []
        for raw_text in cac_van_ban:
            text = self.tien_xu_ly_tieng_viet(raw_text)
            vec = np.zeros(dim, dtype=np.float64)
            clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
            if not clean:
                clean = raw_text.lower().strip()

            words = clean.split()
            features: Counter[str] = Counter()
            weights: Dict[str, float] = {}

            for w in words:
                k = f"w:{w}"
                features[k] += 1
                weights[k] = 2.8 if "_" in w else 2.4

            for i in range(len(words) - 1):
                k = f"bi:{words[i]}|{words[i + 1]}"
                features[k] += 1
                weights[k] = 1.8

            padded = f" {clean} "
            for n in (3, 4, 5):
                for i in range(max(0, len(padded) - n + 1)):
                    ng = padded[i : i + n]
                    if not ng.isspace():
                        k = f"c{n}:{ng}"
                        features[k] += 1
                        weights[k] = 0.8

            if not features:
                features[f"raw:{raw_text}"] = 1
                weights[f"raw:{raw_text}"] = 1.0

            for feat, count in features.items():
                tf = 1.0 + math.log(count)
                trong_so = weights.get(feat, 1.0) * tf
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
                h1 = int.from_bytes(digest[0:4], "little")
                h2 = int.from_bytes(digest[4:8], "little")
                h3 = int.from_bytes(digest[8:12], "little")
                s1 = 1.0 if (digest[12] & 1) == 0 else -1.0
                s2 = 1.0 if (digest[13] & 1) == 0 else -1.0
                s3 = 1.0 if (digest[14] & 1) == 0 else -1.0
                vec[h1 % dim] += s1 * trong_so
                vec[h2 % dim] += s2 * (trong_so * 0.7)
                vec[h3 % dim] += s3 * (trong_so * 0.5)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        return np.vstack(rows)

    def ma_hoa(self, cac_van_ban: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Mã hóa danh sách văn bản tiếng Việt thành ma trận vector biểu diễn.

        Args:
            cac_van_ban: Danh sách văn bản đầu vào.
            batch_size: Kích thước mỗi lô xử lý.

        Returns:
            Ma trận vectors có hình dạng `(n_van_ban, hidden_size)`.
        """
        if not cac_van_ban:
            return np.zeros((0, self.kich_thuoc_an), dtype=np.float64)

        if not self._da_tai:
            self.tai_mo_hinh()

        if self.che_do_mo_phong or self._model is None:
            return self._ma_hoa_mo_phong(cac_van_ban)

        import torch

        van_ban_chuan = [self.tien_xu_ly_tieng_viet(vb) for vb in cac_van_ban]
        tat_ca_vectors: List[np.ndarray] = []

        for i in range(0, len(van_ban_chuan), batch_size):
            batch = van_ban_chuan[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.do_dai_toi_da,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden = outputs.last_hidden_state
            if self.phuong_phap_gop == "cls":
                pooled = hidden[:, 0, :]
            elif self.phuong_phap_gop == "max":
                mask = inputs["attention_mask"].unsqueeze(-1).bool()
                masked = hidden.masked_fill(~mask, -1e9)
                pooled = masked.max(dim=1).values
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = (hidden * mask).sum(dim=1) / denom

            tat_ca_vectors.append(pooled.detach().cpu().numpy())

        return np.vstack(tat_ca_vectors)

    def ma_hoa_van_ban(self, text: str) -> np.ndarray:
        """Mã hóa một văn bản đơn lẻ thành vector 1D."""
        return self.ma_hoa([text])[0]

    def do_tuong_dong(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng cosine giữa hai câu văn bản tiếng Việt."""
        vecs = self.ma_hoa([text1, text2])
        v1 = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-10)
        v2 = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-10)
        return float(np.dot(v1, v2))

    def tim_tu_giong(
        self,
        text: str,
        cac_van_ban: List[str],
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Tìm `top_n` văn bản tương đồng ngữ nghĩa cao nhất với câu truy vấn."""
        vec_query = self.ma_hoa_van_ban(text)
        vec_all = self.ma_hoa(cac_van_ban)

        query_norm = vec_query / (np.linalg.norm(vec_query) + 1e-10)
        all_norm = vec_all / (np.linalg.norm(vec_all, axis=1, keepdims=True) + 1e-10)
        scores = all_norm @ query_norm

        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(cac_van_ban[idx], float(scores[idx])) for idx in top_idx]

    def huan_luyen_dau_phan_loai(
        self,
        ds_van_ban: List[str],
        ds_nhan: List[str],
        he_so_chinh_quy: float = 1.0,
    ) -> Dict[str, Any]:
        """Huấn luyện đầu phân loại tuyến tính chỉnh quy L2 (Ridge Head) trên biểu diễn PhoBERT."""
        if not ds_van_ban or len(ds_van_ban) != len(ds_nhan):
            raise ValueError("Danh sách văn bản và danh sách nhãn phải không rỗng và có cùng độ dài")

        x_mat = self.ma_hoa(ds_van_ban)
        n_samples = x_mat.shape[0]
        x_aug = np.hstack([x_mat, np.ones((n_samples, 1), dtype=np.float64)])

        nhan_duy_nhat = sorted(set(ds_nhan))
        self._ds_nhan = nhan_duy_nhat
        idx_map = {lbl: i for i, lbl in enumerate(nhan_duy_nhat)}
        y_onehot = np.zeros((n_samples, len(nhan_duy_nhat)), dtype=np.float64)
        for row_i, lbl in enumerate(ds_nhan):
            y_onehot[row_i, idx_map[lbl]] = 1.0

        reg = he_so_chinh_quy * np.eye(x_aug.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        self._trong_so_phan_loai = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_onehot)

        logits = x_aug @ self._trong_so_phan_loai
        preds = [self._ds_nhan[int(i)] for i in np.argmax(logits, axis=1)]
        do_chinh_xac = sum(1 for p, y in zip(preds, ds_nhan) if p == y) / n_samples

        return {
            "so_mau": n_samples,
            "so_lop": len(self._ds_nhan),
            "danh_sach_nhan": list(self._ds_nhan),
            "do_chinh_xac_huan_luyen": float(do_chinh_xac),
            "che_do_mo_phong": self.che_do_mo_phong,
        }

    def du_doan_phan_loai(self, ds_van_ban: List[str]) -> List[Dict[str, Any]]:
        """Dự đoán nhãn phân loại và phân bố xác suất softmax cho danh sách văn bản tiếng Việt."""
        if self._trong_so_phan_loai is None or not self._ds_nhan:
            raise RuntimeError("Chưa huấn luyện đầu phân loại. Hãy gọi `huan_luyen_dau_phan_loai` trước.")
        if not ds_van_ban:
            return []

        x_mat = self.ma_hoa(ds_van_ban)
        x_aug = np.hstack([x_mat, np.ones((x_mat.shape[0], 1), dtype=np.float64)])
        logits = x_aug @ self._trong_so_phan_loai
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        ket_qua: List[Dict[str, Any]] = []
        for i in range(len(ds_van_ban)):
            best_idx = int(np.argmax(probs[i]))
            ket_qua.append(
                {
                    "nhan": self._ds_nhan[best_idx],
                    "do_tin_cay": float(probs[i, best_idx]),
                    "xac_suat": {self._ds_nhan[j]: float(probs[i, j]) for j in range(len(self._ds_nhan))},
                }
            )
        return ket_qua

    @property
    def co_mo_hinh(self) -> bool:
        """Cho biết mô hình hoặc chế độ mô phỏng đã được nạp hay chưa."""
        return self._da_tai
