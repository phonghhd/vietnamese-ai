"""PhoBERTWrapper - Wrapper cho PhoBERT tiếng Việt."""

import hashlib
from typing import List

import numpy as np

from vietnamese_ai.utils.logger import Logger


class PhoBERTWrapper:
    """
    Wrapper cho PhoBERT - mô hình BERT pre-trained cho tiếng Việt.

    Khi chưa cài `transformers`/`torch` hoặc `che_do_mo_phong=True`,
    dùng embedding hash xác định (768 chiều) để phát triển/test offline.

    Yêu cầu (chế độ thật): pip install transformers torch

    Sử dụng:
        >>> phobert = PhoBERTWrapper(che_do_mo_phong=True)
        >>> vectors = phobert.ma_hoa(["Học máy rất hay", "AI rất thú vị"])
        >>> tu_giong = phobert.tim_tu_giong("học máy", ["học máy AI", "ẩm thực"], top_n=2)
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        che_do_mo_phong: bool = False,
        kich_thuoc_an: int = 768,
    ):
        self.model_name = model_name
        self.che_do_mo_phong = che_do_mo_phong
        self.kich_thuoc_an = kich_thuoc_an
        self.logger = Logger("PhoBERT")
        self._model = None
        self._tokenizer = None
        self._da_tai = False

    def tai_mo_hinh(self) -> None:
        """Tải mô hình PhoBERT hoặc kích hoạt chế độ mô phỏng."""
        if self.che_do_mo_phong:
            self._da_tai = True
            self.logger.info("PhoBERT chạy chế độ mô phỏng (hash embedding)")
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            self.logger.info(f"Đang tải mô hình: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._da_tai = True
            self.logger.info("Tải mô hình hoàn tất")
        except ImportError:
            self.che_do_mo_phong = True
            self._da_tai = True
            self.logger.warning(
                "Thiếu transformers/torch — chuyển sang chế độ mô phỏng. "
                "Cài: pip install transformers torch"
            )

    def _ma_hoa_mo_phong(self, cac_van_ban: List[str]) -> np.ndarray:
        import math
        import re
        from collections import Counter

        dim = self.kich_thuoc_an
        rows = []
        for text in cac_van_ban:
            vec = np.zeros(dim, dtype=np.float64)
            clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
            if not clean:
                clean = text.lower().strip()

            words = clean.split()
            features: Counter = Counter()
            weights: dict = {}

            for w in words:
                k = f"w:{w}"
                features[k] += 1
                weights[k] = 2.5

            for i in range(len(words) - 1):
                k = f"bi:{words[i]}_{words[i + 1]}"
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
                features[f"raw:{text}"] = 1
                weights[f"raw:{text}"] = 1.0

            for feat, count in features.items():
                tf = 1.0 + math.log(count)
                w = weights.get(feat, 1.0) * tf
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
                h1 = int.from_bytes(digest[0:4], "little")
                h2 = int.from_bytes(digest[4:8], "little")
                h3 = int.from_bytes(digest[8:12], "little")
                s1 = 1.0 if (digest[12] & 1) == 0 else -1.0
                s2 = 1.0 if (digest[13] & 1) == 0 else -1.0
                s3 = 1.0 if (digest[14] & 1) == 0 else -1.0
                vec[h1 % dim] += s1 * w
                vec[h2 % dim] += s2 * (w * 0.7)
                vec[h3 % dim] += s3 * (w * 0.5)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        return np.vstack(rows)

    def ma_hoa(self, cac_van_ban: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Mã hóa văn bản thành vectors (CLS token hoặc mô phỏng).

        Args:
            cac_van_ban: Danh sách văn bản
            batch_size: Kích thước batch

        Returns:
            Ma trận vectors (n_van_ban, hidden_size)
        """
        if not self._da_tai:
            self.tai_mo_hinh()

        if self.che_do_mo_phong or self._model is None:
            return self._ma_hoa_mo_phong(cac_van_ban)

        import torch

        tat_ca_vectors = []

        for i in range(0, len(cac_van_ban), batch_size):
            batch = cac_van_ban[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
            tat_ca_vectors.append(cls_vectors)

        return np.vstack(tat_ca_vectors)

    def ma_hoa_van_ban(self, text: str) -> np.ndarray:
        """Mã hóa một văn bản."""
        return self.ma_hoa([text])[0]

    def do_tuong_dong(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng cosine giữa hai câu văn bản tiếng Việt."""
        vecs = self.ma_hoa([text1, text2])
        v1 = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-10)
        v2 = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-10)
        return float(np.dot(v1, v2))

    def tim_tu_giong(self, text: str, cac_van_ban: List[str], top_n: int = 5) -> List:
        """Tìm văn bản giống nhất."""
        vec_query = self.ma_hoa_van_ban(text)
        vec_all = self.ma_hoa(cac_van_ban)

        query_norm = vec_query / (np.linalg.norm(vec_query) + 1e-10)
        all_norm = vec_all / (np.linalg.norm(vec_all, axis=1, keepdims=True) + 1e-10)
        scores = all_norm @ query_norm

        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(cac_van_ban[idx], float(scores[idx])) for idx in top_idx]

    @property
    def co_mo_hinh(self) -> bool:
        return self._da_tai
