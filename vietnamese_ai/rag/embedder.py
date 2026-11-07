"""BoNhungVanBan - embedding văn bản cho RAG (sentence-transformers hoặc fallback)."""

import hashlib
from typing import List, Optional, Union

import numpy as np

from vietnamese_ai.utils.logger import Logger


class BoNhungVanBan:
    """
    Bộ nhúng văn bản tiếng Việt cho RAG.

    Ưu tiên `sentence-transformers` khi có; không thì dùng hash embedding xác định
    (đủ cho demo/test, không thay semantic model thật).

    Sử dụng:
        >>> embedder = BoNhungVanBan()
        >>> vec = embedder.nhung("Học máy là gì?")
        >>> batch = embedder.nhung(["câu 1", "câu 2"])
    """

    MODEL_MAC_DINH = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        ten_model: str = MODEL_MAC_DINH,
        kich_thuoc: int = 384,
        bat_buoc_st: bool = False,
        phuong_phap: Optional[str] = None,
    ):
        self.ten_model = ten_model
        self.kich_thuoc = kich_thuoc
        self.bat_buoc_st = bat_buoc_st
        self.phuong_phap = phuong_phap
        self.logger = Logger("BoNhungVanBan")
        self._model = None
        self._che_do = "chua_tai"
        self._cache_nhung: dict = {}
        self._max_cache = 2048
        if phuong_phap == "hash":
            self._che_do = "hash"
        else:
            self._tai_model()

    def _tai_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.ten_model)
            dim = int(self._model.get_sentence_embedding_dimension())
            self.kich_thuoc = dim
            self._che_do = "sentence_transformers"
            self.logger.info(f"Đã tải sentence-transformers: {self.ten_model} ({dim} chiều)")
        except ImportError:
            if self.bat_buoc_st:
                raise ImportError(
                    "Cần cài sentence-transformers: pip install sentence-transformers"
                )
            self._che_do = "hash_fallback"
            self.logger.warning(
                "Thiếu sentence-transformers — dùng hash embedding. "
                "Cài: pip install 'vietnamese-ai[rag]'"
            )
        except Exception as e:
            if self.bat_buoc_st:
                raise
            self._che_do = "hash_fallback"
            self.logger.warning(f"Không tải được ST model ({e}) — dùng hash fallback")

    @property
    def che_do(self) -> str:
        return self._che_do

    def _nhung_hash(self, text: str) -> np.ndarray:
        """
        Nhúng ngữ nghĩa từ vựng & n-gram tiếng Việt bằng Signed Multi-Probe Feature Hashing
        kết hợp Sublinear TF weighting. Câu có chung từ vựng/cụm từ sẽ có cosine similarity cao.
        """
        import math
        import re
        from collections import Counter

        dim = self.kich_thuoc
        cache_key = (dim, text)
        if cache_key in self._cache_nhung:
            return self._cache_nhung[cache_key].copy()

        vec = np.zeros(dim, dtype=np.float32)
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

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        if len(self._cache_nhung) >= self._max_cache:
            self._cache_nhung.clear()
        self._cache_nhung[cache_key] = vec.copy()
        return vec

    def nhung(self, van_ban: Union[str, List[str]]) -> np.ndarray:
        """Nhúng một hoặc nhiều câu văn bản thành vector."""
        la_don = isinstance(van_ban, str)
        texts: List[str] = [van_ban] if la_don else list(van_ban)

        if self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            out = np.asarray(vectors, dtype=np.float32)
        else:
            out = np.vstack([self._nhung_hash(t) for t in texts])

        return out[0] if la_don else out

    def nhung_mot(self, van_ban: str) -> np.ndarray:
        """Nhúng một câu văn bản đơn lẻ thành vector 1 chiều."""
        return self.nhung(van_ban)

    def __call__(self, van_ban: str) -> np.ndarray:
        """Callable cho `ham_embed` của RAGPipeline / TrichXuat."""
        return self.nhung(van_ban)


def tao_ham_embed(
    ten_model: Optional[str] = None,
    bat_buoc_st: bool = False,
) -> BoNhungVanBan:
    """Factory tạo BoNhungVanBan dùng làm ham_embed."""
    kwargs = {"bat_buoc_st": bat_buoc_st}
    if ten_model:
        kwargs["ten_model"] = ten_model
    return BoNhungVanBan(**kwargs)
