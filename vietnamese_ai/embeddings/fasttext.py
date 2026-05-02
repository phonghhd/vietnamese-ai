"""FastText tiếng Việt - Character n-gram embeddings."""

from typing import Dict, List, Optional

import numpy as np

from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class FastTextTiengViet(Word2VecTiengViet):
    """
    FastText tiếng Việt - mở rộng Word2Vec với character n-grams.

    Đặc biệt hữu ích cho tiếng Việt vì:
    - Xử lý được từ mới (OOV) bằng cách ghép n-gram ký tự
    - Bắt được cấu trúc ngữ âm tiếng Việt

    Sử dụng:
        >>> ft = FastTextTiengViet(kich_thuoc=100)
        >>> ft.huan_luyen(cac_van_ban, so_vong=5)
        >>> vector = ft.lay_vector("học")  # luôn có vector, ngay cả từ mới
        >>> ft.lay_vector("từ_mới_chưa_từng_thấy")  # vẫn trả về vector
    """

    def __init__(
        self,
        kich_thuoc: int = 100,
        cua_so: int = 5,
        che_do: str = "skipgram",
        toc_do_hoc: float = 0.025,
        toi_thieu_dem: int = 2,
        so_am: int = 5,
        ngram_min: int = 3,
        ngram_max: int = 6,
    ):
        super().__init__(
            kich_thuoc=kich_thuoc,
            cua_so=cua_so,
            che_do=che_do,
            toc_do_hoc=toc_do_hoc,
            toi_thieu_dem=toi_thieu_dem,
            so_am=so_am,
        )
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self._ngram_dien: Dict[str, int] = {}
        self._W_ngram: Optional[np.ndarray] = None
        self.logger = Logger("FastText")

    @staticmethod
    def _tao_ngram(tu: str, ngram_min: int, ngram_max: int) -> List[str]:
        """Tạo character n-grams cho một từ."""
        tu_bao = f"<{tu}>"
        ngrams = []
        for n in range(ngram_min, ngram_max + 1):
            for i in range(len(tu_bao) - n + 1):
                ngrams.append(tu_bao[i:i + n])
        return ngrams if ngrams else [tu_bao]

    def _tao_ngram_dien(self, cac_tu: List[str]) -> None:
        """Tạo từ điển n-gram."""
        self._ngram_dien = {}
        idx = 0
        for tu in cac_tu:
            for ng in self._tao_ngram(tu, self.ngram_min, self.ngram_max):
                if ng not in self._ngram_dien:
                    self._ngram_dien[ng] = idx
                    idx += 1

    def _khoi_tao_trong_so(self) -> None:
        """Khởi tạo ma trọng số cho cả từ và n-gram."""
        super()._khoi_tao_trong_so()
        if self._ngram_dien:
            self._W_ngram = np.random.uniform(
                -0.5, 0.5, (len(self._ngram_dien), self.kich_thuoc)
            ) / self.kich_thuoc

    def lay_vector(self, tu: str) -> np.ndarray:
        """
        Lấy vector của từ bằng cách cộng trung bình n-gram vectors.

        Đặc biệt: luôn trả về vector ngay cả với từ mới (OOV).
        """
        if self._W_ngram is None:
            return super().lay_vector(tu) or np.zeros(self.kich_thuoc)

        ngrams = self._tao_ngram(tu, self.ngram_min, self.ngram_max)
        vectors = []
        for ng in ngrams:
            if ng in self._ngram_dien:
                vectors.append(self._W_ngram[self._ngram_dien[ng]])

        # Nếu từ có trong từ điển, kết hợp với word vector
        if tu in self._tu_dien and self._W_in is not None:
            vectors.append(self._W_in[self._tu_dien[tu]])

        if not vectors:
            return np.zeros(self.kich_thuoc)
        return np.mean(vectors, axis=0)

    def lay_vector_van_ban(self, text: str) -> np.ndarray:
        """Lấy vector văn bản bằng trung bình vector các từ."""
        xl = XuLyVanBan()
        cac_tu = xl.tach_tu(text)
        vectors = [self.lay_vector(tu) for tu in cac_tu]
        vectors = [v for v in vectors if np.any(v)]
        if not vectors:
            return np.zeros(self.kich_thuoc)
        return np.mean(vectors, axis=0)
