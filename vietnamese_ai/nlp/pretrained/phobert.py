"""PhoBERTWrapper - Wrapper cho PhoBERT tiếng Việt."""

from typing import List

import numpy as np

from vietnamese_ai.utils.logger import Logger


class PhoBERTWrapper:
    """
    Wrapper cho PhoBERT - mô hình BERT pre-trained cho tiếng Việt.

    Yêu cầu: pip install transformers torch

    Sử dụng:
        >>> phobert = PhoBERTWrapper()
        >>> vectors = phobert.ma_hoa(["Học máy rất hay", "AI rất thú vị"])
        >>> tu_giong = phobert.tim_tu_giong("học máy", top_n=5)
    """

    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.model_name = model_name
        self.logger = Logger("PhoBERT")
        self._model = None
        self._tokenizer = None
        self._da_tai = False

    def tai_mo_hinh(self) -> None:
        """Tải mô hình PhoBERT."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.logger.info(f"Đang tải mô hình: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._da_tai = True
            self.logger.info("Tải mô hình hoàn tất")
        except ImportError:
            raise ImportError(
                "Cần cài đặt transformers và torch: "
                "pip install transformers torch"
            )

    def ma_hoa(self, cac_van_ban: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Mã hóa văn bản thành vectors (CLS token).

        Args:
            cac_van_ban: Danh sách văn bản
            batch_size: Kích thước batch

        Returns:
            Ma trận vectors (n_van_ban, hidden_size)
        """
        if not self._da_tai:
            self.tai_mo_hinh()

        import torch

        tat_ca_vectors = []

        for i in range(0, len(cac_van_ban), batch_size):
            batch = cac_van_ban[i:i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
            tat_ca_vectors.append(cls_vectors)

        return np.vstack(tat_ca_vectors)

    def ma_hoa_van_ban(self, text: str) -> np.ndarray:
        """Mã hóa một văn bản."""
        return self.ma_hoa([text])[0]

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
