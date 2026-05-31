from typing import List, Union

import numpy as np


class ImageEmbedder:
    """
    Lớp cơ sở cho việc trích xuất đặc trưng (embedding) từ hình ảnh.
    Sử dụng cho Multi-modal RAG.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self._da_tai_model = False

    def _tai_model(self):
        """Khởi tạo mô hình nhúng ảnh (Ví dụ: CLIP)."""
        pass

    def nhung_hinh_anh(self, image_paths: Union[str, List[str]]) -> List[List[float]]:
        """
        Nhúng một hoặc nhiều hình ảnh thành vector.
        Hàm này là mockup interface. Ở môi trường production cần cài đặt transformers/CLIP.
        """
        if not self._da_tai_model:
            self._tai_model()
            self._da_tai_model = True

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Mock vector (512 chiều)
        vectors = []
        for _ in image_paths:
            # Sinh vector ngẫu nhiên để mô phỏng
            vector = np.random.rand(512).tolist()
            vectors.append(vector)

        return vectors

    def nhung_van_ban(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Nhúng một hoặc nhiều đoạn văn bản thành vector (cùng không gian với ảnh).
        Giả lập mô hình CLIP Text Encoder.
        """
        if not self._da_tai_model:
            self._tai_model()
            self._da_tai_model = True

        if isinstance(texts, str):
            texts = [texts]

        # Mock vector (512 chiều)
        vectors = []
        for _ in texts:
            # Sinh vector ngẫu nhiên để mô phỏng
            vector = np.random.rand(512).tolist()
            vectors.append(vector)

        return vectors
