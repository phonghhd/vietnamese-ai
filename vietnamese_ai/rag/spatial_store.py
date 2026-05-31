import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.rag.vector_store import CSDLVector


class SpatialVectorStore(CSDLVector):
    """
    Cơ sở dữ liệu Vector không gian (Spatial RAG).
    Cho phép tìm kiếm lai (Hybrid Search) kết hợp giữa Ngữ nghĩa (Semantic)
    và Khoảng cách vật lý 3D (Euclidean Distance).
    """

    def chen_khong_gian(
        self,
        ma: str,
        vector: np.ndarray,
        toa_do: Tuple[float, float, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Chèn vector kèm tọa độ 3D (x, y, z).
        """
        meta = metadata or {}
        meta["toa_do"] = toa_do
        self.chen(ma, vector, meta)

    def _tinh_khoang_cach_euclide(
        self, coord1: Tuple[float, float, float], coord2: Tuple[float, float, float]
    ) -> float:
        return math.sqrt(
            (coord1[0] - coord2[0]) ** 2
            + (coord1[1] - coord2[1]) ** 2
            + (coord1[2] - coord2[2]) ** 2
        )

    def tim_kiem_khong_gian(
        self,
        query_vector: np.ndarray,
        toa_do_truy_van: Tuple[float, float, float],
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm lai (Hybrid Search).

        Args:
            query_vector: Vector ngữ nghĩa.
            toa_do_truy_van: Tọa độ gốc (X, Y, Z) để tính khoảng cách.
            top_k: Số lượng kết quả.
            alpha: Trọng số ngữ nghĩa (0.0 -> 1.0).
                   alpha=1.0: Chỉ tìm ngữ nghĩa.
                   alpha=0.0: Chỉ tìm theo khoảng cách vật lý.

        Returns:
            Danh sách kết quả.
        """
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).flatten()

        # Bước 1: Tính điểm ngữ nghĩa
        if self.khoang_cach == "cosine":
            semantic_scores = self._cosine_scores(query)
        elif self.khoang_cach == "l2":
            semantic_scores = self._l2_scores(query)
        else:
            semantic_scores = self._vectors @ query

        # Chuẩn hóa điểm ngữ nghĩa về [0, 1] nếu cần thiết
        if len(semantic_scores) > 0 and np.max(semantic_scores) > 0:
            semantic_scores = semantic_scores / np.max(semantic_scores)

        # Bước 2: Tính điểm không gian
        spatial_scores = np.zeros(len(self._ids))
        for i, meta in enumerate(self._metadata):
            toa_do_obj = meta.get("toa_do")
            if toa_do_obj:
                dist = self._tinh_khoang_cach_euclide(toa_do_truy_van, toa_do_obj)
                # Chuyển đổi khoảng cách thành điểm số tương đồng (Distance -> Similarity)
                # dist=0 -> score=1. dist=infinity -> score=0
                spatial_scores[i] = 1.0 / (1.0 + dist)
            else:
                spatial_scores[i] = 0.0

        # Bước 3: Hybrid Score
        final_scores = (alpha * semantic_scores) + ((1.0 - alpha) * spatial_scores)

        # Sắp xếp
        chi_so = np.argsort(final_scores)[::-1]

        ket_qua = []
        for idx in chi_so[:top_k]:
            idx = int(idx)
            ket_qua.append(
                {
                    "ma": self._ids[idx],
                    "diem_tong_hop": float(final_scores[idx]),
                    "diem_ngu_nghia": float(semantic_scores[idx]),
                    "diem_khong_gian": float(spatial_scores[idx]),
                    "metadata": self._metadata[idx],
                }
            )

        return ket_qua
