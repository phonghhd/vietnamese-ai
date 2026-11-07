"""Phân cụm - Các thuật toán học không giám sát."""

from typing import Any, Optional

import numpy as np

from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.utils.lazy_loader import _KMeansNumPy

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.cluster import KMeans as SKKMeans
    from sklearn.metrics import silhouette_score

    _CO_SKLEARN = True
except ImportError:
    _CO_SKLEARN = False
    SKKMeans = _KMeansNumPy
    DBSCAN = _KMeansNumPy
    AgglomerativeClustering = _KMeansNumPy


class PhanCum(BaseModel):
    """
    Bộ phân cụm đa thuật toán.

    Hỗ trợ các thuật toán:
    - 'kmeans': K-Means (mặc định)
    - 'dbscan': DBSCAN
    - 'hierarchical': Phân cụm phân cấp

    Sử dụng:
        >>> phan_cum = PhanCum(so_cum=3)
        >>> phan_cum.huan_luyen(X)
        >>> nhan = phan_cum.du_doan(X)
        >>> diem = phan_cum.danh_gia(X)
    """

    THUAT_TOAN = {
        "kmeans": SKKMeans,
        "dbscan": DBSCAN,
        "hierarchical": AgglomerativeClustering,
    }

    def __init__(
        self,
        so_cum: int = 3,
        thuat_toan: str = "kmeans",
        ten: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(ten or f"PhanCum({thuat_toan}, k={so_cum})")

        if thuat_toan not in self.THUAT_TOAN:
            hop_le = ", ".join(self.THUAT_TOAN.keys())
            raise ValueError(f"Thuật toán '{thuat_toan}' không hợp lệ. Chọn một trong: {hop_le}")

        self.so_cum = so_cum
        self.thuat_toan = thuat_toan
        self.tham_so = kwargs

        if thuat_toan == "dbscan":
            self._mo_hinh = DBSCAN(**kwargs)
        elif thuat_toan == "hierarchical":
            self._mo_hinh = AgglomerativeClustering(n_clusters=so_cum, **kwargs)
        else:
            self._mo_hinh = SKKMeans(n_clusters=so_cum, **kwargs)

        self._du_lieu_huan_luyen: Optional[np.ndarray] = None

    def huan_luyen(self, X: np.ndarray, y: np.ndarray = None) -> None:
        X = np.asarray(X)
        self._mo_hinh.fit(X)
        self._du_lieu_huan_luyen = X
        self.da_huan_luyen = True

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        X = np.asarray(X)
        if hasattr(self._mo_hinh, "predict"):
            return self._mo_hinh.predict(X)
        return self._mo_hinh.fit_predict(X)

    def danh_gia(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Đánh giá bằng Silhouette Score (-1 đến 1, càng cao càng tốt)."""
        nhan = self.du_doan(X)
        if len(set(nhan)) < 2:
            return -1.0
        if _CO_SKLEARN:
            return float(silhouette_score(np.asarray(X), nhan))
        # Tính Silhouette xấp xỉ thuần NumPy
        X_arr = np.asarray(X, dtype=float)
        dists = np.sqrt(np.sum((X_arr[:, None, :] - X_arr[None, :, :]) ** 2, axis=-1))
        scores = []
        cac_nhan = np.unique(nhan)
        for i in range(len(X_arr)):
            c_i = nhan[i]
            cung_cum = (nhan == c_i) & (np.arange(len(X_arr)) != i)
            a_i = float(np.mean(dists[i, cung_cum])) if np.any(cung_cum) else 0.0
            b_i = min(
                float(np.mean(dists[i, nhan == c_khac]))
                for c_khac in cac_nhan
                if c_khac != c_i and np.any(nhan == c_khac)
            )
            scores.append((b_i - a_i) / max(a_i, b_i, 1e-8))
        return float(np.mean(scores))

    def lay_tam_cum(self) -> Optional[np.ndarray]:
        """Lấy tọa độ tâm các cụm (chỉ áp dụng cho KMeans)."""
        if hasattr(self._mo_hinh, "cluster_centers_"):
            return self._mo_hinh.cluster_centers_
        return None
