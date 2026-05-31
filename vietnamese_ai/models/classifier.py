"""Phân loại - Các thuật toán phân loại học máy."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from vietnamese_ai.models.base import BaseModel


class PhanLoai(BaseModel):
    """
    Bộ phân loại đa thuật toán.

    Hỗ trợ các thuật toán:
    - 'logistic': Hồi quy Logistic (mặc định)
    - 'knn': K-Nearest Neighbors
    - 'svm': Support Vector Machine
    - 'cay_quyet_dinh': Cây quyết định
    - 'rung_ngau_nhien': Rừng ngẫu nhiên
    - 'gradient_boosting': Gradient Boosting
    - 'naive_bayes': Naive Bayes

    Sử dụng:
        >>> phan_loai = PhanLoai(thuat_toan='logistic')
        >>> phan_loai.huan_luyen(X_train, y_train)
        >>> du_doan = phan_loai.du_doan(X_test)
        >>> diem = phan_loai.danh_gia(X_test, y_test)
    """

    THUAT_TOAN = {
        "logistic": LogisticRegression,
        "knn": KNeighborsClassifier,
        "svm": SVC,
        "cay_quyet_dinh": DecisionTreeClassifier,
        "rung_ngau_nhien": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "naive_bayes": GaussianNB,
    }

    def __init__(
        self,
        thuat_toan: str = "logistic",
        ten: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(ten or f"PhanLoai({thuat_toan})")

        if thuat_toan not in self.THUAT_TOAN:
            hop_le = ", ".join(self.THUAT_TOAN.keys())
            raise ValueError(f"Thuật toán '{thuat_toan}' không hợp lệ. Chọn một trong: {hop_le}")

        self.thuat_toan = thuat_toan
        self.tham_so = kwargs
        self._mo_hinh = self.THUAT_TOAN[thuat_toan](**kwargs)
        self._so_lop: Optional[int] = None

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        self._mo_hinh.fit(X, y)
        self._so_lop = len(np.unique(y))
        self.da_huan_luyen = True

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện. Gọi huan_luyen() trước.")
        return self._mo_hinh.predict(np.asarray(X))

    def du_doan_xac_suat(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất cho mỗi lớp."""
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        if hasattr(self._mo_hinh, "predict_proba"):
            return self._mo_hinh.predict_proba(np.asarray(X))
        raise AttributeError(f"Thuật toán '{self.thuat_toan}' không hỗ trợ predict_proba")

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        du_doan = self.du_doan(X)
        return float(accuracy_score(np.asarray(y), du_doan))

    def bao_cao(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Báo cáo chi tiết các chỉ số đánh giá."""
        y = np.asarray(y)
        du_doan = self.du_doan(X)
        trung_binh = "binary" if self._so_lop == 2 else "weighted"
        return {
            "do_chinh_xac": float(accuracy_score(y, du_doan)),
            "precision": float(precision_score(y, du_doan, average=trung_binh, zero_division=0)),
            "recall": float(recall_score(y, du_doan, average=trung_binh, zero_division=0)),
            "f1": float(f1_score(y, du_doan, average=trung_binh, zero_division=0)),
        }
