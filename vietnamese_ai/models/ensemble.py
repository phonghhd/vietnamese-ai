"""Mô hình tập hợp - Ensemble methods."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    BaggingClassifier,
    BaggingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error

from vietnamese_ai.models.base import BaseModel


class MoHinhTapHop(BaseModel):
    """
    Mô hình tập hợp (Ensemble) kết hợp nhiều mô hình con.

    Hỗ trợ:
    - 'voting': Bỏ phiếu (mặc định)
    - 'bagging': Bagging
    - 'boosting': AdaBoost

    Sử dụng:
        >>> tap_hop = MoHinhTapHop(
        ...     loai='voting',
        ...     cac_mo_hinh=[
        ...         ('lr', LogisticRegression()),
        ...         ('rf', RandomForestClassifier()),
        ...     ]
        ... )
        >>> tap_hop.huan_luyen(X_train, y_train)
    """

    def __init__(
        self,
        loai: str = "voting",
        cac_mo_hinh: Optional[List] = None,
        nhiem_vu: str = "phan_loai",
        ten: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(ten or f"MoHinhTapHop({loai})")

        self.loai = loai
        self.nhiem_vu = nhiem_vu
        self.cac_mo_hinh = cac_mo_hinh or []

        if not cac_mo_hinh:
            raise ValueError("Cần cung cấp danh sách mô hình con (cac_mo_hinh)")

        self._mo_hinh = self._tao_mo_hinh(loai, nhiem_vu, cac_mo_hinh, kwargs)

    @staticmethod
    def _tao_mo_hinh(loai, nhiem_vu, cac_mo_hinh, kwargs):
        if loai == "voting":
            if nhiem_vu == "phan_loai":
                return VotingClassifier(estimators=cac_mo_hinh, **kwargs)
            return VotingRegressor(estimators=cac_mo_hinh, **kwargs)

        elif loai == "bagging":
            if nhiem_vu == "phan_loai":
                return BaggingClassifier(estimator=cac_mo_hinh[0][1], **kwargs)
            return BaggingRegressor(estimator=cac_mo_hinh[0][1], **kwargs)

        elif loai == "boosting":
            if nhiem_vu == "phan_loai":
                return AdaBoostClassifier(estimator=cac_mo_hinh[0][1], **kwargs)
            return AdaBoostRegressor(estimator=cac_mo_hinh[0][1], **kwargs)

        raise ValueError(f"Loại '{loai}' không hợp lệ. Chọn: voting, bagging, boosting")

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = np.asarray(X), np.asarray(y)
        self._mo_hinh.fit(X, y)
        self.da_huan_luyen = True

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        return self._mo_hinh.predict(np.asarray(X))

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        du_doan = self.du_doan(X)
        y = np.asarray(y)
        if self.nhiem_vu == "phan_loai":
            return float(accuracy_score(y, du_doan))
        return float(mean_squared_error(y, du_doan))
